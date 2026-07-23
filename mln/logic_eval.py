#!/usr/bin/python3

import os
import json
import argparse

import tlogic_train
import concurrent.futures

import utils

test_data = []
predictions = []
filter_id_map = []

base_dist = {}
base_rel_dist = {}


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='MLN evaluation.',
        usage='./mln/logic_eval.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', type=str, help='path to dataset')
    parser.add_argument('--prediction', type=str, help='file path to prediction file')
    parser.add_argument('--save', type=str, help='file path to save evaluation')
    parser.add_argument("--threads", type=int, default=8, help="number of parallel processes")

    return parser.parse_args(args)


# old java output format
def load_predictions_old(file):
    query_with_cands = []
    with open(file, 'r') as fr:
        while True:
            line_query = fr.readline().strip()
            line_cands = fr.readline().strip()

            # prediction file contains all test data so the candidates line may be blank
            if (not line_query) and (not line_cands):
                break

            query_data = line_query.split('\t')
            cands = {}
            if len(line_cands) > 0:
                cands_list = [cand_pair.split('*') for cand_pair in line_cands.split(';')]
                for cand_pair in cands_list:
                    if len(cand_pair) == 2:
                        cands[int(cand_pair[0])] = float(cand_pair[1])

            query_data.append(cands)
            query_with_cands.append(query_data)

    return query_with_cands


# json candidates
def load_predictions(file):
    query_with_cands = []
    with open(file, 'r') as fr:
        while True:
            line_query = fr.readline().strip()
            line_cands = fr.readline().strip()

            # prediction file contains all test data so the candidates line may be blank
            if (not line_query) and (not line_cands):
                break

            query_data = line_query.split('\t')
            cands_list = json.loads(line_cands)
            cands_list = {int(k) : float(v) for k, v in cands_list.items()}
            cands_list = dict(sorted(cands_list.items(), key=lambda item: item[1], reverse=True))

            query_data.append(cands_list)
            query_with_cands.append(query_data)

    return query_with_cands


def task_evaluate(num_threads):
    global predictions
    rank_dict = {}  # query id: rank
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            total_count = len(predictions)
            stop = False
            qid = 0
            batch_size = 500
            while not stop:
                future_to_sidx = {}
                for i in range(num_threads):
                    end_qid = min(total_count, qid + batch_size)
                    if qid >= end_qid:
                        stop = True
                        break
                    ret = executor.submit(get_rank, qid, end_qid, 'avg', True)
                    qid += batch_size
                    future_to_sidx[ret] = i
                if len(future_to_sidx) > 0:
                    for future in concurrent.futures.as_completed(future_to_sidx):
                        batch_ranks = future.result()
                        rank_dict.update(batch_ranks)
                        print('[Calculate ranks]Progress: {}/{}         '.format(len(rank_dict), total_count), end='\r')

            print()
        executor.shutdown(wait=True)
    except Exception as e:
        print(e)
    return rank_dict


def get_rank(qid, qid_end, rtype='avg', do_filter=True):
    global filter_id_map, base_dist, base_rel_dist, predictions

    result = {}
    for i in range(qid, qid_end):
        quad = predictions[i]

        head = int(quad[0])
        rel = int(quad[1])
        tail = int(quad[2])
        query_ts = int(quad[3])
        task = quad[4]

        if task == 'sp':
            target = tail
        else:
            target = head

        if target in quad[5]:
            cands_dict = dict(sorted(quad[5].items(), key=lambda item: item[1], reverse=True))
            # cands_dict = quad[5]
        elif rel in base_rel_dist and target in base_rel_dist[rel]:
            cands_dict = base_rel_dist[rel].copy()
        else:
            cands_dict = base_dist.copy()

        if target not in cands_dict:
            # 个别确实没在 object 位置上出现过的实体会进入此分支
            result[tuple(quad[:5])] = len(base_dist) // 2
            continue

        cands_to_pop = []
        if do_filter:
            for cand in cands_dict:
                if cand == target:
                    break
                quad_key = (head, rel, tail, query_ts)
                if quad_key in filter_id_map:
                    # 不能在这里直接 pop，会影响循环
                    cands_to_pop.append(cand)

        for c in cands_to_pop:
            cands_dict.pop(c)

        target_score = cands_dict[target]

        same_count = 0
        top_rank = 0
        for cand in cands_dict:
            if cands_dict[cand] > target_score:
                top_rank += 1
            elif cands_dict[cand] == target_score:
                same_count += 1
            elif cands_dict[cand] < target_score:
                break

        if rtype == 'best':
            final_rank = top_rank
        elif rtype == 'worst':
            final_rank = top_rank + same_count
        else:
            final_rank = top_rank + (same_count + 1) // 2

        result[tuple(quad[:5])] = final_rank

    return result


def eval_and_save(ranks, file):
    mr = 0
    mrr = 0
    total = 0
    hit_1 = 0
    hit_3 = 0
    hit_10 = 0

    for qid in ranks:
        rk = ranks[qid]
        if rk <= 0:
            continue

        mr += rk
        mrr += 1 / rk
        total += 1
        if rk <= 1:
            hit_1 += 1
            hit_3 += 1
            hit_10 += 1
        elif rk <= 3:
            hit_3 += 1
            hit_10 += 1
        elif rk <= 10:
            hit_10 += 1

    mr = round(mr / total, 6)
    mrr = round(mrr / total, 6)
    hit_1_rate = round(hit_1 / total, 6)
    hit_3_rate = round(hit_3 / total, 6)
    hit_10_rate = round(hit_10 / total, 6)

    with open(file, 'w') as f:
        f.write('total: {}\tMR: {}\tMRR: {}\tHits@1: {}\tHits@3: {}\tHits@10: {}\n\n'.format(
            total, mr, mrr, hit_1_rate, hit_3_rate, hit_10_rate))

        ranks = dict(sorted(ranks.items(), key=lambda item: item[1]))
        for dk in ranks:
            rk = ranks[dk]
            f.write('{}\t{}\n'.format(dk, rk))
        f.close()


def main(args):
    global test_data, filter_id_map, predictions, base_dist, base_rel_dist

    tlogic_train.load_data(os.path.join(args.data_path, 'stat.txt'), os.path.join(args.data_path, 'testids.txt'))
    tlogic_train.init_quad_map()
    print("#Test quadruples (doubled): %d" % len(tlogic_train.quadruples))
    test_data = tlogic_train.quadruples.copy()

    # clear for learn data
    tlogic_train.quadruples = []
    tlogic_train.quadruple_id_map.clear()
    tlogic_train.quadruple_id_map = {}

    tlogic_train.load_data(os.path.join(args.data_path, 'stat.txt'), os.path.join(args.data_path, 'trainids.txt'), os.path.join(args.data_path, 'validids.txt'))
    tlogic_train.init_quad_map()
    filter_id_map = tlogic_train.quadruple_id_map.copy()
    print("#Train+Valid quadruples (doubled): %d" % len(tlogic_train.quadruples))

    base_dist, base_rel_dist = utils.get_base_dist_v2(tlogic_train.quadruples)
    print("#Baseline calculated.")

    predictions = load_predictions(args.prediction)
    print("#Prediction data loaded: %d" % len(predictions))

    rank_dict = task_evaluate(min(args.threads, 10))

    eval_and_save(rank_dict, args.save)


if __name__ == '__main__':
    main(parse_args())
