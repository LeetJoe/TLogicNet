
import os
import json
import argparse
import concurrent.futures

from utils import merge_score, evaluate_avg


MERGE_HIT = 0
BASELINE_HIT = 1
BASELINE_MISS = 2

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluation for current EM epoch',
        usage='python em_eval.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', type=str, help='path to data file')
    parser.add_argument('--pred_kge', type=str, help='path kge prediction file')
    parser.add_argument('--pred_mln', type=str, help='path to mln test prediction file')
    parser.add_argument('--save_ranks', type=str, help='file path save ranks')
    parser.add_argument('--save_result', type=str, help='file path save result')
    parser.add_argument('--threads', type=int, default=20, help='number of threads(workers)')

    # parser.add_argument('--alpha', type=str, help='alpha')
    # parser.add_argument('--beta', type=str, help='beta')

    return parser.parse_args(args)


def load_filter_data(data_path):
    filter_data = set()
    for file in ['trainids.txt', 'validids.txt']:
        data_file = os.path.join(data_path, file)
        with open(data_file, 'r') as f:
            for line in f:
                line_split = line.split()
                h = int(line_split[0])
                r = int(line_split[1])
                t = int(line_split[2])
                ts = int(line_split[3])
                filter_data.add((h, r, t, ts))
            f.close()

    return filter_data


def calc_mln_baseline(facts):
    rel_total_num = {}  # {rel: facts number}
    rel_entity_num = {}  # {rel: ent: facts number}
    entity_num = {}  # {ent: number}
    for fact in facts:
        head = fact[0]
        rel = fact[1]
        tail = fact[2]
        if rel not in rel_total_num:
            rel_total_num[rel] = 0
        rel_total_num[rel] += 2  # head & tail

        if rel not in rel_entity_num:
            rel_entity_num[rel] = {}
        if head not in rel_entity_num[rel]:
            rel_entity_num[rel][head] = 0
        if tail not in rel_entity_num[rel]:
            rel_entity_num[rel][tail] = 0
        rel_entity_num[rel][head] += 1
        rel_entity_num[rel][tail] += 1

        if head not in entity_num:
            entity_num[head] = 0
        if tail not in entity_num:
            entity_num[tail] = 0
        entity_num[head] += 1
        entity_num[tail] += 1

    entity_num = sorted(entity_num.items(), key=lambda x: x[1], reverse=True)
    entity_max = entity_num[0][1]
    entity_num = {item[0]: item[1] for item in entity_num}

    entity_div = entity_max
    # entity_div = len(facts)

    entity_mid_idx = (len(entity_num) - 1) // 2
    i = 0
    entity_mid_score = 0.0
    for entity in entity_num:
        # entity_num[entity] /= (entity_div / 0.8)
        entity_num[entity] = ((entity_num[entity] + entity_div) / (2 * entity_div))  # smooth and make it greater than 0.5
        if i == entity_mid_idx:
            entity_mid_score = entity_num[entity]
        i = i + 1

    rel_mid_score = {}
    for rel in rel_entity_num:
        rel_entity_num[rel] = sorted(rel_entity_num[rel].items(), key=lambda x: x[1], reverse=True)
        rel_div = rel_entity_num[rel][0][1]  # for smooth
        rel_entity_num[rel] = {item[0]: item[1] for item in rel_entity_num[rel]}
        rel_entity_mid_idx = (len(rel_entity_num[rel]) + 1) // 2
        i = 0
        for ent in rel_entity_num[rel]:
            # rel_entity_num[rel][ent] /= (rel_div / 0.8)
            rel_entity_num[rel][ent] = ((rel_entity_num[rel][ent] + rel_div) / (rel_div * 2))
            if i == rel_entity_mid_idx:
                rel_mid_score[rel] = rel_entity_num[rel][ent]
            i = i + 1

    return entity_num, rel_entity_num, entity_mid_score, rel_mid_score


# load mln predictions of test
def load_pred_mln_old(pred_file):
    test_pred = {}
    with open(pred_file, 'r') as fr:
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

            query_key = (int(query_data[0]), int(query_data[1]), int(query_data[2]), int(query_data[3]), query_data[4])

            test_pred[query_key] = cands
        fr.close()

    return test_pred


# load mln predictions of test
def load_pred_mln(pred_file):
    test_pred = {}
    score_max = 0.0
    score_min = 10.0
    with open(pred_file, 'r') as fr:
        while True:
            line_query = fr.readline().strip()
            line_cands = fr.readline().strip()

            # prediction file contains all test data so the candidates line may be blank
            if (not line_query) and (not line_cands):
                break

            query_data = line_query.split('\t')
            cands = json.loads(line_cands)
            cands = {int(k) : float(v) for k, v in cands.items()}
            cands = dict(sorted(cands.items(), key=lambda item: item[1], reverse=True))
            if len(cands) > 0:
                key_list = list(cands.keys())
                if cands[key_list[0]] > score_max:
                    score_max = cands[key_list[0]]
                if cands[key_list[-1]] < score_min:
                    score_min = cands[key_list[-1]]

            query_key = (int(query_data[0]), int(query_data[1]), int(query_data[2]), int(query_data[3]), query_data[4])

            test_pred[query_key] = cands
        fr.close()

    for query_key in test_pred:
        if len(test_pred[query_key]) > 0:
            for k in test_pred[query_key]:
                test_pred[query_key][k] = (test_pred[query_key][k] - score_min) / (score_max - score_min)

    return test_pred


# load mln predictions of hidden
def load_pred_hidden(pred_file):
    hidden_pred = {}
    with open(pred_file, 'r') as f:
        for line in f:
            line_split = line.split()
            score = float(line_split[5])
            if score == -1:
                continue

            hidden_key = (int(line_split[0]), int(line_split[1]), int(line_split[2]), line_split[4])
            if hidden_key not in hidden_pred:
                hidden_pred[hidden_key] = {}

            hidden_pred[hidden_key][int(line_split[3])] = score
        f.close()

    return hidden_pred


def get_rank_thread(fact_list, kge_candidates_list, mln_pred, filter_data, entity_baseline, rel_baseline, entity_mid_score, rel_mid_scores):
    batch_result = {}
    for i in range(len(fact_list)):
        fact = fact_list[i]
        kge_candidates = kge_candidates_list[i]
        fact = fact.strip().split('\t')
        h = int(fact[0])
        r = int(fact[1])
        t = int(fact[2])
        ts = int(fact[3])
        task = fact[4]
        kge_rank = int(fact[5])
        kge_candidates = kge_candidates.strip().split(';')
        kge_candidates = {int(pred.split('*')[0]): float(pred.split('*')[1]) for pred in kge_candidates}
        # kge_candidates = {cand: score for cand, score in kge_candidates.items()}

        if task == 'sp':
            target = t
        else:
            target = h

        filter_idx = set()
        query_key = (h, r, t, ts, task)
        # merge_candidates = kge_candidates.copy()
        merge_candidates = {}
        for cand in kge_candidates:
            merge_candidates[cand] = kge_candidates[cand]
            if kge_candidates[cand] < kge_candidates[target] - 0.01:
                break

        if query_key in mln_pred and len(mln_pred[query_key]) != 0:
            mln_candidates = mln_pred[query_key]
            for cand in mln_candidates:
                if cand not in merge_candidates:
                    merge_candidates[cand] = kge_candidates[cand]

            for cand, score in merge_candidates.items():
                if cand in mln_candidates:
                    merge_candidates[cand] = merge_score(merge_candidates[cand], mln_candidates[cand])

        for cand, score in merge_candidates.items():
            if task == 'sp':
                test_key = (h, r, cand, ts)
            else:
                test_key = (cand, r, t, ts)
            if test_key in filter_data and cand != target:
                filter_idx.add(cand)

        rank_candidates = merge_candidates.copy()

        if target not in rank_candidates:
            hit_type = BASELINE_HIT
            baseline_candidates = entity_baseline.copy()
            if r in rel_baseline:
                rel_candidates = rel_baseline[r].copy()
                rel_mid_score = rel_mid_scores[r]
            else:
                rel_candidates = {}
                rel_mid_score = 0
            if target not in baseline_candidates:
                hit_type = BASELINE_MISS
                baseline_candidates[target] = entity_mid_score
                if len(rel_candidates) > 0:
                    rel_candidates[target] = rel_mid_score

            if target in rel_candidates:
                rank_candidates = rel_candidates
            else:
                rank_candidates = baseline_candidates

        assert(target in rank_candidates)

        for cand in filter_idx:
            if cand in rank_candidates:
                rank_candidates.pop(cand)

        rank_candidates = sorted(rank_candidates.items(), key=lambda x: x[1], reverse=True)
        rank_candidates = {candidate[0]: candidate[1] for candidate in rank_candidates}
        mln_rank = get_rank(rank_candidates, target)
        batch_result[query_key] = (kge_rank, mln_rank)

    return batch_result


# load pred kge and update rand (pred_kge is too huge to load totally)
def get_ranks_task(pred_kge_file, mln_pred, filter_data, threads=10):
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=threads)
    ranks = {}
    entity_baseline, rel_baseline, entity_mid_score, rel_mid_score = calc_mln_baseline(filter_data)

    query_num = len(mln_pred)
    batch_size = max(query_num // 100 + 1, 300)
    with open(pred_kge_file, 'r') as fr:
        stop = False
        while not stop:
            future_to_sidx = {}
            for i in range(threads):
                batch_cursor = 0
                fact_batch = []
                kge_candidates_batch = []
                while batch_cursor < batch_size:
                    fact = fr.readline()
                    kge_candidates = fr.readline()

                    if (not fact) and (not kge_candidates):
                        stop = True
                        break
                    else:
                        batch_cursor += 1
                        fact_batch.append(fact)
                        kge_candidates_batch.append(kge_candidates)
                if len(fact_batch) > 0:
                    ret = executor.submit(get_rank_thread, fact_batch, kge_candidates_batch, mln_pred, filter_data, entity_baseline, rel_baseline, entity_mid_score, rel_mid_score)
                    future_to_sidx[ret] = i

                if stop:
                    break

            if len(future_to_sidx) > 0:
                for future in concurrent.futures.as_completed(future_to_sidx):
                    batch_ranks = future.result()
                    ranks.update(batch_ranks)
                    # if kge_rank != mln_rank:
                    #     if hit_type == BASELINE_HIT:
                    #         mark = '[baseline hit]'
                    #     elif hit_type == BASELINE_MISS:
                    #         mark = '[baseline miss]'
                    #     else:
                    #         mark = ''
                    #     print('({}, {}, {}, {}, {} => {}), {}, {} {}'.format(
                    #         *query_key, target, kge_rank, mln_rank, mark))

                print(' -Progress: {}/{}         '.format(len(ranks), query_num), end='\r')

        executor.shutdown(wait=True)
        print()
        fr.close()
    return ranks


def get_rank(candidates: dict, target):
    target_score = candidates[target]
    num_better = 0
    num_same = 0
    for cand, score in candidates.items():
        if score > target_score:
            num_better += 1
        elif score == target_score:
            num_same += 1
        else:
            break

    return num_better + (num_same + 1) // 2


# todo delete this
'''
# 下面这种设置实测结果并不好，结果比 kge 本身甚至还要更差，不要再往这个方面尝试了；
# 也就是说，只有 mln score 超过 0.9999 那部分才值得以 0.1 权重跟 kge score 合并，其它分段的都不值得给 0.1 的权重，最多给到 0.05；
def merge_mln_score_bad(kge_score, mln_score):
    if mln_score > 0.9999:
        use_score = kge_score * 0.9 + mln_score * 0.1
    elif mln_score > 0.95:
        use_score = kge_score * 0.91 + mln_score * 0.09
    elif mln_score > 0.7:
        use_score = kge_score * 0.95 + mln_score * 0.05
    else:
        use_score = kge_score * 0.999 + mln_score * 0.001
    return use_score
'''

def get_hidden_score(ts_scores, query_ts, w):
    use_ts = -1
    use_score = 0
    for ts, score in ts_scores.items():
        if use_ts == -1:
            use_ts = ts
            use_score = score
        elif abs(ts - query_ts) < abs(use_ts - use_score):
            use_ts = ts
            use_score = score

    if abs(use_ts - use_score) < w:
        return use_score
    else:
        return use_score * w / abs(use_ts - use_score)


def save_ranks(ranks, save_file):
    with open(save_file, 'w') as f:
        f.write('query\tdirection\tkge_rank\tem_rank\n')
        for query, ranks in ranks.items():
            f.write('{},{},{},{}\t{}\t{}\t{}\n'.format(
                *query[:4], query[4], *ranks
            ))
        f.close()


def save_result(result_kge, result_em, save_file):
    with open(save_file, 'w') as f:
        f.write('kge:\nMR: {}, MRR: {}, Hit@1: {}, Hit@3: {}, Hit@10: {}\n\n'.format(*result_kge.values()))
        f.write('kge+mln:\nMR: {}, MRR: {}, Hit@1: {}, Hit@3: {}, Hit@10: {}\n\n'.format(*result_em.values()))
    f.close()


def main(args=None):
    print('Evaluating on EM result...')
    filter_data = load_filter_data(args.data_path)
    pred_mln = load_pred_mln(args.pred_mln)
    # pred_hidden = load_pred_hidden(args.pred_hidden)

    ranks = get_ranks_task(args.pred_kge, pred_mln, filter_data, args.threads)
    save_ranks(ranks, args.save_ranks)
    ranks_kge = {}
    ranks_em = {}
    for q, r in ranks.items():
        ranks_kge[q] = r[0]
        ranks_em[q] = r[1]

    result_kge = evaluate_avg(ranks_kge)
    result_em = evaluate_avg(ranks_em)
    save_result(result_kge, result_em, args.save_result)
    # print(result['mln'])
    print('Evaluation on EM finished.')


if __name__ == '__main__':
    main(parse_args())
