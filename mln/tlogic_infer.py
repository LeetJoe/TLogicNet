#!/usr/bin/python3

import argparse
import math
import numpy as np
import json
import concurrent.futures

"""
已获得 test 数据与逻辑规则的关联关系，只需要进行打分即可。
"""

rules = {}
relation_size = 0
origin_test = []
linked_test = []


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='MLN test inference.',
        usage='./mln/tlogic_infer.py [<args>] [-h | --help]'
    )

    parser.add_argument('--stat', type=str, help='file path to stat.txt')
    parser.add_argument('--test', type=str, help='file path to testids.txt')
    parser.add_argument('--ltest', type=str, help='file path to testids_linked.txt')
    parser.add_argument('--rule', type=str, help='file path to rules.txt')
    parser.add_argument('--weight', type=float, default=-1, help='rule weight threshold')
    parser.add_argument('--save', type=str, help='file path to save candidates')
    parser.add_argument("--threads", type=int, default=4, help="number of parallel processes")

    return parser.parse_args(args)


def load_test(stat_file, test_file):
    global origin_test, relation_size

    with open(stat_file, 'r') as fr:
        line = fr.readline()
        line_split = line.split()
        relation_size = int(line_split[1])
        fr.close()

    with open(test_file, 'r') as fr:
        i = 0
        for line in fr:
            line_split = line.split()
            h = int(line_split[0])
            r = int(line_split[1])
            t = int(line_split[2])
            ts = int(line_split[3])
            origin_test.append([h, r, t, ts])
        fr.close()


# load rules [id, head, body, constraints, precision, head_support, body_support, weight]
def load_rules(rule_file):
    global rules
    with open(rule_file, 'r') as fr:
        for entry in fr:
            line_split = entry.strip().split('\t')
            rules[int(line_split[0])] = [int(line_split[0]), line_split[1], line_split[2], line_split[3], float(line_split[4]), line_split[5], line_split[6], float(line_split[7])]  # weight

        fr.close()


def load_linked_test(test_file):
    global linked_test
    with open(test_file, 'r') as fr:
        for line in fr:
            line_split = line.strip().split('\t')
            if len(line_split) != 6:
                print(line)
            linked_test.append([
                int(line_split[0]),
                int(line_split[1]),
                int(line_split[2]),
                int(line_split[3]),
                line_split[4],
                json.loads(line_split[5]),
            ])
        fr.close()


def task_score_test(num_threads):
    score_dict = {}  # query id: cand_dict
    for idx in range(len(linked_test)):
        query = linked_test[idx][0:5]
        cand_metric = json.loads(linked_test[idx][5])
        print('[infer test data]Progress: {}/{}         '.format(idx, len(linked_test)), end='\r')
    return score_dict


def task_score(num_threads):
    global rules, linked_test
    score_dict = {}  # query id: cand_dict
    batch_size = 100
    batch_num = math.floor(len(linked_test) / batch_size)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        future_to_sidx = {executor.submit(batch_scores, bid, batch_size): bid
                          for bid in range(batch_num)}
        i = 0
        for future in concurrent.futures.as_completed(future_to_sidx):
            # sidx = future_to_sidx[future]
            batch_score_dict = future.result()
            score_dict.update(batch_score_dict)
            i += 1
            print('[infer test data]Progress: {}/{}         '.format(i, batch_num), end='\r')
        print()
    executor.shutdown(wait=True)
    return score_dict


def batch_scores(qid, batch_size):
    global linked_test
    start_idx = qid * batch_size
    end_idx = min((qid + 1) * batch_size, len(linked_test))
    batch_score_dict = {}
    for idx in range(start_idx, end_idx):
        cand_metric = linked_test[idx][5]
        batch_score_dict[idx] = {}
        for cand in cand_metric:
            batch_score_dict[idx][cand] = get_score_w(cand_metric.get(cand), int(linked_test[idx][3]))
        batch_score_dict[idx] = dict(sorted(batch_score_dict[idx].items(), key=lambda item: item[1], reverse=True))

    return batch_score_dict


# Tlogic like
def get_score(rule_times, query_time, lmbda=0.1, coeff=0.5, topk=20):
    global rules
    noisyProd = 1
    scores = []
    for rid in rule_times.keys():
        score_ts = np.exp(
            lmbda * (max(rule_times.get(rid)) - query_time)
        )
        score_rule = rules[int(rid)][4]  # precision
        scores.append(coeff * score_rule + (1 - coeff) * score_ts)
        # noisyProd *= (1 - (coeff * score_rule + (1 - coeff) * score_ts))

    if len(scores) > topk:
        scores.sort(reverse=True)
        scores = scores[:topk]

    for s in scores:
        noisyProd *= (1 - s)

    return 1 - noisyProd

# use weight
def get_score_w(rule_times, query_time, lmbda=0.1, coeff=0.4, topk=20):
    global rules
    noisyProd = 1
    scores = []
    for rid in rule_times.keys():
        score_ts = np.exp(
            lmbda * (rule_times.get(rid) - query_time)
        )
        score_rule = 1 if rules[int(rid)][7] > 1.5 else sigmoid(rules[int(rid)][7])  # weight
        scores.append(coeff * score_rule + (1 - coeff) * score_ts)
        # noisyProd *= (1 - (coeff * score_rule + (1 - coeff) * score_ts))

    if len(scores) > topk:
        scores.sort(reverse=True)
        scores = scores[:topk]

    for s in scores:
        noisyProd *= (1 - s)

    return 1 - noisyProd


def save_candidate(save_file, candidates):
    global linked_test, origin_test

    query_cands = {}
    for i in candidates.keys():
        query_key = tuple(linked_test[i][0:5])
        query_cands[query_key] = candidates[i]

    # inference result with inverse relations
    with open(save_file, 'w') as fd:
        for i in range(len(origin_test)):
            for dir in ['sp', 'po']:
                query_key = (*origin_test[i], dir)
                if query_key in query_cands:
                    cand_line = json.dumps(query_cands[query_key])
                else:
                    cand_line = "{}"
                fd.write('{}\t{}\t{}\t{}\t{}\n{}\n'.format(*query_key, cand_line))
        fd.close()


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def main(args):
    print("MLN test inference&evaluation started.")
    load_rules(args.rule)
    load_test(args.stat, args.test)
    load_linked_test(args.ltest)

    scored_candidates = task_score(args.threads)
    save_candidate(args.save, scored_candidates)

    print("Inference for test data done.")


if __name__ == '__main__':
    main(parse_args())
