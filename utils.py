import os
import numpy as np


def get_stat(data_path):
    with open(os.path.join(data_path, 'stat.txt'), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_map(data_path):
    results = []
    files = ['entitymap.txt', 'relationmap.txt', 'timemap.txt']

    for file in files:
        result = {}
        with open(os.path.join(data_path, file), 'r') as fr:
            for line in fr:
                split = line.strip().split('\t')
                result[split[0]] = split[1]
            results.append(result)
        fr.close()

    return results


def merge_score_v0(kge_score, mln_score):
    if mln_score > 0.9999:
        use_score = kge_score * 0.9 + mln_score * 0.1
    elif mln_score > 0.7:
        use_score = kge_score * 0.95 + mln_score * 0.05
    else:
        use_score = kge_score * 0.999 + mln_score * 0.001
    return use_score


def merge_score(kge_score, mln_score):
    lmda = -np.cos(0.3 * mln_score) + 1
    use_score = kge_score * (1 - lmda) + mln_score * lmda
    return use_score


def evaluate(ranks: dict):
    result = {
        'mr': 0.0,
        'mrr': 0.0,
        'hit@1': 0.0,
        'hit@3': 0.0,
        'hit@10': 0.0
    }

    total_num = len(ranks)
    for rank in ranks.values():
        result['mr'] += rank
        result['mrr'] += 1.0/rank

        if rank <= 1:
            result['hit@1'] += 1
            result['hit@3'] += 1
            result['hit@10'] += 1
        elif rank <= 3:
            result['hit@3'] += 1
            result['hit@10'] += 1
        elif rank <= 10:
            result['hit@10'] += 1

    for metric in ['mr', 'mrr', 'hit@1', 'hit@3', 'hit@10']:
        result[metric] /= total_num

    return result


def evaluate_avg(ranks: dict):
    result = {'sp': {}, 'po': {}, 'spo': {}}
    for spo in ['sp', 'po', 'spo']:
        result[spo] = {
            'mr': 0.0,
            'mrr': 0.0,
            'hit@1': 0.0,
            'hit@3': 0.0,
            'hit@10': 0.0
        }

    total_num = len(ranks) // 2
    for query, rank in ranks.items():
        spo = query[4]
        result[spo]['mr'] += rank
        result[spo]['mrr'] += 1.0/rank

        if rank <= 1:
            result[spo]['hit@1'] += 1
            result[spo]['hit@3'] += 1
            result[spo]['hit@10'] += 1
        elif rank <= 3:
            result[spo]['hit@3'] += 1
            result[spo]['hit@10'] += 1
        elif rank <= 10:
            result[spo]['hit@10'] += 1

    for spo in ['sp', 'po']:
        for metric in ['mr', 'mrr', 'hit@1', 'hit@3', 'hit@10']:
            result[spo][metric] /= total_num

    for metric in ['mr', 'mrr', 'hit@1', 'hit@3', 'hit@10']:
        result['spo'][metric] = (result['sp'][metric] + result['po'][metric]) / 2.0

    return result['spo']