import json
import os
from collections import Counter

import numpy as np


class Rule:
    id = -1
    rule_support = 0
    body_support = 1
    precision = 0
    weight = 0
    grad = 0
    mid_gap = 0
    gap_list = []
    hops = None

    def __init__(self, rule_id, rule_type: int, rule_premise: list, rule_hypothesis: int, rule_constraints=None):
        self.id = rule_id
        self.rule_type = rule_type
        self.rule_premise = rule_premise
        self.rule_hypothesis = rule_hypothesis
        self.rule_constraints = rule_constraints


RULE_TYPE_COMPOSITION = 0
RULE_TYPE_SYMMETRIC = 1
RULE_TYPE_INVERSE = 2
RULE_TYPE_SUBRELATION = 3
RULE_TYPE_TLOGIC = 4

QUADRUPLE_TYPE_OBSERVED = 1
QUADRUPLE_TYPE_HIDDEN = 0

QUADRUPLE_VALID = 1
QUADRUPLE_INVALID = 0

ROUND_DECIMAL = 8

triplet_mts_map = {}
triplet_tss_map = {}
triplet_tss_list_map = {}


# deprecated
def find_hthop(rel_quadruples, rule: Rule, valid_only=False):
    """
    find first_head->last_tail hop matching given rule
    :param rel_quadruples:
    :param rule:
    :param valid_only:
    :return:
        返回结果组织为 head:[[tail1, ts1], [tail2, ts2], ...]
    """
    ht_hop = {}
    init = True
    for rp in rule.rule_premise:
        jump_dict = {}
        for i in range(len(rel_quadruples[rp])):
            h = rel_quadruples[rp][i][0]
            t = rel_quadruples[rp][i][2]
            ts = rel_quadruples[rp][i][3]
            valid = rel_quadruples[rp][i][5]
            if valid_only and valid == QUADRUPLE_INVALID:
                continue
            if h in jump_dict:
                jump_dict[h].append([t, ts])
            else:
                jump_dict[h] = [[t, ts]]
        if init:
            ht_hop = jump_dict.copy()
            init = False
        else:
            new_ht_hop = {}
            for ck, cv_list in ht_hop.items():
                for cv in cv_list:
                    if cv[0] in jump_dict:
                        for next_cv in jump_dict[cv[0]]:
                            if next_cv[1] >= cv[1]:
                                if ck in new_ht_hop:
                                    if next_cv not in new_ht_hop[ck]:
                                        new_ht_hop[ck].append(next_cv)
                                else:
                                    new_ht_hop[ck] = [next_cv]
            ht_hop = new_ht_hop
    return ht_hop


# deprecated
def find_hthop_mints(rel_quadruples, rule: Rule, valid_only=False):
    """
    find first_head->last_tail hop matching given rule
    :param rel_quadruples:
    :param rule:
    :param valid_only:
    :return:
        返回结果组织为 head:{tail: min_ts} 如果 tail 多次出现，min_ts 中记录最小的那个
    """
    ht_hop = {}  # {head: {tail: min_ts}}
    init = True
    for rp in rule.rule_premise:
        jump_dict = {}
        for i in range(len(rel_quadruples[rp])):
            h = rel_quadruples[rp][i][0]
            t = rel_quadruples[rp][i][2]
            ts = rel_quadruples[rp][i][3]
            valid = rel_quadruples[rp][i][5]
            if valid_only and valid == QUADRUPLE_INVALID:
                continue
            if h not in jump_dict:
                jump_dict[h] = {}
            if t not in jump_dict[h]:
                jump_dict[h][t] = set()
            jump_dict[h][t].add(ts)
        if init:
            for hk in jump_dict:
                min_ts = -1
                for tk in jump_dict[hk]:
                    for ts in jump_dict[hk][tk]:
                        if min_ts < 0 or ts < min_ts:
                            min_ts = ts
                    if hk not in ht_hop:
                        ht_hop[hk] = {}
                    ht_hop[hk][tk] = min_ts
            init = False
        else:
            new_ht_hop = {}
            for hk in ht_hop:
                for tk in ht_hop[hk]:
                    min_ts = ht_hop[hk][tk]
                    if tk in jump_dict:
                        for ntk in jump_dict[tk]:
                            for nts in jump_dict[tk][ntk]:
                                if nts >= min_ts:
                                    if hk not in new_ht_hop:
                                        new_ht_hop[hk] = {}
                                    if (ntk not in new_ht_hop[hk]) or (nts < new_ht_hop[hk][ntk]):
                                        new_ht_hop[hk][ntk] = nts
            ht_hop = new_ht_hop
    return ht_hop


def find_hthop_v2(rel_quadruples, rule: Rule):
    path, path_ts = find_rule_body(rel_quadruples, rule)
    hthop = {}
    for i in range(len(path)):
        h = path[i][0]
        t = path[i][-1]
        ts = path_ts[i][1]
        if h not in hthop:
            hthop[h] = []
        hthop[h].append([t, ts])

    return hthop


# add early_time
def find_hthop_v3(rel_quadruples, rule: Rule):
    path, path_ts = find_rule_body(rel_quadruples, rule)
    hthop = {}
    for i in range(len(path)):
        h = path[i][0]
        t = path[i][-1]
        ets = path_ts[i][0]
        lts = path_ts[i][1]
        if h not in hthop:
            hthop[h] = {}
        if t not in hthop[h]:
            hthop[h][t] = []
        hthop[h][t].append([ets, lts])

    return hthop


def find_hthop_late_mints_v1(rel_quadruples, rule: Rule):
    path, path_ts = find_rule_body(rel_quadruples, rule)
    hthop = {}
    for i in range(len(path)):
        h = path[i][0]
        t = path[i][-1]
        ts = path_ts[i][1]  # 时间是路径最晚时间中的最小值
        if h not in hthop:
            hthop[h] = {}
        if (t not in hthop[h]) or (ts < hthop[h][t]):
            hthop[h][t] = ts

    return hthop


def find_hthop_early_maxts_v1(rel_quadruples, rule: Rule):
    path, path_ts = find_rule_body(rel_quadruples, rule)
    hthop = {}
    for i in range(len(path)):
        h = path[i][0]
        t = path[i][-1]
        ts = path_ts[i][0]  # 时间是路径最早时间中的最大值
        if h not in hthop:
            hthop[h] = {}
        if (t not in hthop[h]) or (ts > hthop[h][t]):
            hthop[h][t] = ts

    return hthop


def find_hthop_emlm(rel_quadruples, rule: Rule):
    path, path_ts = find_rule_body(rel_quadruples, rule)
    hthop = {}
    for i in range(len(path)):
        h = path[i][0]
        t = path[i][-1]
        ts = path_ts[i]  # 两个数字的列表，第一个时间是路径最早时间中的最大值，第二个是路径最晚时间中的最小值
        if h not in hthop:
            hthop[h] = {}
        if t not in hthop[h]:
            hthop[h][t] = ts
        else:
            if ts[0] > hthop[h][t][0]:
                hthop[h][t][0] = ts[0]
            if ts[1] < hthop[h][t][1]:
                hthop[h][t][1] = ts[1]

    return hthop


# 考虑 var_constraints 的情况
def find_rule_body(rel_quadruples, rule: Rule, valid_only=True):
    all_jump = []
    for rp in rule.rule_premise:
        jump_dict = {}
        for i in range(len(rel_quadruples[rp])):
            h = rel_quadruples[rp][i][0]
            t = rel_quadruples[rp][i][2]
            ts = rel_quadruples[rp][i][3]
            valid = rel_quadruples[rp][i][5]
            if valid_only and valid == QUADRUPLE_INVALID:
                continue
            if h not in jump_dict:
                jump_dict[h] = {}
            if t not in jump_dict[h]:
                jump_dict[h][t] = set()
            jump_dict[h][t].add(ts)
        all_jump.append(jump_dict)

    path = []
    path_early_ts = []
    path_late_ts = []

    for h0 in all_jump[0]:
        for t0 in all_jump[0][h0]:
            for ts0 in all_jump[0][h0][t0]:
                path.append([h0, t0])
                path_early_ts.append(ts0)
                path_late_ts.append(ts0)

    for i in range(1, len(all_jump)):
        cur_jump = all_jump[i]
        new_path = []
        new_path_dict = {}
        new_path_early_ts = []
        new_path_late_ts = []
        for j in range(len(path)):
            cur_path = path[j]
            cur_path_tail = cur_path[-1]
            if cur_path_tail in cur_jump:
                for jump_tail in cur_jump[cur_path_tail]:
                    temp_path = path[j].copy()
                    temp_path.append(jump_tail)
                    temp_path_key = temp_path.copy()
                    temp_path_key.append(path_early_ts[j])

                    if tuple(temp_path_key) not in new_path_dict:
                        pre_late_ts = path_late_ts[j]
                    else:
                        pre_late_ts = new_path_dict[tuple(temp_path_key)]

                    larger_min_ts = -1
                    for jump_ts in cur_jump[cur_path_tail][jump_tail]:
                        if jump_ts >= pre_late_ts:
                            if larger_min_ts == -1 or larger_min_ts > jump_ts:
                                larger_min_ts = jump_ts

                    if larger_min_ts == -1:
                        continue

                    if tuple(temp_path_key) not in new_path_dict:
                        new_path.append(temp_path)
                        new_path_early_ts.append(path_early_ts[j])

                    new_path_dict[tuple(temp_path_key)] = larger_min_ts

        path = new_path
        path_early_ts = new_path_early_ts

        for k in range(len(path)):
            path_key = path[k].copy()
            path_key.append(path_early_ts[k])
            new_path_late_ts.append(new_path_dict[tuple(path_key)])
        path_late_ts = new_path_late_ts

    if len(rule.rule_constraints) > 0:
        for const in rule.rule_constraints:
            for i in range(len(const) - 1):
                new_path = []
                new_path_early_ts = []
                new_path_late_ts = []
                for j in range(len(path)):
                    if path[j][const[i]] == path[j][const[i+1]]:
                        new_path.append(path[j])
                        new_path_early_ts.append(path_early_ts[j])
                        new_path_late_ts.append(path_late_ts[j])
                path = new_path
                path_early_ts = new_path_early_ts
                path_late_ts = new_path_late_ts

    return path, list(zip(path_early_ts, path_late_ts))


def add_triplet_map(triplet_key: str, ts: int):
    global triplet_mts_map, triplet_tss_list_map
    if triplet_key in triplet_mts_map:
        if ts > triplet_mts_map[triplet_key]:
            triplet_mts_map[triplet_key] = ts
    else:
        triplet_mts_map[triplet_key] = ts

    if triplet_key in triplet_tss_list_map:
        triplet_tss_list_map[triplet_key].append(ts)
    else:
        triplet_tss_list_map[triplet_key] = [ts]


def triplet_map_check(triplet_key: str, min_time):
    global triplet_mts_map
    return (triplet_key in triplet_mts_map) and (triplet_mts_map[triplet_key] >= min_time)


def get_newer_triplet_tss_list(triplet_key: str, min_time):
    global triplet_tss_list_map
    result_list = []
    if triplet_key in triplet_tss_list_map:
        ts_list = triplet_tss_list_map[triplet_key]
        for ts in ts_list:
            if ts > min_time:
                result_list.append(ts)
    return result_list


def add_triplet_tss_map(triplet_key: str, ts: int):
    global triplet_tss_map
    if triplet_key in triplet_tss_map:
        triplet_tss_map[triplet_key].add(ts)
    else:
        triplet_tss_map[triplet_key] = {ts}


def get_triplet_tss_map(triplet_key: str):
    global triplet_tss_map
    if triplet_key in triplet_tss_map:
        return triplet_tss_map[triplet_key]
    else:
        return {}


def triplet_tss_map_check(triplet_key: str, min_time):
    global triplet_tss_map
    result_list = []
    if triplet_key in triplet_tss_map:
        ts_list = triplet_tss_map[triplet_key]
        for ts in ts_list:
            if ts >= min_time:
                result_list.append(ts)
    return result_list


def grad_f1(truth, logit, support=0, a=0.5):
    return truth - logit


def score_f1(logit, support=0, a=0.5):
    return a * logit + (1 - a) * support


def grad_f2(truth, logit, support=0, a=0.5):
    return a * truth + (1 - a) * support - logit


def score_f2(logit, support=0, a=0.5):
    return logit


def grad_f3(truth, logit, support=0, a=0.5):
    return truth - logit


def score_f3(logit, support=0, a=0.5):
    return logit


funcs = {
    'f1': [grad_f1, score_f1],
    'f2': [grad_f2, score_f2],
    'f3': [grad_f3, score_f3],
}


def output_hidden_explainable(hidden_dict, rules, rel_quadruples, path, rel_size):
    rule_type_map = {
        0: 'COMPOSITION',
        1: 'SYMMETRIC',
        2: 'INVERSE',
        3: 'SUBRELATION',
        4: 'TLOGIC'
    }

    origin_path = os.path.join(os.path.dirname(path), '../../origin')

    ent2id = json.load(open(origin_path + '/entity2id.json', 'r'))
    rel2id = json.load(open(origin_path + '/relation2id.json', 'r'))
    ts2id = json.load(open(origin_path + '/ts2id.json', 'r'))

    id2ent = {int(id) : name for name, id in ent2id.items()}
    id2rel = {int(id) : name for name, id in rel2id.items()}
    id2ts = {int(id) : name for name, id in ts2id.items()}

    rel_len = len(id2rel)
    for name, id in rel2id.items():
        id2rel[id+rel_len] = '[IVERSE]' + name

    new_dict = {}
    for key in hidden_dict.keys():
        ts_name = id2ts[key[3]] if key[3] < len(ts2id) else id2ts[len(id2ts)-1] + '+' + str(key[3]-len(ts2id))
        new_key = '{}---{}--->{}--@--{}'.format(id2ent[key[0]], id2rel[key[1]], id2ent[key[2]], ts_name)
        temp_rules = []
        for i in hidden_dict[key]:
            temp_rule = rules[i]
            item = {
                'type': rule_type_map[temp_rule.rule_type],
                'head': id2rel[temp_rule.rule_hypothesis],
                'body': [id2rel[i] for i in temp_rule.rule_premise],
                'gap': temp_rule.mid_gap,
                'conf': temp_rule.precision,
            }

            head = key[0]
            tail = key[2]

            node_list = [{} for j in range(len(temp_rule.rule_premise))]

            pre = [head]
            k = 0
            for j in temp_rule.rule_premise:
                temp_quads = rel_quadruples[j]
                new_pre = []
                for tq in temp_quads:
                    if tq[0] in pre:
                        if tq[0] in node_list[k]:
                            node_list[k][tq[0]].append(tq[2])
                        else:
                            node_list[k][tq[0]] = [tq[2]]
                        new_pre.append(tq[2])
                pre = new_pre
                k += 1

            path_list = [[] for j in range(len(temp_rule.rule_premise) + 1)]
            path_list[len(temp_rule.rule_premise)].append(tail)
            for j in range(len(temp_rule.rule_premise)-1, -1, -1):
                for k in node_list[j].keys():
                    if len(list(set(node_list[j][k]) & set(path_list[j+1]))) != 0:
                        path_list[j].append(k)

            # item['path'] = path_list
            item['path'] = [[id2ent[i] for i in path_item] for path_item in path_list]

            temp_rules.append(item)


        new_dict[new_key] = {
            'hidden': [id2ent[key[0]], id2rel[key[1]], id2ent[key[2]], ts_name],
            'rules': temp_rules
        }

    json.dump(new_dict, open(path, 'w'), indent=4)


def merge_dict_set(main_data: dict, sub_data: dict):
    for k in sub_data:
        if k in main_data:
            for j in sub_data[k]:
                main_data[k].add(j)
        else:
            main_data[k] = sub_data[k]
    return main_data


# code from TLogic
def score1(rule, c=0):
    # score = rule["rule_supp"] / (rule["body_supp"] + c)
    score = rule.precision

    return score


def score1_new(rule, c=0):
    # score = rule["rule_supp"] / (rule["body_supp"] + c)
    score = rule.weight

    return score


# code from TLogic
def score2(max_cands_ts, test_query_ts, lmbda):
    score = np.exp(
        lmbda * (max_cands_ts - test_query_ts)  # max_cands_ts < test_query_ts (maybe)
    )  # Score depending on time difference

    return score


# code from TLogic
def score_12(rule_score, max_cands_ts, test_query_ts, lmbda, a):
    # score = a * score1(rule) + (1 - a) * score2(max_cands_ts, test_query_ts, lmbda)
    score = a * rule_score + (1 - a) * score2(max_cands_ts, test_query_ts, lmbda)

    return score


def get_base_dist(learn_data, fs=False):
    ent_id = 0 if fs else 2
    total_dist = {}
    rel_dist = {}
    rel_count = {}
    for i in range(len(learn_data)):
        item = learn_data[i]
        if item[ent_id] not in total_dist:
            total_dist[item[ent_id]] = 0
        if item[1] not in rel_dist:
            rel_dist[item[1]] = {}
            rel_count[item[1]] = 0
        if item[ent_id] not in rel_dist[item[1]]:
            rel_dist[item[1]][item[ent_id]] = 0

        total_dist[item[ent_id]] += 1
        rel_dist[item[1]][item[ent_id]] += 1
        rel_count[item[1]] += 1

    total_count = len(learn_data)
    for i in total_dist:
        total_dist[i] /= total_count

    for i in rel_dist:
        for j in rel_dist[i]:
            rel_dist[i][j] /= rel_count[i]

    total_dist = dict(sorted(total_dist.items(), key=lambda x: x[1], reverse=True))
    for r in rel_dist:
        rel_dist[r] = dict(sorted(rel_dist[r].items(), key=lambda x: x[1], reverse=True))

    return total_dist, rel_dist


def get_base_dist_v2(learn_data, fs=False):
    total_dist = {}
    rel_dist = {}
    rel_count = {}
    for i in range(len(learn_data)):
        item = learn_data[i]
        if item[0] not in total_dist:
            total_dist[item[0]] = 0
        if item[2] not in total_dist:
            total_dist[item[2]] = 0
        if item[1] not in rel_dist:
            rel_dist[item[1]] = {}
            rel_count[item[1]] = 0
        if item[0] not in rel_dist[item[1]]:
            rel_dist[item[1]][item[0]] = 0
        if item[2] not in rel_dist[item[1]]:
            rel_dist[item[1]][item[2]] = 0

        total_dist[item[0]] += 1
        rel_dist[item[1]][item[0]] += 1
        total_dist[item[2]] += 1
        rel_dist[item[1]][item[2]] += 1
        rel_count[item[1]] += 2

    total_count = len(learn_data) * 2
    for i in total_dist:
        total_dist[i] /= total_count

    for i in rel_dist:
        for j in rel_dist[i]:
            rel_dist[i][j] /= rel_count[i]

    total_dist = dict(sorted(total_dist.items(), key=lambda x: x[1], reverse=True))
    for r in rel_dist:
        rel_dist[r] = dict(sorted(rel_dist[r].items(), key=lambda x: x[1], reverse=True))

    return total_dist, rel_dist


def scale_in_1(scores, max_val):
    if max_val != 1:
        scores = [[x[0]/max_val, x[1]] if abs(x[0]) < max_val else [x[0]/abs(x[0]), x[1]] for x in scores]
    return scores


def cand_score_avg(scores, params):
    return round(sum([i[0] for i in scores]) / len(scores), ROUND_DECIMAL)


def cand_score_tavg(scores, params):
    scores = score_addts(scores, params)
    return cand_score_avg(scores, params)


# noisy-or
def cand_score_nso(scores, params):
    final_score = 1
    for sc in scores:
        final_score *= 1 - sc[0]

    return round(1 - final_score, ROUND_DECIMAL)


def cand_score_tnso(scores, params):
    scores = score_addts(scores, params)
    return cand_score_nso(scores, params)


def score_addts(scores, params):
    alpha = params[0]
    lmbda = params[1]
    query_ts = params[2]
    new_scores = []
    for i in range(len(scores)):
        t_part = np.exp(
            lmbda * (scores[i][1] - query_ts)
        )
        new_scores.append([alpha * scores[i][0] + (1 - alpha) * t_part, scores[i][1]])

    return new_scores

