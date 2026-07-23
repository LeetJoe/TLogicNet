
import os
import json
import math
import datetime
import argparse
import concurrent.futures


relation_size = -1
time_size = -1
mid_score = 1
score_win = 90
hidden = {}
rules = {}
infers = []


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Generate infers and found',
        usage='python infer_found_v1.py [<args>] [-h | --help]'
    )

    parser.add_argument('--data_path', '-d', type=str, help='path to data file')
    parser.add_argument('--mln_path', '-m', type=str, help='path of mln')
    parser.add_argument('--min_path_len', '-ml', type=int, default = 2, help='min path number of hidden')
    parser.add_argument('--infer_time_gap',  '-tg', type=int, default=7, help='time gap')
    parser.add_argument('--score_win',  '-sw', type=int, default=90, help='time score window')
    parser.add_argument('--time_limit',  '-tl', type=int, default=-1, help='infer time range limit')
    parser.add_argument('--topk',  '-t', type=int, help='topk')
    parser.add_argument('--thresh',  '-th', type=float, help='threshold for found')
    parser.add_argument('--threads',  '-td', type=int, default=10, help='threads')

    return parser.parse_args(args)


def load_stat(stat_file):
    global relation_size, time_size
    with open(stat_file, 'r') as fr:
        line = fr.readline()
        line_split = line.split()
        relation_size = int(line_split[1])
        time_size = int(line_split[2])
        fr.close()


def load_rules(rule_file):
    global rules
    with open(rule_file, 'r') as fr:
        for entry in fr:
            line_split = entry.strip().split('\t')
            rules[int(line_split[0])] = float(line_split[7])  # weight

        fr.close()


def load_hidden(hidden_file, min_len=2):
    global hidden, relation_size
    temp_hidden = {}
    with open(hidden_file, 'r') as fr:
        for entry in fr:
            line_split = entry.strip().split('\t')
            if int(line_split[1]) < relation_size:
                key = (int(line_split[0]), int(line_split[1]), int(line_split[2]))
            else:
                key = (int(line_split[2]), int(line_split[1]) - relation_size, int(line_split[0]))
            rule_times = json.loads(line_split[3])

            if key not in temp_hidden:
                temp_hidden[key] = rule_times
            else:
                existing_rule_times = temp_hidden[key]
                for rid in rule_times:
                    if rid not in existing_rule_times:
                        time_list = rule_times[rid]
                        existing_rule_times[rid] = time_list
                    else:
                        time_list = existing_rule_times[rid] + rule_times[rid]
                        existing_rule_times[rid] = time_list
                temp_hidden[key] = existing_rule_times
        fr.close()

    # filter hidden
    for key in temp_hidden:
        path_num = 0
        for rid in temp_hidden[key]:
            path_num += len(temp_hidden[key][rid])
        if path_num >= min_len:
            hidden[key] = temp_hidden[key]


def generate_infer(time_gap, num_threads: int = 10):
    global time_size, hidden, rules, infers
    scored_infer = []
    batch_size = 10000
    hidden_size = len(hidden)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_threads, hidden_size//batch_size + 1)) as executor:
        future_to_sidx = {executor.submit(get_batch_scored_infer, i, min(i + batch_size, hidden_size)): i
                          for i in range(1, hidden_size, batch_size)}
        i = 0
        for future in concurrent.futures.as_completed(future_to_sidx):
            # sidx = future_to_sidx[future]
            i += batch_size
            scored_infer += future.result()
            print('[POST MLN] - Progress of calculating infer score: %d/%d(%d)' % (min(i, hidden_size), hidden_size, len(scored_infer)), end='\r')
        executor.shutdown(wait=True)

    ## todo keep for debug purposes
    # for t in range(1, time_size):
    #     for key in hidden:
    #         score, exp_str = get_score(hidden[key], rules, t)
    #         scored_infer.append([key[0], key[1], key[2], t, score, exp_str])

    #
    tripmap = {}
    for item in scored_infer:
        tripkey = tuple(item[:3])
        if tripkey not in tripmap:
            tripmap[tripkey] = []
        tripmap[tripkey].append(item[3:])

    for tripkey in tripmap:
        if len(tripmap[tripkey]) > 1:
            templist = sorted(tripmap[tripkey], key=lambda x: x[0])
            cur_item = None
            for item in templist:
                if cur_item is None:
                    cur_item = item
                elif item[0] - cur_item[0] > time_gap:
                    infers.append(list(tripkey) + cur_item)
                    cur_item = item
                elif item[1] > cur_item[1]:
                    cur_item = item
            infers.append(list(tripkey) + cur_item)
        elif len(tripmap[tripkey]) == 1:
            infers.append(list(tripkey) + tripmap[tripkey][0])

    infers.sort(key=lambda x:x[4], reverse=True)


def get_batch_scored_infer(start, end):
    global hidden
    scored_infer_batch = []
    hkeys = list(hidden.keys())
    for kid in range(start, end):
        key = hkeys[kid]
        cand_time_scores = get_score(hidden[key])
        for t, score in cand_time_scores.items():
            scored_infer_batch.append([key[0], key[1], key[2], t, score])
        hidden[key] = None
    return scored_infer_batch


# get rule late times by time window
def get_win_rule_times(rule_times, delta):
    global time_size
    cand_win_rule_times = {}
    for rid in rule_times:
        for time_pair in rule_times[rid]:
            if time_pair[1] > time_size - 2:
                continue
            for cand_time in range(time_pair[1] + 1, time_size):
                if time_pair[0] >= cand_time - delta:
                    if cand_time not in cand_win_rule_times:
                        cand_win_rule_times[cand_time] = {}
                    if rid not in cand_win_rule_times[cand_time]:
                        cand_win_rule_times[cand_time][rid] = []
                    cand_win_rule_times[cand_time][rid].append(time_pair[0])
    return cand_win_rule_times


# 考虑 body 路径数量和时间窗口的方法
def get_score(rule_times):
    global rules, score_win
    cand_weight_sum = {}
    rule_time_window = get_win_rule_times(rule_times, score_win)
    if len(rule_time_window) == 0:
        return cand_weight_sum
    for cand_time in rule_time_window:
        weight_sum = 0
        for rid in rule_time_window[cand_time]:
            rule_score = rules[int(rid)]
            t_score = 0
            for tb in rule_time_window[cand_time][rid]:
                t_score += math.exp(
                    0.1 * (tb - cand_time)
                )
            weight_sum += rule_score * t_score
        if weight_sum >= 1: # base limit
            cand_weight_sum[cand_time] = weight_sum
    return cand_weight_sum


def save_infers(mln_path, topk):
    global infers, mid_score
    scored_infer_file = os.path.join(mln_path, 'scored_inferids.txt')
    iter_mid = 0
    mid_score = 1
    with open(scored_infer_file, 'w') as fw:
        for entry in infers:
            fw.write('{}\t{}\t{}\t{}\t{}\n'.format(*entry))
            iter_mid += 1
            if iter_mid == topk // 2:
                mid_score = entry[4]
        fw.close()

    infer_file = os.path.join(mln_path, 'inferids.txt')
    with open(infer_file, 'w') as fw:
        for entry in infers:
            fw.write('{}\t{}\t{}\t{}\t{}\n'.format(*entry[:4], sigmoid((entry[4] - mid_score)/mid_score)))
            topk -= 1
            if topk == 0:
                break
        fw.close()


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def found_from_infer(mln_dir, thresh):
    with open(os.path.join(mln_dir, 'foundids.txt'), 'w') as fw:
        for entry in infers:
            new_score = sigmoid((entry[4] - mid_score)/mid_score)
            if new_score >= thresh:
                fw.write('{}\t{}\t{}\t{}\n'.format(*entry[:4]))
            else:
                break
        fw.close()


if __name__ == '__main__':
    args = parse_args()
    score_win = args.score_win

    print('[({})POST MLN] - Start generating infer and found...                          '.format(datetime.datetime.now()))
    stat_file = os.path.join(args.data_path, 'stat.txt')
    load_stat(stat_file)
    if args.time_limit > 0:
        time_size = int(args.time_limit)

    rule_file = os.path.join(args.mln_path, 'rules.txt')
    load_rules(rule_file)
    hidden_file = os.path.join(args.mln_path, 'hiddenids.txt')
    load_hidden(hidden_file, args.min_path_len)

    generate_infer(args.infer_time_gap, min(args.threads, 10))

    save_infers(args.mln_path, args.topk)

    found_from_infer(args.mln_path, args.thresh)
    print('\n[({})POST MLN] - Finished generating infer and found.                          '.format(datetime.datetime.now()))
