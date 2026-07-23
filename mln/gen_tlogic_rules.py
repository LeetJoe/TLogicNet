import os
import time
import random
import argparse
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from grapher import Grapher
from temporal_walk import Temporal_Walk
from rule_learning import Rule_Learner, rules_statistics


parser = argparse.ArgumentParser()
parser.add_argument("--datapath", "-d", default="", type=str)
parser.add_argument("--rule_lengths", "-l", default=[1, 2, 3], type=int, nargs="+")
parser.add_argument("--num_walks", "-n", default="100", type=int)
parser.add_argument("--min_conf", "-mc", default="0.1", type=float)
parser.add_argument("--transition_distr", "-td", default="exp", type=str)
parser.add_argument("--num_processes", "-p", default=20, type=int)
parser.add_argument("--extra", "-e", default='', type=str)
parser.add_argument("--seed", "-s", default=-1, type=int)
parsed = vars(parser.parse_args())

rule_lengths = parsed["rule_lengths"]
rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths
num_walks = parsed["num_walks"]
min_confidence = parsed["min_conf"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
extra_annots_file = parsed["extra"]
seed = parsed["seed"]

if seed is None or seed < 0:
    seed = random.randint(1, 10000)

dataset_dir = parsed['datapath']
data = Grapher(dataset_dir, extra_annots_file)
# todo 这里只使用了 train 数据来进行随机游走，后面可以考虑把 valid 也加进来
temporal_walk = Temporal_Walk(data.train_idx, data.num_rels, transition_distr)
rl = Rule_Learner(temporal_walk.edges, data.num_rels, dataset_dir, data.num_times, min_confidence)
all_relations = sorted(temporal_walk.edges)  # Learn for all relations


def learn_rules(i, batch_num):
    """
    Learn rules (multiprocessing possible).

    Parameters:
        i (int): process number
        batch_num (int): minimum number of relations for each process

    Returns:
        rl.rules_dict (dict): rules dictionary
    """

    if seed:
        np.random.seed(seed)

    num_rest_relations = len(all_relations) - (i + 1) * batch_num
    if num_rest_relations >= batch_num:
        relations_idx = range(i * batch_num, (i + 1) * batch_num)
    else:
        relations_idx = range(i * batch_num, len(all_relations))

    # num_rules = [0]
    for k in relations_idx:
        rel = all_relations[k]
        for length in rule_lengths:
            # it_start = time.time()
            for _ in range(num_walks):
                walk_successful, walk = temporal_walk.sample_walk(length + 1, rel)
                if walk_successful:
                    rl.create_rule(walk)
            # it_end = time.time()
            # it_time = round(it_end - it_start, 6)
            # num_rules.append(sum([len(v) for k, v in rl.rules_dict.items()]) // 2)
            # num_new_rules = num_rules[-1] - num_rules[-2]
            # print(
            #     "Process {0}: relation {1}/{2}, length {3}: {4} sec, {5} rules".format(
            #         i,
            #         k - relations_idx[0] + 1,
            #         len(relations_idx),
            #         length,
            #         it_time,
            #         num_new_rules,
            #     )
            # )

    return rl.rules_dict


start = time.time()
relation_bulk_size = max(len(all_relations) // num_processes, 1)
output = Parallel(n_jobs=num_processes)(
    delayed(learn_rules)(i, relation_bulk_size) for i in range(num_processes)
)
end = time.time()

all_rules = output[0]
for i in range(1, num_processes):
    all_rules.update(output[i])

total_time = round(end - start, 6)
print("Learning finished in {} seconds.".format(total_time))

rl.rules_dict = all_rules
rl.sort_rules_dict()
dt = datetime.now()
dt = dt.strftime("%d%m%y%H%M%S")
rl.save_rules(dt, rule_lengths, num_walks, transition_distr, seed)
# rl.save_rules_verbalized(dt, rule_lengths, num_walks, transition_distr, seed)
rules_statistics(rl.rules_dict)
