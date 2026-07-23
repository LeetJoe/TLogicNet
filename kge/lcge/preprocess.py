# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import time as pkg_time
import subprocess
import errno
from pathlib import Path
import pickle
import json

import numpy as np

from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Logic and Commonsense-Guided Temporal KGE"
    )
    parser.add_argument(
        '--data_path', default='./src_data', type=str,
        help="Experiment path of data"
    )

    return parser.parse_args()


def prepare_dataset(data_path):
    """
    Generate id maped train/valid/test/found/infer.pickle files for each EM iteration.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics for analysis.
    """
    split_files = {"train": "trainids.txt", "valid": "validids.txt", "test": "testids.txt", "infer": "inferids.txt"}
    raw = {}
    for split, filename in split_files.items():
        with open(data_path + "/" + filename, "r") as f:
            raw[split] = list(map(lambda s: s.strip().split("\t")[:4], f.readlines()))
            print(
                f"Found {len(raw[split])} triples in {split} split "
                f"(file: {filename})."
            )
    with open(data_path + "/stat.txt", 'r') as fr:
        line = fr.readline()
        line_split = line.split()
        entity_size = int(line_split[0])
        relation_size = int(line_split[1])
        timestamp_size = int(line_split[2])
        fr.close()

    found_file = data_path + "/foundids.txt"
    if os.path.exists(found_file):
        with open(found_file, "r") as f:
            raw_found = []
            for line in f:
                line_split = line.strip().split("\t")
                if len(line_split) <= 2:
                    continue
                raw_found.append([int(i) for i in line_split[:4]])

            raw['found'] = raw_found

        if len(raw['found']) > 0:
            raw['train'] += raw['found']

    # map train/valid/test/infer with the ids
    for split in split_files.keys():
        out = open(Path(data_path) / (split + '.pickle'), 'wb')
        pickle.dump(np.array(raw[split]).astype('uint64'), out)
        out.close()

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for split in split_files.keys():
        for lhs, rel, rhs, ts in raw[split]:
            to_skip['lhs'][(int(rhs), int(rel) + relation_size, int(ts))].add(int(lhs))  # reciprocals
            to_skip['rhs'][(int(lhs), int(rel), int(ts))].add(int(rhs))

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(data_path) / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    counters = {
        'lhs': np.zeros(entity_size),
        'rhs': np.zeros(entity_size),
        'both': np.zeros(entity_size)
    }

    for lhs, rel, rhs, _ts in raw['train']:
        counters['lhs'][int(lhs)] += 1
        counters['rhs'][int(rhs)] += 1
        counters['both'][int(lhs)] += 1
        counters['both'][int(rhs)] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(data_path) / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    args = parse_args()

    # prepare
    prepare_dataset(args.data_path)


