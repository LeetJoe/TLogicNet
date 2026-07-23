#!/usr/bin/env python
"""Preprocess a KGE dataset into a the format expected by libkge.

Call as `preprocess.py --dataset <name>`. The original dataset should be stored in
subfolder `name` and have files "train.txt", "valid.txt", and "test.txt". Each file
contains one SPO triple per line, separated by tabs.

During preprocessing, each distinct entity name and each distinct relation name
is assigned an index (dense). The index-to-object mapping is stored in files
"entity_map.del" and "relation_map.del", resp. The triples (as indexes) are stored in
files "train.del", "valid.del", and "test.del". Metadata information is stored in a file
"dataset.yaml".

"""
import subprocess
import argparse
import yaml
import os.path
import numpy as np
from collections import OrderedDict


def store_map(symbol_map, filename):
    with open(filename, "w") as f:
        for symbol, index in symbol_map.items():
            f.write(f"{index}\t{symbol}\n")


def load_map(filename):
    symbol_map = {}
    with open(filename, "r") as f:
        for line in f:
            id_pair = line.strip().split('\t')
            symbol_map[id_pair[1]] = int(id_pair[0])

        f.close()
    return symbol_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--datafolder", type=str, default='')
    parser.add_argument("--dir", type=str)
    parser.add_argument("--order_sop", action="store_true")
    # parser.add_argument("--iteration", default='0', type=str)
    args = parser.parse_args()

    if args.datafolder == '':
        args.datafolder = args.dataset

    args.dir = 'data/' + args.datafolder

    subprocess.run(['sed', '-i','2s/.*/dataset.name: ' + args.datafolder + '/', 'eceformer/' + args.dataset + '-infer.yaml'])

    print(f"Preprocessing {args.dir}...")

    split_files = {"train": "trainids.txt", "valid": "validids.txt", "test": "testids.txt"}
    split_files_without_unseen = {"train_sample": "train_sample.del",
                                  "valid_without_unseen": "valid_without_unseen.del",
                                  "test_without_unseen": "test_without_unseen.del"}

    split_sizes = {}

    if args.order_sop:
        S, P, O, T = 0, 2, 1, 3
    else:
        S, P, O, T = 0, 1, 2, 3

    # read data and collect entities and relations
    raw = {}
    entities_in_train = set()
    relations_in_train = set()
    times_in_train = set()
    for split, filename in split_files.items():
        with open(args.dir + "/" + filename, "r") as f:
            raw[split] = list(map(lambda s: s.strip().split("\t"), f.readlines()))
            print(
                f"Found {len(raw[split])} triples in {split} split "
                f"(file: {filename})."
            )
            split_sizes[split] = len(raw[split])

    if 'train' in raw:
        for i in range(len(raw['train'])):
            entities_in_train.add(raw['train'][i][S])
            entities_in_train.add(raw['train'][i][O])
            relations_in_train.add(raw['train'][i][P])
            times_in_train.add(raw['train'][i][T])
        entities_in_train = list(entities_in_train)
        relations_in_train = list(relations_in_train)
        times_in_train = list(times_in_train)

    filename = "foundids.txt"
    train_aug_file = 'trainids_aug.txt'
    if os.path.exists(args.dir + "/" + filename):
        real_train_map= {}
        for i in range(len(raw['train'])):
            triple_key = (raw['train'][i][S], raw['train'][i][P], raw['train'][i][O])
            if triple_key in real_train_map:
                real_train_map[triple_key].add(raw['train'][i][T])
            else:
                real_train_map[triple_key] = {raw['train'][i][T]}
        with open(args.dir + "/" + filename, "r") as f:
            raw_found = []
            for line in f:
                line_split = line.strip().split("\t")
                if len(line_split) <= 2:
                    continue
                raw_found.append([int(i) for i in line_split[:4]])

            raw['found'] = raw_found

            # filter_found = []
            # for i in range(len(raw_found)):
            #     triple_key = (raw_found[i][S], raw_found[i][P], raw_found[i][O])
            #     keep_it = True
            #     if triple_key in real_train_map:
            #         for tt in real_train_map[triple_key]:
            #             if abs(int(tt) - int(raw_found[i][T])) <= 3:
            #                 keep_it = False
            #                 break
            #
            #     if keep_it:
            #         filter_found.append(raw_found[i])
            # raw['found'] = filter_found

        if len(raw['found']) > 0:
            raw['train'] += raw['found']
            split_sizes['train'] += len(raw['found'])

    # write out triples using indexes
    print("Writing triples...")
    without_unseen_sizes = {}
    for split, filename in split_files.items():
        if split in ["valid", "test"]:
            split_without_unseen = split + "_without_unseen"
            f_wo_unseen = open(os.path.join(args.dir,
                                            split_files_without_unseen[split_without_unseen]), "w")
        elif split in ["train"]:
            split_without_unseen = split + "_sample"
            f_tr_sample = open(os.path.join(args.dir,
                                            split_files_without_unseen[split_without_unseen]), "w")
            train_sample = np.random.choice(split_sizes["train"], split_sizes["valid"], False)

            f_aug_split = open(os.path.join(args.dir, train_aug_file), 'w')
        else:
            split_without_unseen = ''

        size_unseen = 0
        for n, t in enumerate(raw[split]):
            if split in ['train']:
                f_aug_split.write(
                    str(t[S])
                    + "\t"
                    + str(t[P])
                    + "\t"
                    + str(t[O])
                    + "\t"
                    + str(t[T])
                    + "\n"
                )

            if split in ["train"] and n in train_sample:
                f_tr_sample.write(
                    str(t[S])
                    + "\t"
                    + str(t[P])
                    + "\t"
                    + str(t[O])
                    + "\t"
                    + str(t[T])
                    + "\n"
                )
                size_unseen += 1
            elif split in ["valid", "test"] and t[S] in entities_in_train and \
                    t[O] in entities_in_train and t[P] in relations_in_train:
                f_wo_unseen.write(
                    str(t[S])
                    + "\t"
                    + str(t[P])
                    + "\t"
                    + str(t[O])
                    + "\t"
                    + str(t[T])
                    + "\n"
                )
                size_unseen += 1
        if split_without_unseen != '':
            without_unseen_sizes[split_without_unseen] = size_unseen

    # write config
    print("Writing dataset.yaml...")
    with open(args.dir + "/stat.txt", 'r') as fr:
        line = fr.readline()
        line_split = line.split()
        entity_size = int(line_split[0])
        relation_size = int(line_split[1])
        timestamp_size = int(line_split[2])
        fr.close()

    dataset_config = dict(
        name=args.dir,
        num_entities=entity_size,
        num_relations=relation_size,
        num_times=timestamp_size
    )
    for split in split_files.keys():
        if split == 'train':
            dataset_config[f"files.{split}.filename"] = train_aug_file
        else:
            dataset_config[f"files.{split}.filename"] = split_files.get(split)
        dataset_config[f"files.{split}.type"] = "triples"
        dataset_config[f"files.{split}.size"] = split_sizes.get(split)
    for split in split_files_without_unseen.keys():
        dataset_config[f"files.{split}.filename"] = split_files_without_unseen.get(split)
        dataset_config[f"files.{split}.type"] = "triples"
        dataset_config[f"files.{split}.size"] = without_unseen_sizes.get(split)
    print(yaml.dump(dict(dataset=dataset_config)))
    with open(os.path.join(args.dir, "dataset.yaml"), "w+") as filename:
        filename.write(yaml.dump(dict(dataset=dataset_config)))

    print('Finished.')
