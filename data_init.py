import os
import sys
import random
from collections import defaultdict


entity_set = set()
relation_set = set()
time_set = set()
year2id = {}  # for interval time

entity_map = {}
relation_map = {}
time_map = {}

# for interval init
interval_times = {}
year_list = []
history_year = '-500'  # default earliest year, must be as string
future_year = '3000'  # default future year, must be as string


def init_interval(in_path, datalist):
    global interval_times, year_list, history_year, future_year

    for dataset in datalist:
        filename = dataset + '.txt'
        with open(os.path.join(in_path, filename), 'r') as fr:
            for line in fr:
                line_split = line.strip().split('\t')
                start_str = line_split[3].replace('#', '0')
                start_BC = start_str[0] == '-'
                start_tidx = 1 if start_BC else 0

                start_year = start_str.split('-')[start_tidx]
                if start_year == '0000':
                    start_year = history_year
                start_month = start_str.split('-')[start_tidx + 1]
                start_day = start_str.split('-')[start_tidx + 2]
                start = start_year + start_month + start_day
                start_int = -int(start) if start_BC else int(start)
                year_list.append(start_int)
                interval_times[line_split[3]] = start_int

                end_str = line_split[4].replace('#', '0')
                end_BC = end_str[0] == '-'
                end_tidx = 1 if end_BC else 0

                end_year = end_str.split('-')[end_tidx]
                if end_year == '0000':
                    end_year = future_year
                end_month = end_str.split('-')[end_tidx + 1]
                end_day = end_str.split('-')[end_tidx + 2]
                end = end_year + end_month + end_day
                end_int = -int(end) if end_BC else int(end)
                year_list.append(end_int)
                interval_times[line_split[4]] = start_int
            fr.close()

    year_list.sort()
    freq = defaultdict(int)
    for y in year_list:
        freq[y] = freq[y] + 1
    year_class = []
    count = 0
    for key in sorted(freq.keys()):
        count += freq[key]
        if count >= 30:
            year_class.append(key)
            count = 0
    year_class[-1] = year_list[-1]
    prev_year = year_list[0]
    for i, yr in enumerate(year_class):
        year2id[(prev_year, yr)] = i
        prev_year = yr + 1


def load_raw(in_path, filename, is_interval=False):
    global entity_set, relation_set, time_set
    with open(os.path.join(in_path, filename), 'r') as fr:
        hexaple_list = []
        for line in fr:
            line_split = line.strip().split('\t')

            head = line_split[0]
            rel = line_split[1]
            tail = line_split[2]

            if is_interval:
                start2id, end2id = interval_times[line_split[3]], interval_times[line_split[4]]
                for key, time_idx in sorted(year2id.items(), key=lambda x:x[1]):
                    if start2id >= key[0] and start2id <= key[1]:
                        start2id = time_idx
                        startkey = key
                    if end2id >= key[0] and end2id <= key[1]:
                        end2id = time_idx
                        endkey = key
                if start2id == end2id:
                    tss = [startkey]
                else:
                    tss = [startkey, endkey]
            else:
                tss = [line_split[3]]
                time_set.add(line_split[3])

            entity_set.add(head)
            relation_set.add(rel)
            entity_set.add(tail)

            for ts in tss:
                hexaple_list.append([head, rel, tail, ts])

    return hexaple_list


if len(sys.argv) < 3:
    print("Param error! Usage: python data_init.py <data_dir> <is_interval>")
    exit()

data_dir = sys.argv[1]
is_interval = True if int(sys.argv[2]) == 1 else False
origin_dir = os.path.join(data_dir, 'origin')
datalist = ['train', 'valid', 'test']

for dtype in datalist:
    filename = dtype + '.txt'
    if not os.path.isfile(os.path.join(origin_dir, filename)):
        print('Data file "' + os.path.join(origin_dir, filename) + '" not found.')
        exit()

print("[DATA INITIALIZE] - Parsing original data files into id format...")

if is_interval:
    init_interval(origin_dir, datalist)

raw_train = load_raw(origin_dir, 'train.txt', is_interval)
raw_valid = load_raw(origin_dir, 'valid.txt', is_interval)
raw_test = load_raw(origin_dir, 'test.txt', is_interval)

entities = list(entity_set)
entities.sort()
relations = list(relation_set)
relations.sort()
times = list(time_set)
times.sort()

for i in range(len(entities)):
    entity_map[entities[i]] = i

for i in range(len(relations)):
    relation_map[relations[i]] = i

for i in range(len(times)):
    time_map[times[i]] = i

with open(os.path.join(data_dir, 'stat.txt'), 'w') as fo:
    fo.write('{}\t{}\t{}'.format(len(entities), len(relations), len(year2id) if is_interval else len(times)))
    fo.close()

uniq_check = set()
with open(os.path.join(data_dir, 'trainids.txt'), 'w') as fo:
    for i in range(len(raw_train)):
        quadruple = (entity_map[raw_train[i][0]], relation_map[raw_train[i][1]], entity_map[raw_train[i][2]], year2id[raw_train[i][3]] if is_interval else time_map[raw_train[i][3]])
        if quadruple in uniq_check:
            continue
        else:
            uniq_check.add(quadruple)
        fo.write('{}\t{}\t{}\t{}\n'.format(
            *quadruple
        ))
    fo.close()

uniq_check.clear()
with open(os.path.join(data_dir, 'validids.txt'), 'w') as fo:
    for i in range(len(raw_valid)):
        quadruple = (entity_map[raw_valid[i][0]], relation_map[raw_valid[i][1]], entity_map[raw_valid[i][2]], year2id[raw_valid[i][3]] if is_interval else time_map[raw_valid[i][3]])
        if quadruple in uniq_check:
            continue
        else:
            uniq_check.add(quadruple)
        fo.write('{}\t{}\t{}\t{}\n'.format(
            *quadruple
        ))
    fo.close()

uniq_check.clear()
with open(os.path.join(data_dir, 'testids.txt'), 'w') as fo:
    for i in range(len(raw_test)):
        quadruple = (entity_map[raw_test[i][0]], relation_map[raw_test[i][1]], entity_map[raw_test[i][2]], year2id[raw_test[i][3]] if is_interval else time_map[raw_test[i][3]])
        if quadruple in uniq_check:
            continue
        else:
            uniq_check.add(quadruple)
        fo.write('{}\t{}\t{}\t{}\n'.format(
            *quadruple
        ))
    fo.close()

with open(os.path.join(data_dir, 'entitymap.txt'), 'w') as fo:
    for i in range(len(entities)):
        fo.write('{}\t{}\n'.format(i, entities[i]))
    fo.close()

with open(os.path.join(data_dir, 'relationmap.txt'), 'w') as fo:
    for i in range(len(relations)):
        fo.write('{}\t{}\n'.format(i, relations[i]))
    fo.close()

with open(os.path.join(data_dir, 'timemap.txt'), 'w') as fo:
    if is_interval:
        for interval, id in year2id.items():
            fo.write('{}\t{}\n'.format(id, interval))
    else:
        for i in range(len(times)):
            fo.write('{}\t{}\n'.format(i, times[i]))
    fo.close()

print("[DATA INITIALIZE] - Finished.")
