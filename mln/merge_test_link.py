import json
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Merge linked test',
        usage='python merge_test_link.py [<args>] [-h | --help]'
    )

    parser.add_argument('--stat_file', type=str, help='path to stat file')
    parser.add_argument('--file_path', type=str, help='path to batch file')
    parser.add_argument('--save_path', type=str, help='file path of merged result')

    return parser.parse_args(args)


def get_stat(stat_file):
    with open(stat_file, 'r') as fr:
        for line in fr:
            line_split = line.split('\t')
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_batch_file(batch_file):
    batch_data = {}
    with open(batch_file, 'r') as fr:
        for line in fr:
            # s, p, o, t, rid, {candidateId: [earlyTime]}
            line_split = line.strip().split('\t')
            test_key = (line_split[0], line_split[1], line_split[2], line_split[3])
            if test_key not in batch_data:
                batch_data[test_key] = []
            # for cand in cand_dict:
            #     cand_dict[cand] = sorted(cand_dict[cand])
            # cand_dict = dict(sorted(cand_dict.items(), key=lambda x: int(x[0])))
            batch_data[test_key].append([line_split[4], line_split[5]])
        fr.close()

    return batch_data


def merge_data(batch_data, merge_file, rel_size):
    with open(merge_file, 'w') as fw:
        for test_key in batch_data:
            item_merge = {}
            delimiter = ''
            for item in batch_data[test_key]:
                cand_dict = json.loads(item[1])
                for cand, t in cand_dict.items():
                    if cand not in item_merge:
                        item_merge[cand] = {}
                    if item[0] not in item_merge[cand]:
                        item_merge[cand][item[0]] = t
            # item_merge = dict(sorted(item_merge.items(), key=lambda x: x[0]))
            if int(test_key[1]) < rel_size:
                fw.write('{}\t{}\t{}\t{}\tsp\t'.format(*test_key))
            else:
                fw.write('{}\t{}\t{}\t{}\tpo\t'.format(test_key[2], int(test_key[1]) - rel_size, test_key[0], test_key[3]))
            fw.write('{}\n'.format(json.dumps(item_merge)))
        fw.close()


if __name__ == '__main__':
    args = parse_args()
    stat = get_stat(args.stat_file)
    batch_data = load_batch_file(args.file_path)
    merge_data(batch_data, args.save_path, stat[1])



