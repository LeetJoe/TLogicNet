import os
import sys
import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Filter annotation for mln',
        usage='python filter_annotation.py [<args>] [-h | --help]'
    )

    parser.add_argument('--origin', '-o', type=str, help='path to original annotation file')
    parser.add_argument('--save', '-s', type=str, help='path to filtered annotation file')
    parser.add_argument('--valid_thresh', '-t', type=float, help='threshold to filter annotation')

    return parser.parse_args(args)


def load_filter_v2(origin_file, save_file, valid_thresh):
    total_data = []
    with open(origin_file, 'r') as fr:
        for line in fr:
            line_split = line.strip().split('\t')
            total_data.append([line_split[0], line_split[1], line_split[2], line_split[3], line_split[4], float(line_split[5])])
        fr.close()

    total_data.sort(key=lambda x:x[5], reverse=True)
    with open(save_file, 'w') as fw:
        for i in range(len(total_data)):
            score = total_data[i][5]
            if score >= valid_thresh:
                valid = 1
            else:
                valid = 0
            fw.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(*total_data[i], valid))
        fw.close()
    return


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.origin):
        print("[PRE MLN] - ERROR! File not found: {} !".format(args.origin))
        exit()

    load_filter_v2(args.origin, args.save, args.valid_thresh)
    print("[PRE MLN] - Filter annotations for MLN done!")
