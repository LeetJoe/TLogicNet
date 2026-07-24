import os
import json
import numpy as np


class Grapher(object):
    def __init__(self, dataset_dir, extra_file=''):
        """
        Store information about the graph (train/valid/test set).
        Add corresponding inverse quadruples to the data.

        Parameters:
            dataset_dir (str): path to the graph dataset directory

        Returns:
            None
        """

        self.dataset_dir = dataset_dir

        data_stat = self.load_stat()
        self.num_rels = data_stat[1]
        self.num_times = data_stat[2]

        extra_idx = self.load_extra(extra_file)
        train_idx = self.create_store("trainids.txt")
        if len(extra_idx) > 0:
            self.train_idx = np.vstack((train_idx, extra_idx))
        else:
            self.train_idx = train_idx

        self.valid_idx = self.create_store("validids.txt")
        self.test_idx = self.create_store("testids.txt")
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))

        print("Grapher initialized.")


    def load_stat(self):
        """
        load data stat
        """
        with open(self.dataset_dir + "/stat.txt", 'r') as fr:
            line = fr.readline()
            line_split = line.split()
            entity_size = int(line_split[0])
            relation_size = int(line_split[1])
            timestamp_size = int(line_split[2])
            fr.close()

        return entity_size, relation_size, timestamp_size


    def create_store(self, file):
        """
        Store the quadruples from the file as indices.
        The quadruples in the file should be in the format "subject\trelation\tobject\ttimestamp\n".

        Parameters:
            file (str): file name

        Returns:
            store_idx (np.ndarray): indices of quadruples
        """

        with open(os.path.join(self.dataset_dir, file), "r", encoding="utf-8") as f:
            quads = f.readlines()
        store = self.split_quads(quads)
        store_idx = self.add_inverses(store)

        return store_idx


    def load_extra(self, file):
        extra_data = []
        if file != '' and os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    line_split = line.strip().split('\t')
                    if int(line_split[6]) == 1:
                        extra_data.append([int(i) for i in line_split[0:4]])
                f.close()

        extra_data = np.array(extra_data)
        if len(extra_data) != 0:
            extra_data = self.add_inverses(extra_data)
        return extra_data


    def split_quads(self, quads):
        """
        Split quadruples into a list of strings.

        Parameters:
            quads (list): list of quadruples
                          Each quadruple has the form "subject\trelation\tobject\ttimestamp\n".

        Returns:
            split_q (list): list of quadruples
                            Each quadruple has the form [subject, relation, object, timestamp].
        """

        split_q = []
        for quad in quads:
            split_q.append([int(i) for i in quad[:-1].split("\t")])

        return np.array(split_q)


    def add_inverses(self, quads_idx):
        """
        Add the inverses of the quadruples as indices.

        Parameters:
            quads_idx (np.ndarray): indices of quadruples

        Returns:
            quads_idx (np.ndarray): indices of quadruples along with the indices of their inverses
        """

        subs = quads_idx[:, 2]
        rels = [x + self.num_rels for x in quads_idx[:, 1]]
        objs = quads_idx[:, 0]
        tss = quads_idx[:, 3]
        inv_quads_idx = np.column_stack((subs, rels, objs, tss))
        quads_idx = np.vstack((quads_idx, inv_quads_idx))

        return quads_idx
