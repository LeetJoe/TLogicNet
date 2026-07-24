import os
import json
import itertools
import numpy as np
from collections import Counter


class Rule_Learner(object):
    def __init__(self, edges, num_relations, outputpath, ts_range, min_conf):
        """
        Initialize rule learner object.

        Parameters:
            edges (dict): edges for each relation
            num_relations (int): number of relations
            outputpath (str): output path

        Returns:
            None
        """

        self.edges = edges
        self.num_relations = num_relations

        self.found_rules = []
        self.rules_dict = {}
        self.output_dir = outputpath
        self.ts_range = ts_range
        self.ts_shift = np.ceil(self.ts_range/10)
        self.min_confidence = min_conf
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


    def create_rule(self, walk):
        """
        Create a rule given a cyclic temporal random walk.
        The rule contains information about head relation, body relations,
        variable constraints, confidence, rule support, and body support.
        A rule is a dictionary with the content
        {"head_rel": int, "body_rels": list, "var_constraints": list,
         "conf": float, "rule_supp": int, "body_supp": int}

        Parameters:
            walk (dict): cyclic temporal random walk
                         {"entities": list, "relations": list, "timestamps": list}

        Returns:
            rule (dict): created rule
        """

        rule = {}
        rule["head_rel"] = int(walk["relations"][0])
        rule["body_rels"] = [
            (x + self.num_relations if x < self.num_relations else x - self.num_relations) for x in walk["relations"][1:][::-1]
        ]
        rule["var_constraints"] = self.define_var_constraints(
            walk["entities"][1:][::-1]
        )

        if rule not in self.found_rules:
            self.found_rules.append(rule.copy())
            (
                rule["conf"],
                rule["rule_supp"],
                rule["body_supp"],
                rule["ts_gap"],
            ) = self.estimate_confidence(rule)

            if rule["conf"]:
                self.update_rules_dict(rule)

    def define_var_constraints(self, entities):
        """
        Define variable constraints, i.e., state the indices of reoccurring entities in a walk.

        Parameters:
            entities (list): entities in the temporal walk

        Returns:
            var_constraints (list): list of indices for reoccurring entities
        """

        var_constraints = []
        for ent in set(entities):
            all_idx = [idx for idx, x in enumerate(entities) if x == ent]
            var_constraints.append(all_idx)
        var_constraints = [x for x in var_constraints if len(x) > 1]

        return sorted(var_constraints)

    def estimate_confidence(self, rule, num_samples=500):
        """
        Estimate the confidence of the rule by sampling bodies and checking the rule support.

        Parameters:
            rule (dict): rule
                         {"head_rel": int, "body_rels": list, "var_constraints": list}
            num_samples (int): number of samples

        Returns:
            confidence (float): confidence of the rule, rule_support/body_support
            rule_support (int): rule support
            body_support (int): body support
        """

        all_bodies = []
        for _ in range(num_samples):
            sample_successful, body_ents_tss = self.sample_body(
                rule["body_rels"], rule["var_constraints"]
            )
            if sample_successful:
                all_bodies.append(body_ents_tss)

        all_bodies.sort()
        unique_bodies = list(x for x, _ in itertools.groupby(all_bodies))
        body_support = len(unique_bodies)

        confidence, rule_support, ts_gap = 0, 0, 1
        if body_support:
            rule_support, ts_gap = self.calculate_rule_support(unique_bodies, rule["head_rel"])
            confidence = round(rule_support / body_support, 6)

        return confidence, rule_support, body_support, int(ts_gap)

    def sample_body(self, body_rels, var_constraints):
        """
        Sample a walk according to the rule body.
        The sequence of timesteps should be non-decreasing.

        Parameters:
            body_rels (list): relations in the rule body
            var_constraints (list): variable constraints for the entities

        Returns:
            sample_successful (bool): if a body has been successfully sampled
            body_ents_tss (list): entities and timestamps (alternately entity and timestamp)
                                  of the sampled body
        """

        sample_successful = True
        body_ents_tss = []
        cur_rel = body_rels[0]
        rel_edges = self.edges[cur_rel]
        next_edge = rel_edges[np.random.choice(len(rel_edges))]
        cur_ts = next_edge[3]
        cur_node = next_edge[2]
        body_ents_tss.append(next_edge[0])
        body_ents_tss.append(cur_ts)
        body_ents_tss.append(cur_node)

        for cur_rel in body_rels[1:]:
            next_edges = self.edges[cur_rel]
            mask = (next_edges[:, 0] == cur_node) * (next_edges[:, 3] >= cur_ts)
            filtered_edges = next_edges[mask]

            if len(filtered_edges):
                next_edge = filtered_edges[np.random.choice(len(filtered_edges))]
                cur_ts = next_edge[3]
                cur_node = next_edge[2]
                body_ents_tss.append(cur_ts)
                body_ents_tss.append(cur_node)
            else:
                sample_successful = False
                break

        if sample_successful and var_constraints:
            # Check variable constraints
            body_var_constraints = self.define_var_constraints(body_ents_tss[::2])
            if body_var_constraints != var_constraints:
                sample_successful = False

        return sample_successful, body_ents_tss

    def calculate_rule_support(self, unique_bodies, head_rel):
        """
        Calculate the rule support. Check for each body if there is a timestamp
        (larger than the timestamps in the rule body) for which the rule head holds.

        Parameters:
            unique_bodies (list): bodies from self.sample_body
            head_rel (int): head relation

        Returns:
            rule_support (int): rule support
        """

        rule_support = 0
        head_rel_edges = self.edges[head_rel]
        gap_list = []
        for body in unique_bodies:
            mask = (
                (head_rel_edges[:, 0] == body[0])
                * (head_rel_edges[:, 2] == body[-1])
                * (head_rel_edges[:, 3] > body[-2])
            )

            gap_list += (head_rel_edges[mask][:, 3] - body[-2]).tolist()

            if True in mask:
                rule_support += 1

        gap_list.sort()
        if len(gap_list) > 0:
            ts_gap = int(np.floor(np.mean(gap_list))) + self.ts_shift
        else:
            ts_gap = self.ts_shift

        return rule_support, ts_gap

    def update_rules_dict(self, rule):
        """
        Update the rules if a new rule has been found.

        Parameters:
            rule (dict): generated rule from self.create_rule

        Returns:
            None
        """
        if rule['conf'] >= self.min_confidence:
            try:
                self.rules_dict[rule["head_rel"]].append(rule)
            except KeyError:
                self.rules_dict[rule["head_rel"]] = [rule]

    def sort_rules_dict(self):
        """
        Sort the found rules for each head relation by decreasing confidence.

        Parameters:
            None

        Returns:
            None
        """

        for rel in self.rules_dict:
            self.rules_dict[rel] = sorted(
                self.rules_dict[rel], key=lambda x: x["conf"], reverse=True
            )

    def save_rules(self, dt, rule_lengths, num_walks, transition_distr, seed):
        """
        Save all rules.

        Parameters:
            dt (str): time now
            rule_lengths (list): rule lengths
            num_walks (int): number of walks
            transition_distr (str): transition distribution
            seed (int): random seed

        Returns:
            None
        """

        rules_dict = {int(k): v for k, v in self.rules_dict.items()}
        # filename = "{0}_r{1}_n{2}_{3}_s{4}_rules.json".format(
        #     dt, rule_lengths, num_walks, transition_distr, seed
        # )
        # filename = filename.replace(" ", "")
        # with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as fout:
        #     json.dump(rules_dict, fout)
        #     fout.close()
        #
        # filename = 'tlogic_rules_{}.json'.format(num_walks)
        # with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as fout:
        #     json.dump(rules_dict, fout)
        #     fout.close()

        # os.mkdir(os.path.join(self.output_dir, str(seed)))
        filename = 'tlogic_rules_{}.txt'.format(num_walks)
        # filename = str(seed) + '/' + 'tlogic_rules_{}.txt'.format(num_walks)
        with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as fout:
            id = 0
            for hk in rules_dict:
                for item in rules_dict[hk]:
                    item['body_rels'] = [str(i) for i in item['body_rels']]
                    item['body_rels'] = ','.join(item['body_rels'])
                    item['var_constraints'] = [[str(i) for i in item] for item in item['var_constraints']]
                    item['var_constraints'] = [','.join(item) for item in item['var_constraints']]
                    item['var_constraints'] = ';'.join(item['var_constraints'])
                    fout.write(
                        '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                            id, item['head_rel'], item['body_rels'], item['var_constraints'], item['conf'], item['rule_supp'], item['body_supp']
                        )
                    )
                    id += 1
            fout.close()


def rules_statistics(rules_dict):
    """
    Show statistics of the rules.

    Parameters:
        rules_dict (dict): rules

    Returns:
        None
    """

    print(
        "Number of relations with rules: ", len(rules_dict)
    )  # Including inverse relations
    print("Total number of rules: ", sum([len(v) for k, v in rules_dict.items()]))

    lengths = []
    for rel in rules_dict:
        lengths += [len(x["body_rels"].split(',')) for x in rules_dict[rel]]
    rule_lengths = [(k, v) for k, v in Counter(lengths).items()]
    print("Number of rules by length: ", sorted(rule_lengths))
