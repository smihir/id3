from __future__ import division
import arff
import math
import copy
import operator
import itertools


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)



class Dt(Node):
    __raw_data = None
    __data = []
    __attribute_dictionary = {}
    __attribute_infogain = {}
    __continuous_attrbute_split = {}
    m = 2
    tree = None

    def __init__(self, arff_file, m):
        self.m = m

        with open(arff_file, 'rb') as f:
            self.__raw_data = arff.load(f, 'rb')

        # Get data in a cleaner format
        for d in self.__raw_data['data']:
            subject = {}
            for index, attribute in enumerate(self.__raw_data['attributes']):
                subject[attribute[0]] = d[index]
                self.__attribute_dictionary[attribute[0]] = attribute
            self.__data.append(subject)

        self.tree = Node(0)
        parent = self.tree
        self.__buidtree(self.__data, parent)

    def __is_nominal(self, attribute_name):
        datatype = self.__attribute_dictionary[attribute_name][1]
        if datatype == 'REAL' or datatype == 'NUMERIC':
            return False
        return True

    def __nominalval_infogain(self, attribute_name, data):
        if attribute_name == 'class':
            return

        # count of number of instances of each nominal value
        val_instances = [0] * len(self.__attribute_dictionary[attribute_name][1])
        # count of +ve number if instances of each nominal value
        pos_instances = [0] * len(self.__attribute_dictionary[attribute_name][1])
        # count of ive number if instances of each nominal value
        neg_instances = [0] * len(self.__attribute_dictionary[attribute_name][1])

        subset_entropy = [0] * len(self.__attribute_dictionary[attribute_name][1])

        for subject in data:
            val_instances[subject[attribute_name]] += 1
            if subject['class'] == 0:
                neg_instances[subject[attribute_name]] += 1
            else:
                pos_instances[subject[attribute_name]] += 1

        for i, v in enumerate(val_instances):
            if v == 0:
                subset_entropy[i] = 0
                continue

            pos_p = pos_instances[i] / v
            neg_p = neg_instances[i] / v

            if pos_p != 0:
                pos_entropy = pos_p * math.log(1 / pos_p, 2)
            else:
                pos_entropy = 0

            if neg_p != 0:
                neg_entropy = neg_p * math.log(1 / neg_p, 2)
            else:
                neg_entropy = 0

            subset_entropy[i] = pos_entropy + neg_entropy

        if sum(val_instances) != 0:
            all_pos_p = sum(pos_instances) / sum(val_instances)
            all_neg_p = sum(neg_instances) / sum(val_instances)
        else:
            all_pos_p = 0
            all_neg_p = 0

        if all_pos_p != 0:
            all_pos_entropy = all_pos_p * math.log(1 / all_pos_p, 2)
        else:
            all_pos_entropy = 0

        if all_neg_p != 0:
            all_neg_entropy = all_neg_p * math.log(1 / all_neg_p, 2)
        else:
            all_neg_entropy = 0

        all_entropy = all_pos_entropy + all_neg_entropy
        sum_entropy = 0
        for i, v in enumerate(val_instances):
            sum_entropy += (v / sum(val_instances)) * subset_entropy[i]

        self.__attribute_infogain[attribute_name] = all_entropy - sum_entropy

    def __split_infogain(self, attribute_name, left, right):
        val_instances = [0] * 2
        neg_instances = [0] * 2
        pos_instances = [0] * 2
        subset_entropy = [0] * 2

        for d in left:
            val_instances[0] += 1
            if d['class'] == 0:
                neg_instances[0] += 1
            else:
                pos_instances[0] += 1

        for d in right:
            val_instances[1] += 1
            if d['class'] == 0:
                neg_instances[1] += 1
            else:
                pos_instances[1] += 1

        for i, v in enumerate(val_instances):
            if v == 0:
                subset_entropy[i] = 0
                continue

            pos_p = pos_instances[i] / v
            neg_p = neg_instances[i] / v

            if pos_p != 0:
                pos_entropy = pos_p * math.log(1 / pos_p, 2)
            else:
                pos_entropy = 0

            if neg_p != 0:
                neg_entropy = neg_p * math.log(1 / neg_p, 2)
            else:
                neg_entropy = 0

            subset_entropy[i] = pos_entropy + neg_entropy

        if sum(val_instances) != 0:
            all_pos_p = sum(pos_instances) / sum(val_instances)
            all_neg_p = sum(neg_instances) / sum(val_instances)
        else:
            all_pos_p = 0
            all_neg_p = 0

        if all_pos_p != 0:
            all_pos_entropy = all_pos_p * math.log(1 / all_pos_p, 2)
        else:
            all_pos_entropy = 0

        if all_neg_p != 0:
            all_neg_entropy = all_neg_p * math.log(1 / all_neg_p, 2)
        else:
            all_neg_entropy = 0

        all_entropy = all_pos_entropy + all_neg_entropy
        sum_entropy = 0
        for i, v in enumerate(val_instances):
            sum_entropy += (v / sum(val_instances)) * subset_entropy[i]

        return all_entropy - sum_entropy

    def __continuous_infogain(self, attribute_name, data):
        data.sort(key=operator.itemgetter(attribute_name))

        groups = []
        C = []
        for k, g in itertools.groupby(data, operator.itemgetter(attribute_name)):
            groups.append(list(g))

        for g1, g2 in zip(groups, groups[1:]):
            g1_class_set = []
            g2_class_set = []
            for v in g1:
                g1_class_set.append(v['class'])
            for v in g2:
                g2_class_set.append(v['class'])

            # don't split if all values are equal
            if set(g1_class_set) == set(g2_class_set):
                if len(set(g1_class_set)) == 1:
                    continue
            C.append((g1[0][attribute_name] + g2[0][attribute_name]) / 2)

        # check how to calculate entropy when there is only 1 group
        if len(groups) == 1:
            for g in groups:
                C.append(g[0])

        infogain = []
        for candidate in C:
            left = [d for d in data if d[attribute_name] <= candidate]
            right = [d for d in data if d[attribute_name] > candidate]
            infogain.append(self.__split_infogain(attribute_name, left, right))

        if len(infogain) == 0:
            print 'staahp'

        index, max_val = max(enumerate(infogain), key=operator.itemgetter(1))

        self.__attribute_infogain[attribute_name] = max_val
        self.__continuous_attrbute_split[attribute_name] = C[index]

    def __calculate_infogain(self, attribute_name, data):
        if self.__is_nominal(attribute_name):
            self.__nominalval_infogain(attribute_name, data)
        else:
            self.__continuous_infogain(attribute_name, data)

    def __buidtree(self, clean_data, parent):

        # Base cases:
        # (ii) there are fewer than m training instances reaching the node
        if len(clean_data) <= self.m:
            print 'make leaf not node'
            return

        # clean data structures, to be used in recursion
        self.__continuous_attrbute_split.clear()
        self.__attribute_infogain.clear()
        data = copy.deepcopy(clean_data)

        attribute_dictionary = {}

        for key, value in self.__attribute_dictionary.iteritems():
            if data[0].has_key(key):
                attribute_dictionary[key] = value

        # (i) all of the training instances reaching the node belong to the same class
        if len(set([d['class'] for d in data])) == 1:
            print 'make leaf not node'
            return
        # (iii) no feature has positive information gain
        for k, v in attribute_dictionary.iteritems():
            self.__calculate_infogain(k, data)
        max_infogain = max(self.__attribute_infogain.iteritems(), key=operator.itemgetter(1))
        if max_infogain < 0:
            print 'make leaf not node'
            return
        print max_infogain
        # (iv) there are no more remaining candidate splits at the node.
        if len(self.__continuous_attrbute_split) == 0:
            print 'make leaf not node'
            return

        # make node not leaf
        # parent.add_child(self.__tree, max_infogain)

        # recurse
        if self.__is_nominal(max_infogain[0]):
            data.sort(key=operator.itemgetter(max_infogain[0]))
            groups = []
            for k, g in itertools.groupby(data, operator.itemgetter(max_infogain[0])):
                groups.append(list(g))

            for g in groups:
                for d in g:
                    d.pop(max_infogain[0], None)
                    child = Node(max_infogain[0])
                    parent.add_child(child)
                self.__buidtree(g, child)
        else:
            data.sort(key=operator.itemgetter(max_infogain[0]))

            for k, d in enumerate(data):
                if d[max_infogain[0]] > self.__continuous_attrbute_split[max_infogain[0]]:
                    index = k
                    break

            groups = list()
            groups.append(data[:index])
            groups.append(data[index:])
            for g in groups:
                for d in g:
                    d.pop(max_infogain[0], None)
                    child = Node(max_infogain[0])
                    parent.add_child(child)
                self.__buidtree(g, self.tree)

    def print_tree(self, Node, level):
        for i in range(level):
            
        print Node.data
        newlevel = level + 1
        for c in Node.children:
            self.print_tree(c, newlevel)


if __name__ == "__main__":
    print 'Start'
    dt = Dt('heart_train.arff', 2)
    dt.print_tree(dt.tree, 0)
    print 'OK'