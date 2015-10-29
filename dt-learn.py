from __future__ import division
import arff
import math
import copy
import operator
import itertools
import sys
import random


class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)


class Dt(Node):

    def __init__(self, raw_data, m):
        self.__raw_data = None
        self.__data = list()
        self.__attribute_dictionary = dict()
        self.__attribute_infogain = dict()
        self.__continuous_attrbute_split = dict()
        self.tree = None

        self.m = m

        self.__raw_data = copy.deepcopy(raw_data)

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

        index, max_val = max(enumerate(infogain), key=operator.itemgetter(1))

        self.__attribute_infogain[attribute_name] = max_val
        self.__continuous_attrbute_split[attribute_name] = C[index]

    def __calculate_infogain(self, attribute_name, data):
        if self.__is_nominal(attribute_name):
            self.__nominalval_infogain(attribute_name, data)
        else:
            self.__continuous_infogain(attribute_name, data)

    def __buidtree(self, clean_data, parent):

        # clean data structures, to be used in recursion
        self.__continuous_attrbute_split.clear()
        self.__attribute_infogain.clear()
        data = copy.deepcopy(clean_data)

        attribute_dictionary = {}

        for key, value in self.__attribute_dictionary.iteritems():
            if data[0].has_key(key):
                attribute_dictionary[key] = value

        # Base cases:
        # (i) all of the training instances reaching the node belong to the same class
        if len(set([d['class'] for d in data])) == 1:
            return
        # (ii) there are fewer than m training instances reaching the node
        if len(clean_data) < self.m:
            return
        # (iii) no feature has positive information gain
        max_infogain_list = list()
        for k, v in attribute_dictionary.iteritems():
            self.__calculate_infogain(k, data)
        max_infogain_list.append(max(self.__attribute_infogain.iteritems(), key=operator.itemgetter(1)))

        for key, value in self.__attribute_infogain.iteritems():
            if value == max_infogain_list[0][1] and key != max_infogain_list[0][0]:
                max_infogain_list.append([key, value])

        key_index = list()
        for val in max_infogain_list:
            key = val[1]
            for k, v in enumerate(self.__raw_data['attributes']):
                if v[0] == val[0]:
                    key_index.append(k)
        min_index = min(key_index)

        attr_name = self.__raw_data['attributes'][min_index]

        for m in max_infogain_list:
            if m[0] == attr_name[0]:
                max_infogain = m
                break

        if max_infogain < 0:
            return
        # (iv) there are no more remaining candidate splits at the node.
        if len(self.__attribute_infogain) == 0:
            return

        attribute_infogain = copy.deepcopy(self.__attribute_infogain)
        continuous_attrbute_split = copy.deepcopy(self.__continuous_attrbute_split)

        # make node not leaf
        # parent.add_child(self.__tree, max_infogain)

        # recurse
        if self.__is_nominal(max_infogain[0]):
            data.sort(key=operator.itemgetter(max_infogain[0]))

            groups = []
            nom_to_index = []
            for k, g in itertools.groupby(data, operator.itemgetter(max_infogain[0])):
                groups.append(list(g))
                nom_to_index.append(k)

            missing_index = list()
            if len(groups) != len(attribute_dictionary[max_infogain[0]][1]):
                for i in range(len(attribute_dictionary[max_infogain[0]][1])):
                    if nom_to_index.count(i) == 0:
                        missing_index.append(i)

            for i in missing_index:
                node = dict()
                node['key'] = max_infogain[0]
                node['count'] = [0, 0]
                node['value'] = self.__attribute_dictionary[max_infogain[0]][1][i]
                node['cvalue'] = i
                node['negpos'] = self.__attribute_dictionary['class'][1][0]
                parent.add_child(Node(node))
                attribute_dictionary[max_infogain[0]][1][i]

            for g in groups:
                count = list()
                count.append([d['class'] for d in g].count(0))
                count.append([d['class'] for d in g].count(1))
                node = dict()
                node['key'] = max_infogain[0]
                node['count'] = count
                node['data'] = g
                for d in g:
                    node['value'] = self.__attribute_dictionary[max_infogain[0]][1][d[max_infogain[0]]]
                    node['cvalue'] = d[max_infogain[0]]
                    d.pop(max_infogain[0], None)
                child = Node(node)
                parent.add_child(child)
                self.__buidtree(g, child)
        else:
            data.sort(key=operator.itemgetter(max_infogain[0]))
            for k, d in enumerate(data):
                if d[max_infogain[0]] > continuous_attrbute_split[max_infogain[0]]:
                    index = k
                    break

            groups = list()
            groups.append(data[:index])
            groups.append(data[index:])
            i = 0
            for g in groups:
                count = list()
                count.append([d['class'] for d in g].count(0))
                count.append([d['class'] for d in g].count(1))
                node = dict()
                node['key'] = max_infogain[0]
                node['value'] = continuous_attrbute_split[max_infogain[0]]
                node['cvalue'] = continuous_attrbute_split[max_infogain[0]]
                node['count'] = count
                node['eq'] = i
                node['data'] = g
                #for d in g:
                #    d.pop(max_infogain[0], None)
                child = Node(node)
                parent.add_child(child)
                self.__buidtree(g, child)
                i += 1

    def print_tree(self, Node, level):
        if level != -1:
            for i in range(level):
                print "|    ",

            posneg = ''
            if len(Node.children) == 0:
                if Node.data.has_key('data'):
                    #print 'Node has data!'
                    posneg = ': ' + self.__attribute_dictionary['class'][1][0] if Node.data['count'][0] >= Node.data['count'][1] else ': ' + self.__attribute_dictionary['class'][1][1]
                else:
                    #print 'sowee Node has no data!'
                    posneg = ': ' + Node.data['negpos']
            if self.__is_nominal(Node.data['key']):
                print Node.data['key'] + ' = ' + str(Node.data['value']) + ' [' + str(Node.data['count'][0])\
                  + ' ' + str(Node.data['count'][1]) + ']' + posneg
            else:
                cmp_str = ' <= ' if Node.data['eq'] == 0 else ' > '
                print Node.data['key'] + cmp_str + str(Node.data['value']) + ' [' + str(Node.data['count'][0])\
                  + ' ' + str(Node.data['count'][1]) + ']' + posneg

        newlevel = level + 1
        for c in Node.children:
            self.print_tree(c, newlevel)

    def __predict(self, index, data, parent):
        if len(parent.children) == 0:
            posneg = self.__attribute_dictionary['class'][1][0] if data['class'] == 0 else self.__attribute_dictionary['class'][1][1]
            if parent.data.has_key('data'):
                node_posneg = ': negative' if parent.data['data'][0]['class'] == 0 else ': positive'
            else:
                node_posneg = ': ' + parent.data['negpos']
            print str(index + 1) + ' Actual: ' + posneg + ' Predicted: ' + node_posneg

            if not parent.data.has_key('data'):
                return parent.data['cvalue']

            if data['class'] != parent.data['data'][0]['class']:
                return 0
            else:
                return 1

        new_parent = None
        for child in parent.children:
            if self.__is_nominal(child.data['key']):
                if data[child.data['key']] == child.data['cvalue']:
                    #print 'match ' + child.data['value'] + ' ' + child.data['key']
                    new_parent = child
            else:
                key_value = data[child.data['key']]
                candidate_split = child.data['cvalue']
                if child.data['eq'] == 0 and key_value <= candidate_split:
                    new_parent = child
                if child.data['eq'] == 1 and key_value > candidate_split:
                    new_parent = child
        if new_parent:
            return self.__predict(index, data, new_parent)
        else:
            print 'No Parent, how did I reach here :('
            return 0

    def predict(self, arff_file):
        with open(arff_file) as f:
            test_data = arff.load(f, 'rb')

        # Get data in a cleaner format
        clean_test_data = list()
        for d in test_data['data']:
            subject = {}
            for index, attribute in enumerate(self.__raw_data['attributes']):
                subject[attribute[0]] = d[index]
            clean_test_data.append(subject)

        v = 0
        for k, d in enumerate(clean_test_data):
            v += self.__predict(k, d, self.tree)
        print 'Number of correctly classified: ' + str(v) + ' Total number of test instances: ' + str(len(clean_test_data))


def test2():
    print 'Running test2'


    for i in range(10):
        print 'Iteration: ' + str(i + 1)
        with open(sys.argv[1], 'rb') as f:
            raw_data = arff.load(f, 'rb')

        fract = sys.argv[5]
        capture = int(len(raw_data['data']) * (int(fract) / 100))

        random.seed(i)
        pruned_data = list()
        index_tracker = dict()
        for k in range(capture):
            index = random.randint(0, len(raw_data['data']) - 1)
            pruned_data.append(raw_data['data'][index])
            del raw_data['data'][index]

        new_data = dict()
        for key, value in raw_data.iteritems():
            if key == 'data':
                new_data['data'] = copy.deepcopy(pruned_data)
                continue
            new_data[key] = copy.deepcopy(value)

        print 'Data Size: ' + str(len(new_data['data']))

        dt = Dt(new_data, int(sys.argv[3]))
        dt.print_tree(dt.tree, -1)
        dt.predict(sys.argv[2])
        del dt
        print '------------------------------'


if __name__ == "__main__":

    if len(sys.argv) == 6 and int(sys.argv[4]) == 1:
        test2()
    else:
        with open(sys.argv[1], 'rb') as f:
            raw_data = arff.load(f, 'rb')
        dt = Dt(raw_data, int(sys.argv[3]))
        dt.print_tree(dt.tree, -1)
        dt.predict(sys.argv[2])
