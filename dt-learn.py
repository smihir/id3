__author__ = 'Sejal Chauhan'
__author_email__ = 'sejalc@cs.wisc.edu'
__version__ = '1.0'


import math
import numpy
import scipy
import copy
import sys
import arff
import collections
import operator


from collections import namedtuple
attri = namedtuple('attri', ['Name', 'Nominal', 'NominalValue'])

# Populate the attribute dictionary
def attributeArray(attributes):
    attributeElements = []
    for attr in attributes:
        attrName = attr[0]
        isAttrNominal = None
        nominalValues = None
        if (attr[1] == "REAL" or attr[1] == "NUMERIC"):
            isAttrNominal = False
        else:
            isAttrNominal = True
            nominalValues = attr[1]
        attributeElements.append(attri(attrName, isAttrNominal, nominalValues))
    return attributeElements

# Populate the instance values
def convertData(attributeElements, data):
    tempData = [list() for _ in attributeElements]
    for item in data:
        for index, value in enumerate(item):
            tempData[index].append(value)
    return(tempData)


# do not consider class while calculating the entropy of each feature

#find the entropy of the instances in class
def AllEntropy(attrinstances):
    neg_count = 0
    pos_count = 0
    pos_entro = 0
    neg_entro = 0
    entrop = 0
    for element in attrinstances:
        if element[len(element)-1] == 0:
            neg_count += 1
        else:
            pos_count += 1
    total = neg_count + pos_count
    if float(total) != 0:
        pos_entro = pos_count/float(total)
    if pos_entro != 0:
        pos_entro = pos_entro * (math.log((pos_entro), 2))
    else:
        pass
        #makeleaf
    if float(total) != 0:
        neg_entro = neg_count/float(total)
    else:
        #makeleaf
        pass
    if neg_entro !=0:
        neg_entro = neg_entro * (math.log((neg_entro), 2))
    entrop = (-1) * (pos_entro + neg_entro)
    print "Entrop: " + str(entrop)
    return entrop

#find the entropy of the instances in class
def EachEntropy(element, attrinstances, m, arffdata):
    attri = {}
    n = 0
    entrop = []
    Entro = 0
    for each in element.NominalValue:
        attri[n] = []
        n += 1
    for ele in arffdata:
        attri[ele[m]].append(ele)
    for num in range(n):
        entrop.append(AllEntropy(attri[num]))
        Entro += entrop[num]

    print "Entro: " + str(Entro)
    return Entro

#calculate the entropy of each feature sans the class
def EntroyCal(arffelements, attrinstances, arffdata, all_entropy):
    featureentrop = 0
    maxinfogain = 0
    maxelem = 0
    infogain = []
    i = 0
    for element in arffelements:
        if element.Nominal ==True:
            for each in attrinstances:
                entrop =  EachEntropy(element, attrinstances, i, arffdata)
                infogain.append(InfoGainCal(entrop, all_entropy))
        elif element.Nominal == False:
            entrop = CandidateSplit(arffelements[i], attrinstances, arffdata, i)
            infogain.append(InfoGainCal(entrop, all_entropy))
        i = i +1
    infogain.index(min(infogain))
    print i
    return infogain.index(min(infogain))

#calculate the information gain for each feature
def InfoGainCal(feature_entropy, all_entropy):
    info_gain = float(all_entropy - feature_entropy)
    return info_gain

def CandidateSplit(numfeature, attrinstances, arffdata, i):
    arffdata.sort(key = operator.itemgetter(i))
    set = {}
    canditate_threshold = []
# iterate through the list to find instances with same value for attribute
    for key in range(len(arffdata)):
        if (arffdata[key][i] == arffdata[key+1][i]):
            set[arffdata[key]].append(arffdata[key][i])
        else:
            set[str(arffdata[key][i])].append(arffdata[key][i])


    entrop =  AllEntropy(attrinstances)
    return entrop

#def StoppingCriteria(arffinstances):
#all of the training instances reaching the node belong to the same class
#there are fewer than m training instances reaching the node
#no feature has positive information gain
#there are no more remaining candidate splits at the node


class Node(object):
    def __init__(self, label, left = None, right = None):
        self.label = label
        self.left = left
        self.right = right

    def traverse(rootnode):
        this = rootnode
        while this:
            next = list()
            for n in this:
                print n.value,
                if n.left: next.append(n.left)
                if n.right: next.append(n.right)
            print
            this = next
'''
t = Node()
traverse(t)
'''

def main():
    print ("Decision Tree")
    #getting training set
    #fp = open(sys.argv[1])

    #getting test set
    #fpt = open(sys.argv[2])
    #data_test = arff.load(fpt, 'rb')
    #fpt.close()
    #m = int(sys.agrv[3])

    fp = open('/Users/sejalc/Projects/ml/heart_train.arff', 'rb')
    data = arff.load(fp, 'rb')
    fp.close()

    # get attribute and data from the arff file
    arffAttributes = data['attributes']
    arffData = data['data']

    # transform the structure into more usage form
    attrElem = attributeArray(arffAttributes)
    attrInstances = convertData(arffAttributes, arffData)

    #calculate the total entropy of the class
    all_entropy = AllEntropy(arffData)

    #calculate all the feature's entropy and return the feature for
    next_node = EntroyCal(attrElem, attrInstances, arffData, all_entropy)

    #check if the stopping criteria has been met
    #StoppingCriteria()

    #make the root node

    #make next node
    print('---')
    return


main()





'''
def MakeSubTree():
for row in data:
    data = (data.get('data'))
    #print attribute
    for item in data:
        for i in item:
            item = (data[0])
            print item

MakeSubTree()

CalculateConditionalEntropy()
CalInfoGain()

C = DetermineCandidateSplit()

if m < atoi(str)
    MakeLeaf()
    DetermineProbability()
else
    MakeNode()

'''
