from IPython.display import display
from tqdm import tqdm
import global_variables as gv
import numpy as np
import pandas as pd
import math
import pickle

# display all columns in dataframe
pd.set_option('display.max_columns', None)

class Node:
    def __init__(self):
        self.name = ''           # name of the node
        self.label = ''          # class label if this is the leaf node
        self.test_cond = ''      # test condition
        self.parent = None
        self.child = []          # child nodes
        self.edge_label = []     # edge label for each child
        self.partition = None    # partition of the test condition
        self.leaf_node = False   # is this a leaf node?
        self.prob = -1           # probability > -1 if this is the leaf node
        self.tree_level = -1     # node's level in the tree
    
    def printNode(self):
        '''Display node's information.'''

        print('Node:', self.name)
        print('\nTree level:', self.tree_level)
        print('Leaf node:', self.leaf_node)
        print('Class label:', self.label)
        print('Test condition:', self.test_cond)
        print('Children: ', len(self.child))
        print('Edge labels:', self.edge_label)
        print('Prob:', self.prob)

        # display partition
        if self.partition is not None:
            display(self.partition)
    
    def drawNode(self, target_attr):
        '''
        Recursive function to display structure of the node.
        '''

        # stop if this is the leaf node
        # print class label and probability
        if self.leaf_node == True:
            print(target_attr + " '" + self.label + "' " + str(self.prob))
        else:
            # iterate over the child nodes
            for i, node in enumerate(self.child):
                # print '|' and a space
                for j in range(0, self.tree_level):
                    print('| ', end='')
                
                # print node's test condition
                print(self.test_cond + " = '" + str(self.edge_label[i]) + "': ", end='')
                if node.leaf_node == False:
                    print('')
                
                # display structure of the child node
                node.drawNode(target_attr)


class DecisionTree:
    def __init__(self, data, targetAttr, attrsList,
                 max_depth=7, min_samples_leaf=5, max_leaf_nodes=100):
        # initialize instance's variables
        self.size_of_tree = 0
        self.num_of_leaves = 0
        self.leaves = []
        self.tree_level = -1
        self.TARGET = targetAttr
        self.ATTRIBUTES = attrsList
        
        # hyper-parameters
        self.min_samples_leaf = min_samples_leaf  # minimum number of samples required to be at a leaf node
        self.max_leaf_nodes = max_leaf_nodes      # Grow a tree with max_leaf_nodes in best-first fashion
        self.max_depth = max_depth                # maximum depth of the tree
        
        # set overal entropy
        self.TARGET_INFO = self.expected_info(data[self.TARGET])
        
        # build tree
        print('Building a Decision Tree...')
        self.tree = self.buildTree(data)
        print('Completed training Decision Tree classifier.\n')
    
    def buildTree(self, data):
        '''Call recursive function to build the tree.'''
        return self.TreeGrowth(data, self.ATTRIBUTES)
    
    def drawTree(self):
        '''Call recursive function to display structure of the tree.'''

        # print tree's information
        print("DecisionTree: Loan Return Classifier\n")
        print("Number of Leaves:", self.num_of_leaves)
        print("Size of the tree:", self.size_of_tree)
        print("max_depth:", self.max_depth)
        print("min_samples_leaf:", self.min_samples_leaf)
        print("max_leaf_nodes:", self.max_leaf_nodes)
        print("")

        # output structure of the tree
        self.tree.drawNode(self.TARGET)
        print("")
    
    def getTree(self):
        '''Returns tree'''
        return self.tree
    
    def expected_info(self, y):
        '''
        Compute expected information needed.

        Parameter:
            y: a dataframe series

        Return the expected information.
        '''

        # get a count for each category
        label_dict = y.value_counts().to_dict()
        total = sum(label_dict.values())
        info = 0.0

        # compute expected information
        for value in label_dict.values():
            ratio = value / total
            info -= ratio * math.log(ratio,2)

        return info
    
    def info_gain(self, D_attr, attr_label):
        '''
        Compute information gain.
        
        Parameters:
            D_attr: partition of training data
            attr_label: feature's name

        Return information gain for the given attribue
        '''

        # get attribute's categories
        categories = D_attr[attr_label].unique()

        # compute total expected information for the given attribute
        gain = 0.0
        for cat in categories:
            temp = D_attr[D_attr[attr_label] == cat]
            gain += (temp.shape[0]/D_attr.shape[0]) * self.expected_info(temp[self.TARGET])

        # return information gain
        return self.TARGET_INFO - gain

    def find_best_split(self, E1, F1):
        '''
        Find the best split.

        Parameters:
            E1: training records
            F1: a set of attributes

        Return the best attribute for the next split.
        '''
        
        best_split = ''
        max_gain = None
        for i, attr in enumerate(F1):
            attr_gain = self.info_gain(E1[[attr, self.TARGET]], attr)

            if max_gain is None or max_gain < attr_gain:
                max_gain = attr_gain
                best_split = F1[i]
        
        return best_split
    
    def stopping_cond(self, E1, F1, depth):
        '''
        Terminate the tree-growing process by testing whether all the records
        have either the same class label or the same attribute values.
        Another way to terminate the recursive function is to test whether the 
        number of records has fallen below some minimum threshold.

        Parameters:
            E1: training records
            F1: a set of attributes

        Returns stoping condition (True or False)
        '''
        
        # set stopping condition to False
        stop_cond = False
        
        # stopping conditions
        no_samples = len(E1) == 0                                       # no samples left
        no_attributes = len(F1) == 0                                    # no attributes left
        same_label = len(E1[self.TARGET].unique()) == 1                 # all records have same class label
        below_min_samples_leaf = self.min_samples_leaf > E1.shape[0]    # the number of records has fallen below min_samples_leaf
        above_max_depth = depth >= self.max_depth                       # reach the number of max depth allowed
        
        # reach or exceed the max leaf nodes
        above_max_leaf_node = (self.max_leaf_nodes is not None and self.num_of_leaves >= self.max_leaf_nodes)
        
        if no_samples or no_attributes or same_label or \
           below_min_samples_leaf or above_max_depth or above_max_leaf_node:
            stop_cond = True
        else:
            # check if all records have same attribute value except for the target
            same_attr_val = True
            for attr in F1:
                if attr != self.TARGET and len(E1[attr].unique()) > 1:
                    same_attr_val = False
                    break
            stop_cond = same_attr_val

        return stop_cond

    def Classify(self, E1):
        '''
        Determine the class label to be assigned to a leaf node.

        Parameter:
            E1: training record or partition of training record

        Return a tuple of predicted label and corresponding probability
        '''

        # get value counts for each class label
        temp = E1[self.TARGET].value_counts().to_dict()
        total = sum(temp.values())

        # get class label that has highest probability
        pred_label = ''
        max_prob = 0
        for label in temp.keys():
            prob = round(temp[label] / total, 5)

            if max_prob == 0 or max_prob < prob:
                max_prob = prob
                pred_label = label
        
        if len(pred_label) == 0:
            display(E1)

        return (pred_label, max_prob)

    def createLeafNode(self, samples):
        # create leaf node
        leaf = Node()
        leaf.label, leaf.prob = self.Classify(samples)      # get class label and prob
        leaf.leaf_node = True                               # set node as a leaf node
        self.num_of_leaves += 1                             # increment number of leaves
        self.size_of_tree += 1                              # increment size of the tree
        self.leaves.append(leaf)                            # add leaf node to a list
        return leaf
        
    def TreeGrowth(self, E, F, depth=0):
        '''
        Recursive function for building a decision tree.
        
        Parameters:
            E: training data
            F: a list of features

        Return a node of the tree.
        '''
        
        if self.stopping_cond(E, F, depth) == True:
            # create a leaf node and assign class label
            return self.createLeafNode(E)
        else:
            # increment tree_level
            self.tree_level += 1

            # create tree's node
            root = Node()
            root.test_cond = self.find_best_split(E, F)   # get the best split attribute
            root.tree_level = depth                       # set node's level in the tree
            self.size_of_tree += 1                        # increment size of the tree
            
            # possible outcomes of test_cond
            V = list(E[root.test_cond].unique())
            
            # loop through each value in test_cond
            for v in tqdm(V):
                E_v = E[E[root.test_cond] == v]           # select partition for every test_cond value
                
                # get a list of available attributes for the given samples
                attributes = list(E_v.columns)
                attributes.remove(gv.TARGET)
                
                # create child node
                if len(E_v) == 0:
                    # convert node to a leaf node if there are no samples left
                    child = self.createLeafNode(E)
                    child.partition = E
                else:
                    child = self.TreeGrowth(E_v, attributes, depth+1)
                    child.partition = E_v
                    
                child.parent = root
                child.tree_level = root.tree_level + 1
                child.name = root.test_cond
                child.test_value = v
                
                # root
                root.child.append(child)         # add child as descendent
                root.edge_label.append(v)        # label the edge (root -> child)
            # -- end for loop
            
            # return node
            return root


def classifyLoan(dataIn, node):
    '''
    Recursive function for classifying a loan.
    
    Parameters:
    - dataIn: an example to be classified
    - node: current node
    
    Return a list of two items (class label and probability).
    '''
    
    prediction = []
    
    # if this is a leaf node, return class label and probability
    if node.leaf_node == True:
        return [node.label, node.prob]
    else:
        # look at the test condition and test value
        for i, test_val in enumerate(node.edge_label):
            # look at node's child for which the test condition is met
            if dataIn[node.test_cond] == test_val:
                out = classifyLoan(dataIn, node.child[i])
                
                # get the class label with highest probability
                if (len(out) > 0 and len(prediction) == 0) or \
                    (len(out) > 0 and len(prediction) > 0 and out[1] > prediction[1]):
                    prediction = out
    
    return prediction


def displayMetrics(dataIn):
    '''
    Display evaluation metrics.
    
    Parameter:
    - dataIn: predictions dataframe
    '''
    
    undefined = dataIn[dataIn['predicted'] == "Undefined"]
    if undefined.shape[0] > 0:
        print('Unable to classify ', undefined.shape[0] , ' loan(s):')
        display(undefined)
        print('')
    
    dataIn = dataIn[dataIn['predicted'] != "Undefined"]
    
    # get actual/predicted label
    actual = list(dataIn[gv.TARGET])
    predicted = list(dataIn['predicted'])
    
    # get a list of unique classes
    classes = set(actual)
    
    # set up the matrix
    matrix = [list() for x in range(len(classes))]
    for i in range(len(classes)):
        matrix[i] = [0 for x in range(len(classes))]
    
    # set up lookup dictionary for index of class label
    lookup = dict()
    for i, value in enumerate(classes):
        lookup[value] = i
        
    # build the matrix
    for i in range(len(list(dataIn[gv.TARGET]))):
        x = lookup[actual[i]]
        y = lookup[predicted[i]]
        matrix[x][y] += 1
    
    # build confusion matrix data frame
    confusion_matrix = pd.DataFrame(matrix, columns=classes, index=classes)
    
    # compute evaluation metrics
    accuracy = round(dataIn[dataIn[gv.TARGET] == dataIn['predicted']].shape[0] * 100 / dataIn.shape[0], 4)
    true_pos = np.diag(confusion_matrix)
    false_pos = np.sum(confusion_matrix, axis=0) - true_pos
    false_neg = np.sum(confusion_matrix, axis=1) - true_pos
    true_neg = confusion_matrix.values.sum() - (false_pos + false_neg + true_pos)
    precision = round(true_pos / (true_pos + false_pos), 2)
    recall = round(true_pos / (true_pos + false_neg), 2)
    f_score = round((2 * precision * recall) / (precision + recall), 2)
    tpr = round(true_pos / (true_pos + false_neg), 2)          # Sensitivity = TP/(TP+FN)
    tnr = round(true_neg / (true_neg + false_pos), 2)          # Specificity = TN/(TN+FP)
    fpr = round(false_pos / (true_neg + false_pos), 2)         # FPR = FP/(TN+FP)
    fnr = round(false_neg / (true_pos + false_neg), 2)         # FNR = FN/(TP+FN)
    
    # build classification report data frame
    metrics = ['TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-Measure']
    metrics_report = pd.DataFrame([tpr, fpr, tnr, fnr, precision, recall, f_score], index=metrics).T
    metrics_report.reset_index(inplace=True)
    
    # compute weighted average
    weighted_avg = [round(np.mean(metrics_report[x]), 2) for x in metrics]
    weighted_avg = ['Weighted Avg'] + weighted_avg
    metrics_report.loc[len(metrics_report)] = weighted_avg
    metrics_report.rename(columns = {'index':'Loan Return'}, inplace = True)
    
    # display evaluation metrics
    print('\nAccuracy:', accuracy)
    print('\nClassification Report:\n')
    display(metrics_report)
    
    print('\nConfusion Matrix:\n')
    display(confusion_matrix)