
import numpy as np
from numpy import random

def Build_extra_tree_ensemble():
    pass

class ExtraTreeMaker():

    def average_outputs(self, Y):
        pass
    
    def GenExtraTree(self, node):
        

        self.select_attrs(node)

    def select_attrs(self, node):
        return random.sample(node, self.K)

    def function(self):
        pass


class Node():
    pass

class Config():


def GenExtraTree(node, config):
    X = node.X
    Y = node.Y

    max_leaf_size = config.max_leaf_size

    if len(X) < max_leaf_size:
        return X

    k_attr_arr = SelectAttrs(node, config)


def SelectAttrs(node, config):
    K = config.K
    attr_indices = np.arrange(node.X.shape[1])
    k_attr_arr = random.sample(attr_indices, K)
    
    return k_attr_arr

def PickRandomSplit(node, attr_i):
    X = node.X

    np.min()
    random.uniform(node, )

    return split




