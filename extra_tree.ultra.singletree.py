
import numpy as np
from numpy import random

from sklearn.datasets import load_boston, make_friedman1
from copy import copy

"""
data type
=================================================
"""
class ExtraTree():

    def __init__(self):
        pass

NODE_LIST = []

class ExtraTreeConfig():

    def __init__(self, n_attrs, target_index, target_sq_index, K, min_node_samples):
        self.n_attrs = n_attrs
        self.target_index = target_index
        self.target_sq_index = target_sq_index
        self.K = K
        self.min_node_samples = min_node_samples

class ExtraTreeNode():

    def __init__(self, samples, variance, variance_val ):
        self.samples = samples
        self.variance = variance
        self.variance_val = variance_val
        self.spliter = None
        self.attr_index = None
        self.node_index = None
        self.leaf_flag = False
        
        self.targets = None
        self.targets_avg = None

class SplitResult():

    def __init__(self, attr_index, spliter, left_variance, right_variance, left_subsamples, right_subsamples, ):
        self.left_variance=left_variance
        self.left_subsamples=left_subsamples
        self.right_variance=right_variance
        self.right_subsamples=right_subsamples
        self.attr_index = attr_index
        self.spliter = spliter

        # if len(left_subsamples)==0:
        #     print spliter, np.asarray(right_subsamples)[:,attr_index]
        # if len(right_subsamples)==0:
        #     print spliter, np.asarray(left_subsamples)[:,attr_index]

        self.left_variance_val = calc_variance_val(left_variance)
        self.right_variance_val = calc_variance_val(right_variance)
        self.lr_variance_val = self.left_variance_val + self.right_variance_val


"""
utils function
=================================================
"""
def compare_score(split_result_a, split_result_b):
    if split_result_a.lr_variance_val < split_result_b.lr_variance_val:
        return 1
    else:
        return 0

def calc_variance_val(va):
    # print "calc_variance_val: ",va
    return va[1] - va[0]**2/va[2]

def add_variance(va, vb):
    va[0] += vb[0]
    va[1] += vb[1]
    va[2] += vb[2]

def sub_variance(va, vb):
    va[0] -= vb[0]
    va[1] -= vb[1]
    va[2] -= vb[2]

def random_pick_attr_splits(node, config, attr_index):
    samples = node.samples
    ind_middle = random.random_integers(0, len(samples)-2 )
    ind_a = random.random_integers(0, ind_middle )
    ind_b = random.random_integers(ind_middle+1, len(samples)-1 )
    # print "a:",ind_a, samples[ind_a][attr_index]
    # print "b:",ind_b, samples[ind_b][attr_index]

    spliter = (samples[ind_a][attr_index] + samples[ind_b][attr_index])*0.5
    return spliter


def random_pick_k_attrs(node, config):
    return random.choice(config.n_attrs, config.K, replace=False)



"""
node op
=================================================
"""

def gen_left_node(cur_split_result):
    new_node = ExtraTreeNode(
        cur_split_result.left_subsamples, 
        cur_split_result.left_variance, 
        cur_split_result.left_variance_val)

    append_node_list(new_node)
    return new_node

def gen_right_node(cur_split_result):
    new_node = ExtraTreeNode(
        cur_split_result.right_subsamples, 
        cur_split_result.right_variance, 
        cur_split_result.right_variance_val)

    append_node_list(new_node)
    return new_node
    
def append_node_list(node):
    global NODE_LIST
    NODE_LIST.append(node)
    node.node_index = len(NODE_LIST)-1

def convert_node_leaf(node, config):
    node.leaf_flag = True
    # node.targets = 
    target_index = config.target_index
    node.targets = np.asarray(node.samples)[:,target_index]
    variance = node.variance
    node.targets_avg = variance[0]/variance[2]
    # node.targets_avg = np.mean(node.targets)

def after_split_node(node, cur_split_result):
    # free no more use samples pointer, since samples is split-up
    node.samples = None
    node.attr_index = cur_split_result.attr_index
    node.spliter = cur_split_result.spliter

def append_left_node(node, left_node):
    node.left_child = left_node

def append_right_node(node, right_node):
    node.right_child = right_node


"""
tree op
=================================================
"""
def gen_root_node(samples):
    new_node = ExtraTreeNode(samples, [0,0,0], -1)
    append_node_list(new_node)
    return new_node

def build_extratree(node, config):
    samples = node.samples
    n_samples = len(samples)
    n_attrs = config.K

    if n_samples <= config.min_node_samples:
        convert_node_leaf(node, config)
        return node

    k_attrs = random_pick_k_attrs(node, config)
    k_splits = [0]*n_attrs
    for i,attr_index in enumerate(k_attrs):
        spliter = random_pick_attr_splits(node, config, attr_index)
        k_splits[i] = spliter

    # DEBUG 
    # print k_attrs, k_splits

    # optimal, small*large or large*small, which iterate is fast?
    one_pass_mode = True
    if one_pass_mode:
        cur_split_result = split_samples_by_attr_one_pass(node, config, k_attrs, k_splits)
    else:
        attr_index = k_attrs[0]
        spliter = k_splits[0]
        cur_split_result = split_samples_by_attr_normal(node, config, attr_index, spliter)

        for k_index in xrange(1,n_attrs):
            attr_index = k_attrs[k_index]
            spliter = k_splits[k_index]
            
            # normal split 
            # temp_split_result = split_samples_by_attr_normal(node, config, attr_index, spliter)

            # optimal, exchange split
            temp_split_result = split_samples_by_attr_exchange(node, config, attr_index, spliter, cur_split_result)

            if compare_score(temp_split_result, cur_split_result) > 0:
                cur_split_result = temp_split_result


    after_split_node(node, cur_split_result)

    left_node = gen_left_node(cur_split_result)
    append_left_node(node, left_node)
    right_node = gen_right_node(cur_split_result)
    append_right_node(node, right_node)
    
    build_extratree(left_node, config)
    build_extratree(right_node, config)

def split_samples_by_attr_one_pass(node, config, k_attrs, k_splits):

    samples = node.samples
    n_samples = len(samples)
    target_index = config.target_index
    target_sq_index = config.target_sq_index
    K = config.K

    k_split_result_arr = [0]*K
    k_left_subsamples_arr = [ [0]*n_samples ]*K
    k_right_subsamples_arr = [ [0]*n_samples ]*K

    for row_index,row in enumerate(samples):
        for k_index,spliter in enumerate(k_splits):
            attr_index = k_attrs
            left_subsamples = k_left_subsamples_arr[k_index]
            if row[attr_index] < spliter:
                left_subsamples[1]

    return 

def split_samples_by_attr_normal(node, config, attr_index, spliter):
    samples = node.samples
    n_samples = len(samples)
    target_index = config.target_index
    target_sq_index = config.target_sq_index

    # original
    left_subsamples = []
    right_subsamples = []
    # optimal, large_scale_array is array for append data, it will first init with a very large size, and iterate with real length
    left_subsamples = [0]*n_samples
    right_subsamples = [0]*n_samples

    left_variance = [0,0,0]
    right_variance = [0,0,0]

    left_index = 0
    right_index = 0

    for row_index,row in enumerate(samples):
        # optimal, as prepared in dataset
        target = row[target_index]
        target_sq = row[target_sq_index]

        if row[attr_index] < spliter:
            left_variance[0] += target
            left_variance[1] += target_sq
            left_variance[2] += 1

            left_subsamples[left_index] = row
            left_index += 1
        else:
            right_variance[0] += target
            right_variance[1] += target_sq
            right_variance[2] += 1

            right_subsamples[right_index] = row
            right_index += 1

    left_subsamples = left_subsamples[0:left_index]
    right_subsamples = right_subsamples[0:right_index]

    split_result = SplitResult(attr_index, spliter, left_variance, right_variance, left_subsamples, right_subsamples)
    return split_result


def split_samples_by_attr_exchange(node, config, attr_index, spliter, last_split_result):
    n_samples = len(node.samples)
    target_index = config.target_index
    target_sq_index = config.target_sq_index

    left_subsamples = [0]*n_samples
    right_subsamples = [0]*n_samples

    last_left_subsamples = last_split_result.left_subsamples
    # print np.asarray(last_left_subsamples)
    last_right_subsamples = last_split_result.right_subsamples
    left_variance = copy(last_split_result.left_variance)
    right_variance = copy(last_split_result.right_variance)

    left_index = 0
    right_index = 0

    for row_index,row in enumerate(last_left_subsamples):
        if row[attr_index] < spliter:
            left_subsamples[left_index] = row
            left_index += 1
        else:
            target = row[target_index]
            target_sq = row[target_sq_index]
            
            left_variance[0] -= target
            left_variance[1] -= target_sq
            left_variance[2] -= 1

            right_variance[0] += target
            right_variance[1] += target_sq
            right_variance[2] += 1
            
            right_subsamples[right_index] = row
            right_index += 1

    for row_index, row in enumerate(last_right_subsamples):
        if row[attr_index] < spliter:
            target = row[target_index]
            target_sq = row[target_sq_index]
            
            left_variance[0] += target
            left_variance[1] += target_sq
            left_variance[2] += 1

            right_variance[0] -= target
            right_variance[1] -= target_sq
            right_variance[2] -= 1
            
            left_subsamples[left_index] = row
            left_index += 1
        else:
            right_subsamples[right_index] = row
            right_index += 1

    left_subsamples = left_subsamples[0:left_index]
    right_subsamples = right_subsamples[0:right_index]

    split_result = SplitResult(attr_index, spliter, left_variance, right_variance, left_subsamples, right_subsamples)

    return split_result

def print_extratree(node, depth, indent=2):
    big_indent = "".join([" "]*indent)
    if node.leaf_flag:
        print big_indent, "leaf: ", node.targets
    else:
        print big_indent,"node(depth:{}): ".format(depth)
        print_extratree(node.left_child, depth+1, indent+2)
        print_extratree(node.right_child, depth+1, indent+2)

def dumps_extratree():
    pass

def walk_extratree(root_node, pred_row):
    cur_node = root_node
    while not cur_node.leaf_flag:
        attr_index = cur_node.attr_index
        spliter = cur_node.spliter
        if pred_row[attr_index] < spliter:
            cur_node = cur_node.left_child
        else:
            cur_node = cur_node.right_child

    targets_avg = cur_node.targets_avg
    return targets_avg
    # print cur_node.samples



"""
unittest
=================================================
"""

def test_build_extratree():
    X, Y = load_toy_dataset()

    X = np.asarray(X)
    Y = np.asarray(Y, dtype=np.float32)

    Y2 = Y**2
    # XY = np.concatenate((X, [Y], [Y2]), axis=0)
    XY = np.column_stack((X, Y, Y2))
    samples = XY

    n_attrs = len(X[0])
    target_index = len(samples) - 2
    target_sq_index = len(samples) - 1
    K = n_attrs
    min_node_samples = 1
    config = ExtraTreeConfig(target_index, target_sq_index, K, min_node_samples)

    root_node = gen_root_node(samples)
    build_extratree(root_node, config)



"""
main and API
=================================================
"""
class UltraExtraTrees():

    def __init__(self, K=None, min_node_samples=1, n_estimator=10):
        self.K = K
        self.min_node_samples = min_node_samples
        self.n_estimator = n_estimator

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y, dtype=np.float32)

        Y2 = Y**2
        # XY = np.concatenate((X, [Y], [Y2]), axis=0)
        XY = np.column_stack((X, Y, Y2))
        print XY.shape
        samples = XY
        n_columns = len(XY[0])

        n_attrs = len(X[0])
        target_index = n_columns - 2
        target_sq_index = n_columns - 1
        K = n_attrs # K is different actually
        min_node_samples = self.min_node_samples
        config = ExtraTreeConfig(n_attrs, target_index, target_sq_index, K, min_node_samples)

        root_node = gen_root_node(samples)
        self.root_node = root_node

        build_extratree(root_node, config)
        return self

    def predict(self, X):
        Y = [0]*len(X)
        for i,row in enumerate(X):
            pred_y = walk_extratree(self.root_node, row)
            Y[i] = pred_y
        return Y

def load_dataset():
    boston = load_boston()
    data = boston.data

def load_toy_dataset():
    X, Y = make_friedman1(n_samples=200, n_features=15)
    # X = [
    #     [1,1,1,1,1],
    #     [2,2,2,2,2],
    #     [3,3,3,3,3],
    # ]
    # Y = [1.1,2.2,3.3]

    return np.asarray(X), np.asarray(Y)

def main():
    X, Y = load_toy_dataset()
    uext = UltraExtraTrees()
    uext.fit(X, Y)
    
    print_extratree(uext.root_node, 0, 0)
    pY = uext.predict(X)
    # print np.asarray(zip(Y, pY), dtype=np.float32)


if __name__ == '__main__':
    main()
    # test_build_extratree()


