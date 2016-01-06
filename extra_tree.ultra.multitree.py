
import numpy as np
from numpy import random



def MultiTree_build_n_extratree():

    samples = node.samples
    n_samples = len(samples)
    n_split = config.n_extratrees
    n_attrs = len(k_attrs)
    
    # get n_split for each attr in all extratrees 
    k_attrs = MultiTree_random_pick_n_split_for_k_attrs(node, config)
    k_attrs_n_split_arr, k_order_n_split_arr = MultiTree_random_pick_n_split_for_k_attrs(node, config, k_attrs)

    # init the first one
    k_sub_rowindex_arr_in_rank_order = [[]]*n_attrs
    # k_sub_rowindex_arr_in_split_order = [[]]*n_attrs

    arr_i = 0
    attr_i = k_attrs[arr_i]
    attr_n_split_arr = k_attrs_n_split_arr[arr_i]
    order_n_split_arr = k_order_n_split_arr[arr_i]
    cur_split_variance_arr, cur_sub_rowindex_arr = MultiTree_split_samples(node, config, attr_i, order_n_split_arr)
    
    # k_sub_rowindex_arr_in_split_order[attr_i] = change_split_arr_order(cur_sub_rowindex_arr, attr_n_split_arr)
    k_sub_rowindex_arr_in_rank_order[attr_i] = cur_sub_rowindex_arr
    cur_split_variance_arr = change_split_arr_order(cur_split_variance_arr, attr_n_split_arr)

    # keep track of each extratree best split
    cur_split_attr_arr = [attr_i]*n_split

    # loop = n_attrs(small) * n_samples(large), is this expensive or not?
    for arr_i in xrange(1, len(k_attrs)):
        attr_i = k_attrs[arr_i]
        attr_n_split_arr = k_attrs_n_split_arr[arr_i]
        order_n_split_arr = k_order_n_split_arr[arr_i]
        split_variance_arr, result_sub_rowindex_arr = MultiTree_split_samples(node, config, attr_i, order_n_split_arr)
        # k_sub_rowindex_arr_in_split_order[attr_i] = change_split_arr_order(result_sub_rowindex_arr, attr_n_split_arr)
        k_sub_rowindex_arr_in_rank_order[attr_i] = result_sub_rowindex_arr
        split_variance_arr = change_split_arr_order(split_variance_arr, attr_n_split_arr)
        

        # update cur_split_variance_arr
        for update_i,cur_split_variance in enumerate(cur_split_variance_arr):
            cur_split_variance = cur_split_variance_arr[update_i]
            x_split_variance = split_variance_arr[update_i]
            
            # less variance
            if (x_split_variance[0]+x_split_variance[1]) < (cur_split_variance[0]+cur_split_variance[1]) :
                cur_split_attr_arr[update_i] = attr_i
                cur_split_variance_arr[update_i] = x_split_variance

    # after get best split for each extratree, we split the samples
    k_sub_sample_index = []
    for split_index,attr_i in cur_split_attr_arr:
        order_index = k_attrs_n_split_arr[attr_i][split_index]
        left_sub_rowindex = range(0,order_index)
        right_sub_rowindex = range(order_index, n_split)
        
        sub_rowindex_arr = k_sub_rowindex_arr_in_rank_order[attr_i]
        left_sub_samples = []
        for index in left_sub_rowindex:
            for row_index in sub_rowindex_arr[index]:
                

        tree_index = split_index


def change_split_arr_order(arr, ):
    pass


def MultiTree_random_pick_k_attrs(node, config):
    # since we only care about all feature, so ...
    return np.arange(node.samples.shape[1])


def MultiTree_random_pick_n_split(node, config, k_attrs):
    samples = node.samples
    n_samples = len(samples)
    n_split = config.n_extratrees
    n_attrs = len(k_attrs)

    """ 
    k_attrs_n_split_arr: [split_index] = order_index
    k_order_n_split_arr: [order_index] = split_val
    """
    k_attrs_n_split_arr = [[0]*n_split]*n_attrs
    k_order_n_split_arr = [[0]*n_split]*n_attrs

    # for each attr, we build n_cut_point
    for arr_i,attr_i in enumerate(k_attrs):
        order_linklist = []

        # split_index is for keep track of each extratree
        for split_index in xrange(n_split):
            ia, ib = random.sample(np.arange(n_samples), 2)
            xa = samples[ia][attr_i]
            xb = samples[ib][attr_i]
            new_split_val = (xa+xb)/2

            # order insert
            insert_done = False
            for i,x in enumerate(order_linklist):
                if new_split_val<x[0]:
                    order_linklist.insert(i,[new_split_val, split_index])
                    insert_done = True
                    break
            if not insert_done:
                order_linklist.append(i, [new_split_val, split_index])

        attr_n_split_arr = k_attrs_n_split_arr[arr_i]
        order_n_split_arr = k_order_n_split_arr[arr_i]
        # sort order_linklist, and 
        for order_index, x in enumerate(order_linklist):
            attr_n_split_arr[x[1]] = order_index
            order_n_split_arr[order_index] = x[0]

    return k_attrs_n_split_arr, k_order_n_split_arr

def add_variance(va, vb):
    va[0] += vb[0]
    va[1] += vb[1]
    va[2] += vb[2]

def sub_variance(va, vb):
    va[0] -= vb[0]
    va[1] -= vb[1]
    va[2] -= vb[2]

def MultiTree_split_samples(node, config, attr_i, order_split_arr):
    samples = node.samples
    targets = node.targets
    n_split = len(order_split_arr)
    n_split_result = n_split + 1

    """
    [targets_sum, targets_sq_sum, n_target]
    """
    result_variance_arr = [ [0,0,0] ]*n_split_result
    result_sub_rowindex_arr = [ [] ]*n_split_result

    # so it can use result_index easily, plus order_split_arr is small
    temp_order_split_arr = order_split_arr + [order_split_arr[-1]+1]

    # calc each result variance and its sub rowindex of samples
    for row_index,row in enumerate(samples):
        attr_val = row[attr_i]
        for result_index, spliter in enumerate(temp_order_split_arr):
            if attr_val<spliter:
                temp_variance = result_variance_arr[result_index]
                
                target = targets[row_index]
                temp_variance[0] += target
                temp_variance[1] += target**2
                temp_variance[2] += 1
                
                result_sub_rowindex_arr[result_index].append(row_index)
                
                break

    # gen var_sr and var_sl for each split, and split here is by rank order 
    split_variance_arr = [ [0,0] ]*n_split
    result_sub_rowindex_arr_indices = [ [[],[]] ]*n_split
    cum_variance = [0,0,0]

    # sl first
    for split_index in xrange(n_split):
        add_variance(cum_variance, result_variance_arr[split_index])

        targets_sum = cum_variance[0]
        targets_sq_sum = cum_variance[1]
        n_target = cum_variance[2]
        
        variance = targets_sq_sum - targets_sum**2
        split_variance_arr[split_index][0] = variance
        # result_sub_rowindex_arr_indices[split_index][0] =  range(0,split_index)

    add_variance(cum_variance, result_variance_arr[-1])
    
    # then sr
    for split_index in xrange(n_split):
        sub_variance(cum_variance, result_variance_arr[split_index])
        
        targets_sum = cum_variance[0]
        targets_sq_sum = cum_variance[1]
        n_target = cum_variance[2]

        variance = targets_sq_sum - targets_sum**2
        split_variance_arr[split_index][1] = variance
        # result_sub_rowindex_arr_indices[split_index][1] = range(split_index+1,len(result_sub_rowindex_arr))

    return split_variance_arr, result_sub_rowindex_arr


