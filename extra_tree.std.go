package main

import (
    "math/rand"
    "time"
    
    "fmt"
    "strings"
    "github.com/kr/pretty"
)

/*
data type
=================================================
*/
type Data float32
type DataArray []float32
type DataArrayPointers []DataArray
func NewDataArray(length int) DataArray{
    p := make(DataArray, length)
    return p
}


type IndexArray []int
func NewIndexArray(length int) IndexArray{
    p := make(IndexArray, length)
    return p
}


type ExtraTreeConfig struct {
    n_attrs int 
    // target_index int 
    // target_sq_index int 
    K int
    min_node_samples int
    total_samples int
    target_data_arr DataArray
    target_sq_data_arr DataArray
    sample_data_mat DataArray
    X []DataArray // original data
}
func NewExtraTreeConfig(n_attrs int, K int, min_node_samples int) *ExtraTreeConfig{
    p := &ExtraTreeConfig{
        n_attrs: n_attrs,
        K: K,
        min_node_samples: min_node_samples,
    }

    return p
}

type ExtraTreeNode struct {
    samples IndexArray
    variance DataArray
    variance_val float32
    spliter float32
    attr_index int
    node_index int
    leaf_flag bool
    targets DataArray // disable it in production, if dont need it
    targets_avg float32

    left_child *ExtraTreeNode
    right_child *ExtraTreeNode
}
func NewExtraTreeNode(samples IndexArray, variance DataArray, variance_val float32) *ExtraTreeNode{
    p := &ExtraTreeNode{
        samples: samples,
        variance: variance,
        variance_val: variance_val,
    }
    return p
}

var NODE_LIST []*ExtraTreeNode


type SplitResult struct {
    left_variance DataArray
    left_subsamples IndexArray
    right_variance DataArray
    right_subsamples IndexArray
    attr_index int
    spliter float32

    left_variance_val float32
    right_variance_val float32
    lr_variance_val float32
}
func NewSplitResult(attr_index int, spliter float32, left_variance DataArray, right_variance DataArray, left_subsample_index_arr IndexArray, right_subsample_index_arr IndexArray) *SplitResult{

    left_variance_val := calc_variance_val(left_variance)
    right_variance_val := calc_variance_val(right_variance)
    p := &SplitResult{
        attr_index: attr_index,
        spliter: spliter,
        
        left_variance: left_variance,
        right_variance: right_variance,
        left_variance_val: left_variance_val,
        right_variance_val: right_variance_val,
        lr_variance_val: left_variance_val+right_variance_val,
        
        left_subsamples: left_subsample_index_arr,
        right_subsamples: right_subsample_index_arr,
    }
    return p
}



/*
utils function
=================================================
*/
func copy_data_arr(arr DataArray) DataArray{
    p := make(DataArray, len(arr))
    for i,x := range arr {
        p[i] = x
    }
    return p
}

func random_integer(min int, max int) int {
    return rand.Intn(max+1 - min) + min
}

func compare_score(split_result_a *SplitResult, split_result_b *SplitResult) int {
    if split_result_a.lr_variance_val < split_result_b.lr_variance_val {
        return 1
    } else {
        return 0
    }
}

func calc_variance_val(va DataArray) float32{
    // print "calc_variance_val: ",va
    return va[1] - va[0]*va[0]/va[2]
}

func random_pick_k_attrs(node *ExtraTreeNode, config *ExtraTreeConfig) IndexArray{
    arr := rand.Perm(config.n_attrs)
    return arr[0:config.K]
}

func random_pick_attr_splits(node *ExtraTreeNode, config *ExtraTreeConfig, attr_index int) float32{
    samples := node.samples
    n_samples := len(samples)
    sample_data_mat := config.sample_data_mat
    total_samples := config.total_samples
    attr_row_index := attr_index*total_samples

    ind_middle := random_integer(0, n_samples-2)
    ind_a := random_integer(0, ind_middle)
    ind_b := random_integer(ind_middle+1, n_samples-1)

    spliter := (sample_data_mat[attr_row_index+samples[ind_a]]+sample_data_mat[attr_row_index+samples[ind_b]])*0.5

    // pretty.Println(spliter, ind_a, sample_data_mat[attr_row_index+ind_a], ind_b, sample_data_mat[attr_row_index+samples[ind_b]])
    return spliter
}


/*
node op
=================================================
*/
func gen_left_node(cur_split_result *SplitResult) *ExtraTreeNode{
    new_node := NewExtraTreeNode(cur_split_result.left_subsamples, cur_split_result.left_variance, cur_split_result.left_variance_val)

    append_node_list(new_node)
    return new_node
}

func gen_right_node(cur_split_result *SplitResult) *ExtraTreeNode{
    new_node := NewExtraTreeNode(cur_split_result.right_subsamples, cur_split_result.right_variance, cur_split_result.right_variance_val)

    append_node_list(new_node)
    return new_node
}

func append_node_list(node *ExtraTreeNode) {
    NODE_LIST = append(NODE_LIST, node)
    node.node_index = len(NODE_LIST)-1
}


func convert_node_leaf(node *ExtraTreeNode, config *ExtraTreeConfig) {
    node.leaf_flag = true
    sample_index_arr := node.samples
    target_data_arr := config.target_data_arr

    targets := make(DataArray, len(sample_index_arr))
    for i,sample_index := range sample_index_arr {
        targets[i] = target_data_arr[sample_index]
    }
    node.targets = targets

    variance := node.variance
    node.targets_avg = variance[0]/variance[2]
}

func after_split_node(node *ExtraTreeNode, cur_split_result *SplitResult) {
    node.samples = nil
    node.attr_index = cur_split_result.attr_index
    node.spliter = cur_split_result.spliter
}

func append_left_node(node *ExtraTreeNode, left_node *ExtraTreeNode) {
    node.left_child = left_node
}
func append_right_node(node *ExtraTreeNode, right_node *ExtraTreeNode) {
    node.right_child = right_node
}




/*
tree op
=================================================
*/
func gen_root_node(samples IndexArray) *ExtraTreeNode {
    new_node := NewExtraTreeNode(samples, nil, -1)
    append_node_list(new_node)
    return new_node
}

func build_extratree(node *ExtraTreeNode, config *ExtraTreeConfig) *ExtraTreeNode{
    samples := node.samples
    n_samples := len(samples)
    K := config.K

    if n_samples <= config.min_node_samples {
        convert_node_leaf(node, config)
        return node
    }

    k_attrs := random_pick_k_attrs(node, config)
    k_splits := make(DataArray, K)
    for k_index,attr_index := range k_attrs {
        spliter := random_pick_attr_splits(node, config, attr_index)
        k_splits[k_index] = spliter
    }

    // init first one
    cur_split_result := split_samples_by_attr_mode(node, config, k_attrs[0], k_splits[0], nil)

    for k_index := 1; k_index < K; k_index++ {
        attr_index := k_attrs[k_index]
        spliter := k_splits[k_index]

        temp_split_result := split_samples_by_attr_mode(node, config, attr_index, spliter, nil)

        if compare_score(temp_split_result, cur_split_result) > 0 {
            cur_split_result = temp_split_result
        }
    }

    // pretty.Println(cur_split_result)

    after_split_node(node, cur_split_result)

    left_node := gen_left_node(cur_split_result)
    append_right_node(node, left_node)
    right_node := gen_right_node(cur_split_result)
    append_left_node(node, right_node)

    build_extratree(left_node, config)
    build_extratree(right_node, config)

    return node
}


func split_samples_by_attr_mode(node *ExtraTreeNode, config *ExtraTreeConfig, attr_index int, spliter float32, last_split_result *SplitResult) *SplitResult{
    
    sample_index_arr := node.samples
    n_samples := len(sample_index_arr)
    // K := config.K
    total_samples := config.total_samples
    // target_index := config.target_index
    // target_sq_index := config.target_sq_index
    
    sample_data_mat := config.sample_data_mat
    target_data_arr := config.target_data_arr
    target_sq_data_arr := config.target_sq_data_arr

    left_subsample_index_arr := make(IndexArray, n_samples)
    left_index := 0
    right_subsample_index_arr := make(IndexArray, n_samples)
    right_index := 0

    left_variance := make(DataArray, 3)
    right_variance := make(DataArray, 3)

    var cur_target float32
    var cur_target_sq float32
    attr_row_index := attr_index*total_samples

    // pretty.Println(spliter, sample_index_arr)

    if last_split_result != nil { // exchange 

        left_variance = copy_data_arr(last_split_result.left_variance)
        right_variance = copy_data_arr(last_split_result.right_variance)
        
        // last left 
        for _,sample_index := range last_split_result.left_subsamples {
            cur_target = target_data_arr[sample_index]
            cur_target_sq = target_sq_data_arr[sample_index]

            if sample_data_mat[attr_row_index+sample_index] < spliter { // ok
                left_subsample_index_arr[left_index] = sample_index
                left_index += 1
            } else { // should right
                left_variance[0] -= cur_target
                left_variance[1] -= cur_target_sq
                left_variance[2] -= 1

                right_variance[0] += cur_target
                right_variance[1] += cur_target_sq
                right_variance[2] += 1

                right_subsample_index_arr[right_index] = sample_index
                right_index += 1
            }
        }

        // last right 
        for _,sample_index := range last_split_result.right_subsamples {
            cur_target = target_data_arr[sample_index]
            cur_target_sq = target_sq_data_arr[sample_index]

            if sample_data_mat[attr_row_index+sample_index] < spliter { // should to left
                left_variance[0] += cur_target
                left_variance[1] += cur_target_sq
                left_variance[2] += 1

                right_variance[0] -= cur_target
                right_variance[1] -= cur_target_sq
                right_variance[2] -= 1

                left_subsample_index_arr[left_index] = sample_index
                left_index += 1
            } else { // ok
                right_subsample_index_arr[right_index] = sample_index
                right_index += 1
            }
        }

    } else { // normal 
        for _,sample_index := range sample_index_arr {
            cur_target = target_data_arr[sample_index]
            cur_target_sq = target_sq_data_arr[sample_index]

            // to left or right
            if sample_data_mat[attr_row_index+sample_index] < spliter { // to left
                left_variance[0] += cur_target
                left_variance[1] += cur_target_sq
                left_variance[2] += 1

                left_subsample_index_arr[left_index] = sample_index
                left_index += 1

            } else { // to right
                right_variance[0] += cur_target
                right_variance[1] += cur_target_sq
                right_variance[2] += 1

                right_subsample_index_arr[right_index] = sample_index
                right_index += 1
            }
        }
    }

    left_subsample_index_arr = left_subsample_index_arr[0:left_index]
    right_subsample_index_arr = right_subsample_index_arr[0:right_index]

    cur_split_result := NewSplitResult(attr_index, spliter, left_variance, right_variance, left_subsample_index_arr, right_subsample_index_arr)

    return cur_split_result
}

func walk_extratree(root_node *ExtraTreeNode, pred_row DataArray) float32{
    cur_node := root_node
    for cur_node.leaf_flag == false {
        attr_index := cur_node.attr_index
        spliter := cur_node.spliter
        if pred_row[attr_index] < spliter {
            cur_node = cur_node.left_child
        } else {
            cur_node = cur_node.right_child
        }
    }

    targets_avg := cur_node.targets_avg
    return targets_avg
}

func print_extratree(node *ExtraTreeNode, depth int, indent int) {
    arr := make([]string, indent)
    for i,_ := range arr {
        arr[i] = " "
    }
    big_indent := strings.Join(arr,"")

    if node == nil {
        return
    }

    if node.leaf_flag {
        fmt.Println(big_indent,"leaf: ",node.targets)
    } else {
        fmt.Printf("%snode(depth:%s): \n",big_indent,depth)
        depth += 1
        indent += 2
        print_extratree(node.left_child, depth, indent)
        print_extratree(node.right_child, depth, indent)
    }
}



/*
API
=================================================
*/
type ExtraTreesRegressor struct {
    K int
    min_node_samples int
    n_estimators int
    root_node *ExtraTreeNode
    config *ExtraTreeConfig
}
func NewExtraTreesRegressor(K int, min_node_samples int, n_estimators int) *ExtraTreesRegressor{
    //init 
    rand.Seed(time.Now().Unix())

    p := &ExtraTreesRegressor{
        K: K,
        min_node_samples: min_node_samples,
        n_estimators: n_estimators,
    }
    return p
}
func (ext *ExtraTreesRegressor) fit(X []DataArray, Y DataArray) *ExtraTreesRegressor{
    n_samples := len(X)
    n_attrs := len(X[0])
    K := n_attrs // just use all attrs, change it to K := ext.K, for sub feature space
    min_node_samples := ext.min_node_samples

    if n_samples != len(Y) {
        panic("Assertion: len(X) == len(Y)")
    }

    // transpose X
    XT := make(DataArray, K*n_samples)
    samples := make(IndexArray, n_samples) // as sample_index_arr
    for sample_index,row := range X {
        samples[sample_index] = sample_index
        for attr_index,column := range row {
            XT[attr_index*n_samples+sample_index] = column
        }
    }

    // gen target_sq_data_arr
    target_sq_data_arr := copy_data_arr(Y)
    for target_index,target := range Y {
        target_sq_data_arr[target_index] = target*target
    }

    // gen config
    config := NewExtraTreeConfig(n_attrs, K, min_node_samples)
    config.sample_data_mat = XT
    config.target_data_arr = Y
    config.target_sq_data_arr = target_sq_data_arr
    config.total_samples = n_samples
    config.X = X
    ext.config = config

    // start build extratree from root_node
    root_node := gen_root_node(samples)
    ext.root_node = root_node

    build_extratree(root_node, config)
    return ext
}
func (ext *ExtraTreesRegressor) predict(X []DataArray) (pY DataArray){
    
    return nil
}



/*
generate dataset
=================================================
*/
func make_friedman1(n_samples int, n_attrs int) (X []DataArray, Y DataArray){
    
    return nil,nil
}

func make_toy_dataset(n_samples int, n_attrs int) (X []DataArray, Y DataArray){

    X = make([]DataArray, n_samples)
    Y = make(DataArray, n_samples)
    var val float32
    for i := 0; i < n_samples; i++ {
        val = float32(i+1)
        X[i] = make(DataArray, n_attrs)
        temp_arr := X[i]
        for j,_ := range temp_arr {
            temp_arr[j] = val
        }
        Y[i] = val+0.2
    }

    return X, Y
}



/*
main unittest
=================================================
*/

func main() {

    X, Y := make_toy_dataset(100000, 30)
    start := time.Now()

    // pretty.Println(X, Y)
    ext := NewExtraTreesRegressor(1, 1, 1)
    ext.fit(X, Y)

    elapsed := time.Since(start)
    pretty.Printf("elapsed time: %.16f\n", elapsed.Seconds())

    // print_extratree(ext.root_node, 0, 0)

}







