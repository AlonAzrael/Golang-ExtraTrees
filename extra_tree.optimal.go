package main

import (
    "math/rand"
    "math"
    "time"

    // for debug    
    "fmt"
    "strings"
    "github.com/kr/pretty"

    // for profiling
    // "runtime/pprof"
    // "github.com/davecheney/profile"
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
    par_stack_arr []IndexArray // 0,1
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
    par_stack_number int
    par_stack_header_index int
    par_stack_footer_index int // footer always greater than header

    target_sum float32
    target_sq_sum float32
    n_target int
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

var NODE_LIST []*ExtraTreeNode




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


func calc_variance_val(va DataArray) float32{
    // print "calc_variance_val: ",va
    return va[1] - va[0]*va[0]/va[2]
}

func random_pick_k_attrs(node *ExtraTreeNode, config *ExtraTreeConfig) IndexArray{
    arr := rand.Perm(config.n_attrs)
    return arr[0:config.K]
}

func random_pick_attr_splits(node *ExtraTreeNode, config *ExtraTreeConfig, attr_index int) float32{
    // sample_data_mat := config.sample_data_mat
    // total_samples := config.total_samples
    // attr_row_index := attr_index*total_samples

    spliter := float32(0.5)

    // pretty.Println(spliter, ind_a, sample_data_mat[attr_row_index+ind_a], ind_b, sample_data_mat[attr_row_index+samples[ind_b]])
    return spliter
}


/*
node op
=================================================
*/

func append_node_list(node *ExtraTreeNode) {
    // NODE_LIST = append(NODE_LIST, node)
    // node.node_index = len(NODE_LIST)-1
}


func convert_node_leaf(node *ExtraTreeNode, config *ExtraTreeConfig) {
    
    
}




/*
tree op
=================================================
*/
func gen_root_node(config *ExtraTreeConfig) *ExtraTreeNode {
    new_node := &ExtraTreeNode{
        par_stack_number: 0,
        par_stack_header_index: 0,
        par_stack_footer_index: config.total_samples-1,
        leaf_flag: false,
    }
    append_node_list(new_node)
    return new_node
}


func build_extratree_loop(root_node *ExtraTreeNode, config *ExtraTreeConfig) *ExtraTreeNode{
    // point to a space of par_stack
    
    min_node_samples := config.min_node_samples
    // K := config.K
    total_samples := config.total_samples
    par_stack_arr := config.par_stack_arr
    sample_data_mat := config.sample_data_mat
    target_data_arr := config.target_data_arr
    target_sq_data_arr := config.target_sq_data_arr

    SPLIT_MODE_NORMAL := 0
    SPLIT_MODE_EXCHANGE := 1

    node_queue := make([]*ExtraTreeNode, total_samples*2-1)
    node_queue[0] = root_node
    cur_queue_index := 0
    end_queue_index := 1

// init variables for loop
var cur_node *ExtraTreeNode
var par_stack_header_index int
var par_stack_footer_index int
var n_samples int
var k_attrs IndexArray
// k_attrs = make(IndexArray, K)
// for i := 0; i < K; i++ {
//     k_attrs[i] = i
// }
// var k_splits DataArray
var cur_par_stack_number int
var cur_par_stack IndexArray
var next_par_stack_number int
var next_par_stack IndexArray

var spliter float32
split_mode := SPLIT_MODE_NORMAL
var left_index int
var right_index int
var ind_middle int
var ind_a int
var ind_b int

var attr_row_index int
var lr_variance_val float32
var left_variance_val float32
var left_target_sum float32
var left_target_sq_sum float32
var left_n_target int
var right_variance_val float32
var right_target_sum float32
var right_target_sq_sum float32
var right_n_target int

var cur_target float32
var cur_target_sq float32
var sample_index int

cur_lr_variance_val := float32(math.Inf(1))
var cur_attr_index int
var cur_spliter float32

var cur_left_variance_val float32
var cur_left_target_sum float32
var cur_left_target_sq_sum float32
var cur_left_n_target int
var cur_left_par_stack_header_index int
var cur_left_par_stack_footer_index int 

var cur_right_variance_val float32
var cur_right_target_sum float32
var cur_right_target_sq_sum float32
var cur_right_n_target int
var cur_right_par_stack_header_index int
var cur_right_par_stack_footer_index int

var left_node *ExtraTreeNode
var right_node *ExtraTreeNode



// loop until no more node to split
for cur_queue_index < end_queue_index {
    
    cur_node = node_queue[cur_queue_index]
    cur_queue_index += 1

    par_stack_header_index = cur_node.par_stack_header_index
    par_stack_footer_index = cur_node.par_stack_footer_index
    n_samples = par_stack_footer_index - par_stack_header_index + 1
    cur_par_stack_number = cur_node.par_stack_number

    // print_node(cur_node, config, 0)

    // convert to leaf
    if n_samples <= min_node_samples {
        cur_node.leaf_flag = true
        targets := make(DataArray, cur_node.n_target)
        
        cur_par_stack = par_stack_arr[cur_par_stack_number]

        i := 0
        for par_stack_index := par_stack_header_index; par_stack_index <= par_stack_footer_index; par_stack_index++  {
            sample_index = cur_par_stack[par_stack_index]
            targets[i] = target_data_arr[sample_index]
            i++
        }
        cur_node.targets = targets
        cur_node.targets_avg = cur_node.target_sum/float32(cur_node.n_target)
        
        // pretty.Println("leaf",targets)
        continue
    }

    // prepare 
    k_attrs = random_pick_k_attrs(cur_node, config) // OPTM

cur_lr_variance_val = float32(math.Inf(1))
split_mode = SPLIT_MODE_NORMAL

for _,attr_index := range k_attrs {
    attr_row_index = attr_index*total_samples

    // get cur_par_stack and next_par_stack
    cur_par_stack = par_stack_arr[cur_par_stack_number]
    if cur_par_stack_number == 0 {
        next_par_stack_number = 1
    } else {
        next_par_stack_number = 0
    }
    next_par_stack = par_stack_arr[next_par_stack_number]

    // start split for each attr of cur_node, OPTM
    ind_middle = random_integer(par_stack_header_index, par_stack_footer_index-1)
    ind_a = random_integer(par_stack_header_index, ind_middle)
    ind_b = random_integer(ind_middle+1, par_stack_footer_index)


    spliter = (sample_data_mat[attr_row_index+cur_par_stack[ind_a]]+sample_data_mat[attr_row_index+cur_par_stack[ind_b]])*0.5

    left_index = par_stack_header_index
    right_index = par_stack_footer_index

    left_target_sum = 0
    left_target_sq_sum = 0
    left_n_target = 0
    right_target_sum = 0
    right_target_sq_sum = 0
    right_n_target = 0

// split node
if split_mode == SPLIT_MODE_NORMAL {

    for par_stack_index := par_stack_header_index; par_stack_index <= par_stack_footer_index; par_stack_index++  {
        sample_index = cur_par_stack[par_stack_index]

        cur_target = target_data_arr[sample_index]
        cur_target_sq = target_sq_data_arr[sample_index]

        // to left or right
        if sample_data_mat[attr_row_index+sample_index] < spliter { // to left
            left_target_sum += cur_target
            left_target_sq_sum += cur_target_sq
            left_n_target += 1

            next_par_stack[left_index] = sample_index
            left_index += 1

        } else { // to right
            right_target_sum += cur_target
            right_target_sq_sum += cur_target_sq
            right_n_target += 1

            next_par_stack[right_index] = sample_index
            right_index -= 1
        }
    }

    // change mode
    // split_mode = SPLIT_MODE_EXCHANGE

} else if split_mode == SPLIT_MODE_EXCHANGE {

    

}

    left_index -= 1
    right_index += 1

    left_variance_val = left_target_sq_sum - left_target_sum*left_target_sum/float32(left_n_target) 
    right_variance_val = right_target_sq_sum - right_target_sum*right_target_sum/float32(right_n_target)
    lr_variance_val = left_variance_val + right_variance_val

    // pretty.Println("n_target",spliter,left_n_target,right_n_target)
    if lr_variance_val < cur_lr_variance_val { 
        // smaller variance_val, it is good, save split_result
        cur_lr_variance_val = lr_variance_val
        cur_attr_index = attr_index
        cur_spliter = spliter
        
        cur_left_variance_val = left_variance_val
        cur_left_target_sum = left_target_sum
        cur_left_target_sq_sum = left_target_sq_sum
        cur_left_n_target = left_n_target
        cur_left_par_stack_header_index = par_stack_header_index
        cur_left_par_stack_footer_index = left_index 
        
        cur_right_variance_val = right_variance_val
        cur_right_target_sum = right_target_sum
        cur_right_target_sq_sum = right_target_sq_sum
        cur_right_n_target = right_n_target
        cur_right_par_stack_header_index = right_index
        cur_right_par_stack_footer_index = par_stack_footer_index

        cur_par_stack_number = next_par_stack_number
        
        // pretty.Println(cur_left_par_stack_header_index, cur_left_par_stack_footer_index,cur_right_par_stack_header_index,cur_right_par_stack_footer_index)
    }


    } // end k_attrs loop


    // update cur_node
    cur_node.attr_index = cur_attr_index
    cur_node.spliter = cur_spliter

    // append left and right node
    left_node = &ExtraTreeNode{
        par_stack_number: cur_par_stack_number,
        par_stack_header_index: cur_left_par_stack_header_index,
        par_stack_footer_index: cur_left_par_stack_footer_index,

        target_sum: cur_left_target_sum,
        target_sq_sum: cur_left_target_sq_sum,
        n_target: cur_left_n_target,
        variance_val: cur_left_variance_val,

        leaf_flag: false,
    }
    cur_node.left_child = left_node
    // print_node(left_node, config, 1)

    right_node = &ExtraTreeNode{
        par_stack_number: cur_par_stack_number,
        par_stack_header_index: cur_right_par_stack_header_index,
        par_stack_footer_index: cur_right_par_stack_footer_index,

        target_sum: cur_right_target_sum,
        target_sq_sum: cur_right_target_sq_sum,
        n_target: cur_right_n_target,
        variance_val: cur_right_variance_val,

        leaf_flag: false,
    }
    cur_node.right_child = right_node
    // print_node(right_node, config, 2)

    node_queue[end_queue_index] = left_node
    end_queue_index += 1
    node_queue[end_queue_index] = right_node
    end_queue_index += 1

// debug control
// if cur_queue_index == 3 {
//     break
// }

} // end node_queue loop

    // pretty.Println(end_queue_index)
    return root_node
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
        fmt.Printf("%snode(depth:%d): \n",big_indent,depth)
        depth += 1
        indent += 2
        print_extratree(node.left_child, depth, indent)
        print_extratree(node.right_child, depth, indent)
    }
}

func print_node(cur_node *ExtraTreeNode, config *ExtraTreeConfig, child_flag int) {
    cur_par_stack_number := cur_node.par_stack_number
    cur_par_stack := config.par_stack_arr[cur_par_stack_number]
    par_stack_header_index := cur_node.par_stack_header_index
    par_stack_footer_index := cur_node.par_stack_footer_index
    
    if child_flag == 0 {
        pretty.Println("parent",cur_par_stack[par_stack_header_index:par_stack_footer_index+1],"cur_par_stack_number",cur_par_stack_number)
        // pretty.Println(cur_node.par_stack_header_index, cur_node.par_stack_footer_index)

    } else if child_flag == 1 {
        pretty.Println("left_child",cur_par_stack[par_stack_header_index:par_stack_footer_index+1],"cur_par_stack_number",cur_par_stack_number)

    } else if child_flag == 2 {
        pretty.Println("right_child",cur_par_stack[par_stack_header_index:par_stack_footer_index+1],"cur_par_stack_number",cur_par_stack_number)
        pretty.Println("par_stack 0",config.par_stack_arr[0])
        pretty.Println("par_stack 1", config.par_stack_arr[1])
        // pretty.Println(cur_node.par_stack_header_index, cur_node.par_stack_footer_index)
        pretty.Println("")
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
    target_sq_data_arr := make(DataArray, len(Y))
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
    config.par_stack_arr = make([]IndexArray, 2)
    for i,_ := range config.par_stack_arr {
        temp_arr := make(IndexArray, n_samples)
        for j,_ := range temp_arr {
            temp_arr[j] = j
        }
        config.par_stack_arr[i] = temp_arr
    }
    ext.config = config

    // start build extratree from root_node
    root_node := gen_root_node(config)
    ext.root_node = root_node

    build_extratree_loop(root_node, config)
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

    // cfg := profile.CPUProfile

    X, Y := make_toy_dataset(100000, 30)
    
    start := time.Now()

    // p := profile.Start(cfg)

    // pretty.Println(X, Y)
    ext := NewExtraTreesRegressor(1, 1, 1)
    ext.fit(X, Y)

    // p.Stop()

    elapsed := time.Since(start)
    pretty.Printf("elapsed time: %.16f\n", elapsed.Seconds())


    // print_extratree(ext.root_node, 0, 0)

}







