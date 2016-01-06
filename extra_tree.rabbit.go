package main

import (
    "math/rand"
    "math"
    "time"
    "runtime"
    "sync"

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
    
    rabbit_par_stack IndexArray
    all_attr_index IndexArray

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

func calc_mse(tY DataArray, pY DataArray) float32{
    var mse_val float64
    mse_val = 0
    
    for i,y := range tY {
        mse_val += math.Pow(float64(y - pY[i]), 2)
    }
    mse_val /= float64(len(tY))

    return float32(mse_val)
}

func calc_variance_val(va DataArray) float32{
    // print "calc_variance_val: ",va
    return va[1] - va[0]*va[0]/va[2]
}

var PESUEDO_RANDOM_NUMBER_LIST []float32
var PR_INDEX int
func init_random_seed_pesuedo(n_seed int) {
    //init 
    rand.Seed(time.Now().Unix())

    PESUEDO_RANDOM_NUMBER_LIST = make([]float32, n_seed)
    PR_INDEX = 0

    for i := 0; i < n_seed; i++ {
        PESUEDO_RANDOM_NUMBER_LIST[i] = rand.Float32()
    }
}
func random_integer_pesuedo(min int, max int) int {
    pf := PESUEDO_RANDOM_NUMBER_LIST[PR_INDEX]
    PR_INDEX += 1
    if PR_INDEX == len(PESUEDO_RANDOM_NUMBER_LIST) {
        PR_INDEX = 0
    }
    var a ,b float32
    a = float32(min)
    b = float32(max)
    return int((b+0.99 - a)*pf+a)
}

func random_pick_k_attrs(node *ExtraTreeNode, config *ExtraTreeConfig) IndexArray{
    K := config.K
    n_attrs := config.n_attrs
    if K == n_attrs {
        // arr := make(IndexArray, n_attrs)
        // for i := 0; i < n_attrs; i++ {
        //     arr[i] = i
        // }
        // return arr
        return config.all_attr_index
    } else {
        arr := rand.Perm(config.n_attrs)
        return arr[0:config.K]
    }
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


func convert_node_leaf(cur_node *ExtraTreeNode, config *ExtraTreeConfig) {
    cur_node.leaf_flag = true
    targets := make(DataArray, cur_node.n_target)
    par_stack_header_index := cur_node.par_stack_header_index
    par_stack_footer_index := cur_node.par_stack_footer_index
    rabbit_par_stack := config.rabbit_par_stack
    target_data_arr := config.target_data_arr

    i := 0
    sample_index := 0
    for par_stack_index := par_stack_header_index; par_stack_index <= par_stack_footer_index; par_stack_index++  {
        sample_index = rabbit_par_stack[par_stack_index]
        targets[i] = target_data_arr[sample_index]
        i++
    }
    cur_node.targets = targets
    cur_node.targets_avg = cur_node.target_sum/float32(cur_node.n_target)
    // pretty.Println("leaf",targets)
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
    rabbit_par_stack := config.rabbit_par_stack
    sample_data_mat := config.sample_data_mat
    target_data_arr := config.target_data_arr
    target_sq_data_arr := config.target_sq_data_arr

    SPLIT_MODE_NORMAL := 0
    // SPLIT_MODE_EXCHANGE := 1

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

var spliter float32
split_mode := SPLIT_MODE_NORMAL
var left_index int
var right_index int
var ind_middle int
var ind_a int
var ind_b int

var swap_index int

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

    // print_node(cur_node, config, 0)

    // convert to leaf
    if n_samples <= min_node_samples {
        convert_node_leaf(cur_node, config)
        continue
    }

    // prepare 
    k_attrs = random_pick_k_attrs(cur_node, config) // OPTM

cur_lr_variance_val = float32(math.Inf(1))
split_mode = SPLIT_MODE_NORMAL

for _,attr_index := range k_attrs {
    attr_row_index = attr_index*total_samples

    // start split for each attr of cur_node, OPTM
    // ind_middle = random_integer_pesuedo(par_stack_header_index, par_stack_footer_index-1)
    // ind_a = random_integer_pesuedo(par_stack_header_index, ind_middle)
    // ind_b = random_integer_pesuedo(ind_middle+1, par_stack_footer_index)

    ind_middle = (par_stack_header_index+par_stack_footer_index)/2
    ind_a = par_stack_header_index
    ind_b = par_stack_footer_index

    // defer func() { //catch or finally
    //     if err := recover(); err != nil { //catch
    //         pretty.Println(par_stack_header_index, par_stack_footer_index)
    //         pretty.Println(ind_middle, ind_a, ind_b)
    //     }
    // }()

    middle := sample_data_mat[attr_row_index+rabbit_par_stack[ind_middle]]
    a := sample_data_mat[attr_row_index+rabbit_par_stack[ind_a]]
    b := sample_data_mat[attr_row_index+rabbit_par_stack[ind_b]]

    if middle == a && middle == b && a == b { // assuming this attr has some problem
        continue
    }

    spliter = (a+b)*0.5

    left_index = par_stack_header_index
    right_index = par_stack_footer_index
    swap_index = left_index

// split node
if split_mode == SPLIT_MODE_NORMAL {

    left_target_sum = 0
    left_target_sq_sum = 0
    right_target_sum = 0
    right_target_sq_sum = 0

    // pretty.Println(spliter)

    for left_index <= right_index {

        sample_index = rabbit_par_stack[left_index]
        cur_target = target_data_arr[sample_index]
        cur_target_sq = target_sq_data_arr[sample_index]

        if sample_data_mat[attr_row_index+sample_index] < spliter {
            if swap_index != left_index {
                rabbit_par_stack[swap_index], rabbit_par_stack[left_index] = rabbit_par_stack[left_index], rabbit_par_stack[swap_index]
            }
            swap_index += 1
            left_target_sum += cur_target
            left_target_sq_sum += cur_target_sq
        } else {
            right_target_sum += cur_target
            right_target_sq_sum += cur_target_sq
        }
        left_index += 1
    }

    // pretty.Println(spliter,rabbit_par_stack[par_stack_header_index:swap_index],rabbit_par_stack[swap_index:par_stack_footer_index+1])

    // change mode
    // split_mode = SPLIT_MODE_EXCHANGE

}
    left_n_target = swap_index - par_stack_header_index
    right_n_target = par_stack_footer_index - swap_index + 1
    left_index = swap_index - 1
    right_index = swap_index

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
        
        // pretty.Println(cur_left_par_stack_header_index, cur_left_par_stack_footer_index,cur_right_par_stack_header_index,cur_right_par_stack_footer_index)
    }


    } // end k_attrs loop

    // if all attr cant be split, then this node will be a leaf
    if cur_lr_variance_val == float32(math.Inf(1)) {
        convert_node_leaf(cur_node, config)
        continue
    }

    // re-split rabbit_par_stack
    left_index = par_stack_header_index 
    right_index = par_stack_footer_index
    swap_index = left_index
    attr_row_index = cur_attr_index*total_samples
    for left_index <= right_index {
        sample_index = rabbit_par_stack[left_index]

        if sample_data_mat[attr_row_index+sample_index] < cur_spliter {
            if swap_index != left_index {
                rabbit_par_stack[swap_index], rabbit_par_stack[left_index] = rabbit_par_stack[left_index], rabbit_par_stack[swap_index]
            }
            swap_index += 1
        }
        left_index += 1
    }


    // update cur_node
    cur_node.attr_index = cur_attr_index
    cur_node.spliter = cur_spliter

    // append left and right node
    left_node = &ExtraTreeNode{
        // par_stack_number: cur_par_stack_number,
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
        // par_stack_number: cur_par_stack_number,
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
// if cur_queue_index == 1 {
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
        fmt.Println(big_indent,"leaf: ",node.targets, node.targets_avg)
    } else {
        fmt.Printf("%snode(depth:%d): \n",big_indent,depth)
        depth += 1
        indent += 2
        print_extratree(node.left_child, depth, indent)
        print_extratree(node.right_child, depth, indent)
    }
}

func print_node(cur_node *ExtraTreeNode, config *ExtraTreeConfig, child_flag int) {
    cur_par_stack := config.rabbit_par_stack
    par_stack_header_index := cur_node.par_stack_header_index
    par_stack_footer_index := cur_node.par_stack_footer_index
    
    if child_flag == 0 {
        pretty.Println("parent",cur_par_stack[par_stack_header_index:par_stack_footer_index+1])
        // pretty.Println(cur_node.par_stack_header_index, cur_node.par_stack_footer_index)

    } else if child_flag == 1 {
        pretty.Println("left_child",cur_par_stack[par_stack_header_index:par_stack_footer_index+1])

    } else if child_flag == 2 {
        pretty.Println("right_child",cur_par_stack[par_stack_header_index:par_stack_footer_index+1])
        pretty.Println("rabbit_par_stack", cur_par_stack)
        // pretty.Println(cur_node.par_stack_header_index, cur_node.par_stack_footer_index)
        pretty.Println("")
    }
    
}


/*
API
=================================================
*/

// single tree
type ExtraTreeRegressor struct {
    K int
    min_node_samples int
    root_node *ExtraTreeNode
    config *ExtraTreeConfig
}
func NewExtraTreeRegressor(K int, min_node_samples int) *ExtraTreeRegressor{
    // init_random_seed_pesuedo(1000)

    p := &ExtraTreeRegressor{
        K: K,
        min_node_samples: min_node_samples,
    }
    return p
}
func (ext *ExtraTreeRegressor) fit(X []DataArray, Y DataArray) *ExtraTreeRegressor{
    n_samples := len(X)
    n_attrs := len(X[0])
    K := n_attrs // just use all attrs, change it to K := ext.K, for sub feature space
    min_node_samples := ext.min_node_samples

    if n_samples != len(Y) {
        panic("Assertion: len(X) == len(Y)")
    }

    
    XT := make(DataArray, K*n_samples)
    var target_data_arr DataArray
    target_sq_data_arr := make(DataArray, len(Y))
    shuffle_flag := true

    if shuffle_flag {
        // shuffle sample_index 
        // source := rand.NewSource(time.Now().UnixNano())
        // rander := rand.New(source)
        shuffle_sample_index_arr := rand.Perm(n_samples)
        target_data_arr = make(DataArray, len(Y))
        
        for row_index,sample_index := range shuffle_sample_index_arr {
            row := X[sample_index] // random row
            for attr_index, column := range row {
                XT[attr_index*n_samples+row_index] = column
            }
            y := Y[sample_index]
            target_data_arr[row_index] = y
            target_sq_data_arr[row_index] = y*y
        }
    } else {
        // transpose X
        target_data_arr = Y
        samples := make(IndexArray, n_samples) // normal 
        for sample_index,row := range X {
            samples[sample_index] = sample_index
            for attr_index,column := range row {
                XT[attr_index*n_samples+sample_index] = column
            }
        }
        // gen target_sq_data_arr
        for target_index,target := range Y {
            target_sq_data_arr[target_index] = target*target
        }
    }
    // pretty.Println(XT)
    // pretty.Println(target_data_arr)

    // gen config
    config := NewExtraTreeConfig(n_attrs, K, min_node_samples)
    config.sample_data_mat = XT
    config.target_data_arr = target_data_arr
    config.target_sq_data_arr = target_sq_data_arr
    config.total_samples = n_samples
    config.X = X
    temp_arr := make(IndexArray, n_samples)
    for i := 0; i < n_samples; i++ {
        temp_arr[i] = i
    }
    config.rabbit_par_stack = temp_arr
    temp_arr = make(IndexArray, n_attrs)
    for i := 0; i < n_attrs; i++ {
        temp_arr[i] = i
    }
    config.all_attr_index = temp_arr
    ext.config = config

    // debug
    // return ext

    // start build extratree from root_node
    root_node := gen_root_node(config)
    ext.root_node = root_node

    build_extratree_loop(root_node, config)
    return ext
}
func (ext *ExtraTreeRegressor) predict(X []DataArray) (pY DataArray) {
    pY = make(DataArray, len(X))
    for row_index,row := range X {
        pY[row_index] = walk_extratree(ext.root_node, row)
    }

    return pY
}


// tree ensemble
type ExtraTreeRegressorEnsemble struct {
    K int
    min_node_samples int
    n_estimators int
    n_jobs int
    tree_arr []*ExtraTreeRegressor
}
func NewExtraTreeRegressorEnsemble(K int, min_node_samples int, n_estimators int) *ExtraTreeRegressorEnsemble{
    init_random_seed_pesuedo(1000)

    p := &ExtraTreeRegressorEnsemble{
        K: K,
        min_node_samples: min_node_samples,
        n_estimators: n_estimators,
    }
    tree_arr := make([]*ExtraTreeRegressor, n_estimators)
    for i := 0; i < n_estimators; i++ {
        tree_arr[i] = NewExtraTreeRegressor(K, min_node_samples)
    }
    p.tree_arr = tree_arr
    return p
}
func (exte *ExtraTreeRegressorEnsemble) fit(X []DataArray, Y DataArray) *ExtraTreeRegressorEnsemble{
    n_estimators := exte.n_estimators
    tree_arr := exte.tree_arr

    if n_estimators <= 4 {
        for i := 0; i < n_estimators; i++ {
            tree_arr[i].fit(X, Y)
        }
    } else {
        n_jobs := 4
        runtime.GOMAXPROCS(runtime.NumCPU())
        n_task_each_job := n_estimators / n_jobs 
        n_rest_task := n_estimators % n_jobs

        // tree_index_chain := make(chan *ExtraTreeRegressor)
        var wg sync.WaitGroup
        wg.Add(n_jobs)

        start := 0
        end := n_task_each_job + n_rest_task
        for job_i := 0; job_i < n_jobs; job_i++ {
            go func (job_i int, start int, end int, tree_arr []*ExtraTreeRegressor) {
                // defer wg.Done()

                for tree_index := start; tree_index < end; tree_index++ {
                    tree_arr[tree_index].fit(X, Y)
                    fmt.Println("the job",job_i,"finish",tree_index)
                }
                // fmt.Println(job_i," is done")
                wg.Done()
            }(job_i, start, end, tree_arr)

            start = end
            end += n_task_each_job
        }

        fmt.Println("Waiting To Finish")
        wg.Wait()
    }
    return exte
}
func (exte *ExtraTreeRegressorEnsemble) predict(X []DataArray) (pY DataArray){

    n_estimators := exte.n_estimators
    var targets_avg float32
    pY = make(DataArray, len(X))

    for row_index,row := range X {
        targets_avg = 0
        for tree_index := 0; tree_index < n_estimators; tree_index++ {
            cur_tree := exte.tree_arr[tree_index]
            targets_avg += walk_extratree(cur_tree.root_node, row)
        }
        pY[row_index] = targets_avg/float32(n_estimators)
    }

    return pY
}







/*
generate dataset
=================================================
*/
func arange(start int, stop int) IndexArray {
    step := 1
    // N := int(math.Ceil((stop - start) / step));
    N := stop - start
    rnge := make(IndexArray, N)
    i := 0
    for x := start; x < stop; x += step {
        rnge[i] = x;
        i += 1
    }
    return rnge
}

func make_friedman1(n_samples int, n_attrs int) (X []DataArray, Y DataArray){
    
    n_attrs = 10
    X = make([]DataArray, n_samples)
    Y = make(DataArray, n_samples)
    attr_index_arr := arange(0, n_attrs)
    for i := 0; i < n_samples; i++ {
        row := make(DataArray, n_attrs)
        for j,_ := range row {
            row[j] = rand.Float32()
        }
        X[i] = row
        // attr_index_arr := rand.Perm(n_attrs)[0:5]
        x0 := row[attr_index_arr[0]]
        x1 := row[attr_index_arr[1]]
        x2 := row[attr_index_arr[2]]
        x3 := row[attr_index_arr[3]]
        x4 := row[attr_index_arr[4]]
        Pi := float32(math.Pi)
        E := float32(math.E)

        Y[i] = float32(10*math.Sin(float64(Pi*x0*x1))) + 20*(x2-0.5)*(x2-0.5) + 10*x3 + 5*x4 + E
    }

    return X, Y
}

func make_toy_dataset(n_samples int, n_attrs int) (X []DataArray, Y DataArray){

    X = make([]DataArray, n_samples)
    Y = make(DataArray, n_samples)
    var val float32
    for i := 0; i < n_samples; i++ {
        val = float32(i+0)
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

func test_ext() {
    // cfg := profile.CPUProfile

    // X, Y := make_toy_dataset(100000, 30)
    X, Y := make_friedman1(100, 10)
    // tX, tY := make_friedman1(100, 10)
    
    start := time.Now()

    // p := profile.Start(cfg)

    // pretty.Println(X, Y)
    ext := NewExtraTreeRegressor(1, 1)
    ext.fit(X, Y)

    // p.Stop()

    elapsed := time.Since(start)
    pretty.Printf("elapsed time: %.16f\n", elapsed.Seconds())

    // pY := ext.predict(tX)
    // mse_val := calc_mse(tY, pY)
    // pretty.Println(mse_val)

    print_extratree(ext.root_node, 0, 0)
}

func test_exte() {
    
    // cfg := profile.CPUProfile

    // X, Y := make_toy_dataset(100000, 30)
    X, Y := make_friedman1(100000, 10)
    tX, tY := make_friedman1(100, 10)
    
    start := time.Now()

    // p := profile.Start(cfg)

    // pretty.Println(X, Y)
    n_estimators := 100
    exte := NewExtraTreeRegressorEnsemble(1, 1, n_estimators)
    exte.fit(X, Y)

    // p.Stop()

    elapsed := time.Since(start)
    pretty.Printf("elapsed time: %.16f\n", elapsed.Seconds())

    pY := exte.predict(tX)
    mse_val := calc_mse(tY, pY)
    pretty.Println(mse_val)

    // tree_arr := exte.tree_arr
    for i := 0; i < n_estimators; i++ {
        // print_extratree(tree_arr[i].root_node, 0, 0)
        // pretty.Println("")
    }
    

}

func main() {

    test_exte()

}







