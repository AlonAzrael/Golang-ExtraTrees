package main

import (
    "math/rand"
    "fmt"
    "time"
    "log"
)

func random_integer(min int, max int) int {
    return rand.Intn(max+1 - min) + min
}

func test_random_integer() {
    rand.Seed(time.Now().Unix())
    for i := 0; i < 10; i++ {
        x:=random_integer(10,11)
        fmt.Println(x)
    }
}

func test_array() {
    method_number := 1
    size := 10000000
    counter := size
    if method_number == 0 {
        var arr []float32
        for i := 0; i < counter; i++ {
            arr = append(arr, float32(i) )
        }
    } else if method_number == 1 {
        arr := make([]float32, size)
        for i := 0; i < counter; i++ {
            arr[i] += float32(i)
        }
        arr = arr[0:counter]
        // fmt.Println(arr)
    } else if method_number == 2 {
        arr := make([]float32, size)
        for i,x := range arr {
            arr[i] = float32(i) + x
        }
        arr = arr[0:counter]
    }
}

type DataArray []float32
type DataArrayPointers []DataArray
func gen_rows() DataArrayPointers{
    method_number := 0
    n_rows := 1000000
    n_columns := 15

    var rows DataArrayPointers
    if method_number == 0 {
        rows = make(DataArrayPointers, n_rows)
        var temp_row DataArray

        for i := 0; i < n_rows; i++ {
            rows[i] = make(DataArray, n_columns)
            temp_row = rows[i]
            for j := 0; j < n_columns; j++ {
                temp_row[j] = float32(j)
            }
        }
    } else if method_number == 1{
        
    }
    
    return rows
}

type Mat struct {
    n_rows uint
    n_columns uint
}
func (m *Mat) set() {
    
}
func (m *Mat) get() {
    
}


func gen_rows1d() DataArray{
    // n_rows := 1000000
    // n_columns := 15

    n_rows := 15
    n_columns := 1000000

    rows_1d := make(DataArray, n_rows*n_columns)
    for i := 0; i < n_rows; i++ {
        prefix := i*n_columns
        for j := 0; j < n_columns; j++ {
            rows_1d[prefix+j] = float32(j)
        }
    }
    return rows_1d
}
func test_small_large_loop(rows DataArrayPointers) {
    method_number := 0
    n_small := 14
    x := 0
    if method_number == 0 {
        for small_index := 0; small_index < n_small; small_index++ {
            for i := 0; i < len(rows); i++ {
                // rows[i][small_index] += 1
                x += 1
            }
        }
    } else if method_number == 1 {
        // var temp_row DataArray
        for row_index := 0; row_index < len(rows); row_index++ {
            // temp_row = rows[row_index]
            for small_index := 0; small_index < n_small; small_index++ {
                // temp_row[small_index] += 1
                // x += 1
            }
        }
    } else if method_number == 2 {
        for i := 0; i < n_small*len(rows); i++ {
            x += 1
        }
    }
    fmt.Println(x)
}

func test_small_large_loop_by_rows1d(rows1d DataArray) {
    method_number := 1
    // n_small := 8
    
    // n_rows := 1000000
    // n_columns := 15

    n_rows := 15
    n_columns := 1000000

    x := 0
    if method_number == 0 {
        for small_index := 0; small_index < n_columns; small_index++ {
            for i := 0; i < n_rows; i++ {
                rows1d[i*n_columns+small_index] += 1
            }
        }
    } else if method_number == 1 {
        // var temp_row DataArray
        for row_index := 0; row_index < n_rows; row_index++ {
            // temp_row = rows[row_index]
            for small_index := 0; small_index < n_columns; small_index++ {
                rows1d[row_index*n_columns+small_index] += 1
            }
        }
    } else if method_number == 2 {
        for i := 0; i < n_rows*n_columns; i++ {
            rows1d[i] += 1
        }
    }
    fmt.Println(x)
}

func test_make_array() {
    arr_size := 100
    temp_arr := make([][]int, 30000)
    for i := 0; i < len(temp_arr); i++ {
        temp_arr[i] = make([]int, arr_size)
        _ = temp_arr[0]
    }

}

func main() {

    // rows := gen_rows()
    // rows1d := gen_rows1d()
    // rows := gen_mat()

    start := time.Now()

    // test_small_large_loop(rows)
    // test_small_large_loop_by_rows1d(rows1d)
    // test_array()
    test_make_array()

    elapsed := time.Since(start)
    log.Printf("elapsed time: %.16f", elapsed.Seconds())
    
}

