# MiniTorch Module 3

## 3.2 Diagnostic Output
Parallel Accelerator Diagnostics

Below is the raw diagnostic output for parallel optimizations of the tensor_map, tensor_zip, tensor_reduce, and _tensor_matrix_multiply functions.

```
(.venv) salik@dhcp-vl2051-207 mod3-saliktehami % python project/parallel_check.py

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (164) 
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        in_storage: Storage,                                             | 
        in_shape: Shape,                                                 | 
        in_strides: Strides,                                             | 
    ) -> None:                                                           | 
        for i in prange(len(out)):---------------------------------------| #0
            # Create thread-local indices inside the parallel loop       | 
            out_index = np.empty(MAX_DIMS, np.int32)                     | 
            in_index = np.empty(MAX_DIMS, np.int32)                      | 
                                                                         | 
            to_index(i, out_shape, out_index)                            | 
            broadcast_index(out_index, out_shape, in_shape, in_index)    | 
            o = index_to_position(out_index, out_strides)                | 
            j = index_to_position(in_index, in_strides)                  | 
            out[o] = fn(in_storage[j])                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (174) 
is hoisted out of the parallel loop labelled #0 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (175) 
is hoisted out of the parallel loop labelled #0 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (209)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (209) 
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        b_storage: Storage,                                            | 
        b_shape: Shape,                                                | 
        b_strides: Strides,                                            | 
    ) -> None:                                                         | 
        for i in prange(len(out)):-------------------------------------| #1
            # Thread-local indices                                     | 
            out_index = np.empty(MAX_DIMS, np.int32)                   | 
            a_index = np.empty(MAX_DIMS, np.int32)                     | 
            b_index = np.empty(MAX_DIMS, np.int32)                     | 
                                                                       | 
            to_index(i, out_shape, out_index)                          | 
            o = index_to_position(out_index, out_strides)              | 
            broadcast_index(out_index, out_shape, a_shape, a_index)    | 
            j = index_to_position(a_index, a_strides)                  | 
            broadcast_index(out_index, out_shape, b_shape, b_index)    | 
            k = index_to_position(b_index, b_strides)                  | 
            out[o] = fn(a_storage[j], b_storage[k])                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (222) 
is hoisted out of the parallel loop labelled #1 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (223) 
is hoisted out of the parallel loop labelled #1 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (224) 
is hoisted out of the parallel loop labelled #1 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (258)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (258) 
-----------------------------------------------------------------|loop #ID
    def _reduce(                                                 | 
        out: Storage,                                            | 
        out_shape: Shape,                                        | 
        out_strides: Strides,                                    | 
        a_storage: Storage,                                      | 
        a_shape: Shape,                                          | 
        a_strides: Strides,                                      | 
        reduce_dim: int,                                         | 
    ) -> None:                                                   | 
        out_index = np.zeros(MAX_DIMS, np.int32)-----------------| #2
        reduce_size = a_shape[reduce_dim]                        | 
                                                                 | 
        # Parallelize the outer loop over output elements        | 
        for i in prange(len(out)):-------------------------------| #3
            # Thread-local indices                               | 
            out_index = np.empty(MAX_DIMS, np.int32)             | 
            local_index = np.empty(MAX_DIMS, np.int32)           | 
                                                                 | 
            to_index(i, out_shape, out_index)                    | 
            o = index_to_position(out_index, out_strides)        | 
                                                                 | 
            # Copy indices to local                              | 
            for j in range(len(out_shape)):                      | 
                local_index[j] = out_index[j]                    | 
                                                                 | 
            # Sequential reduction                               | 
            for s in range(reduce_size):                         | 
                local_index[reduce_dim] = s                      | 
                j = index_to_position(local_index, a_strides)    | 
                out[o] = fn(out[o], a_storage[j])                | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (273) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (274) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: local_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (292)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (292) 
----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                            | 
    out: Storage,                                                                       | 
    out_shape: Shape,                                                                   | 
    out_strides: Strides,                                                               | 
    a_storage: Storage,                                                                 | 
    a_shape: Shape,                                                                     | 
    a_strides: Strides,                                                                 | 
    b_storage: Storage,                                                                 | 
    b_shape: Shape,                                                                     | 
    b_strides: Strides,                                                                 | 
) -> None:                                                                              | 
    """NUMBA tensor matrix multiply function."""                                        | 
    # Basic compatibility check                                                         | 
    assert a_shape[-1] == b_shape[-2]                                                   | 
                                                                                        | 
    # Get batch strides (0 if not batched)                                              | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                              | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                              | 
                                                                                        | 
    # Get key dimensions                                                                | 
    batch_size = max(out_shape[0], 1)  # Number of batches                              | 
    M = out_shape[1]  # Rows in output                                                  | 
    N = out_shape[2]  # Cols in output                                                  | 
    K = a_shape[-1]  # Shared dimension (cols in A, rows in B)                          | 
                                                                                        | 
    # Parallel over both batch and M dimensions for better utilization                  | 
    for batch in prange(batch_size):----------------------------------------------------| #5
        # Calculate batch offsets once per batch                                        | 
        a_batch_offset = batch * a_batch_stride                                         | 
        b_batch_offset = batch * b_batch_stride                                         | 
        out_batch_offset = batch * out_strides[0]                                       | 
                                                                                        | 
        for i in prange(M):-------------------------------------------------------------| #4
            # Calculate row offset for A once per i                                     | 
            a_row_offset = a_batch_offset + i * a_strides[1]                            | 
                                                                                        | 
            for j in range(N):                                                          | 
                # Calculate final output position                                       | 
                out_idx = out_batch_offset + i * out_strides[1] + j * out_strides[2]    | 
                                                                                        | 
                # Initialize accumulator                                                | 
                acc = 0.0                                                               | 
                                                                                        | 
                # Inner product loop - single multiply per iteration                    | 
                for k in range(K):                                                      | 
                    # Calculate positions with minimal operations                       | 
                    a_pos = a_row_offset + k * a_strides[2]                             | 
                    b_pos = b_batch_offset + k * b_strides[1] + j * b_strides[2]        | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                          | 
                                                                                        | 
                # Single write to global memory                                         | 
                out[out_idx] = acc                                                      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #5, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--5 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--4 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--5 (parallel)
   +--4 (serial)


 
Parallel region 0 (loop #5) had 0 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#5).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

## Passing 3.3 and 3.4 Tests
### 3.3 Test Results
```
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0 -- /usr/bin/python3.12
cachedir: .pytest_cache
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/content/mod3-saliktehami/.hypothesis/examples')
rootdir: /content/mod3-saliktehami
configfile: pyproject.toml
plugins: env-1.1.4, hypothesis-6.54.0
collected 117 items / 60 deselected / 57 selected                                                  

tests/test_tensor_general.py::test_create[cuda] PASSED                                       [  1%]
tests/test_tensor_general.py::test_one_args[cuda-fn0] PASSED                                 [  3%]
tests/test_tensor_general.py::test_one_args[cuda-fn1] PASSED                                 [  5%]
tests/test_tensor_general.py::test_one_args[cuda-fn2] PASSED                                 [  7%]
tests/test_tensor_general.py::test_one_args[cuda-fn3] PASSED                                 [  8%]
tests/test_tensor_general.py::test_one_args[cuda-fn4] PASSED                                 [ 10%]
tests/test_tensor_general.py::test_one_args[cuda-fn5] PASSED                                 [ 12%]
tests/test_tensor_general.py::test_one_args[cuda-fn6] PASSED                                 [ 14%]
tests/test_tensor_general.py::test_one_args[cuda-fn7] PASSED                                 [ 15%]
tests/test_tensor_general.py::test_one_args[cuda-fn8] PASSED                                 [ 17%]
tests/test_tensor_general.py::test_one_args[cuda-fn9] PASSED                                 [ 19%]
tests/test_tensor_general.py::test_one_args[cuda-fn10] PASSED                                [ 21%]
tests/test_tensor_general.py::test_one_args[cuda-fn11] PASSED                                [ 22%]
tests/test_tensor_general.py::test_one_args[cuda-fn12] PASSED                                [ 24%]
tests/test_tensor_general.py::test_one_args[cuda-fn13] PASSED                                [ 26%]
tests/test_tensor_general.py::test_two_args[cuda-fn0] PASSED                                 [ 28%]
tests/test_tensor_general.py::test_two_args[cuda-fn1] PASSED                                 [ 29%]
tests/test_tensor_general.py::test_two_args[cuda-fn2] PASSED                                 [ 31%]
tests/test_tensor_general.py::test_two_args[cuda-fn3] PASSED                                 [ 33%]
tests/test_tensor_general.py::test_two_args[cuda-fn4] PASSED                                 [ 35%]
tests/test_tensor_general.py::test_two_args[cuda-fn5] PASSED                                 [ 36%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn0] PASSED                           [ 38%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn1] PASSED                           [ 40%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn2] PASSED                           [ 42%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn3] PASSED                           [ 43%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn4] PASSED                           [ 45%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn5] PASSED                           [ 47%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn6] PASSED                           [ 49%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn7] PASSED                           [ 50%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn8] PASSED                           [ 52%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn9] PASSED                           [ 54%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn10] PASSED                          [ 56%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn11] PASSED                          [ 57%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn12] PASSED                          [ 59%]
tests/test_tensor_general.py::test_one_derivative[cuda-fn13] PASSED                          [ 61%]
tests/test_tensor_general.py::test_two_grad[cuda-fn0] PASSED                                 [ 63%]
tests/test_tensor_general.py::test_two_grad[cuda-fn1] PASSED                                 [ 64%]
tests/test_tensor_general.py::test_two_grad[cuda-fn2] PASSED                                 [ 66%]
tests/test_tensor_general.py::test_two_grad[cuda-fn3] PASSED                                 [ 68%]
tests/test_tensor_general.py::test_two_grad[cuda-fn4] PASSED                                 [ 70%]
tests/test_tensor_general.py::test_two_grad[cuda-fn5] PASSED                                 [ 71%]
tests/test_tensor_general.py::test_reduce[cuda-fn0] PASSED                                   [ 73%]
tests/test_tensor_general.py::test_reduce[cuda-fn1] PASSED                                   [ 75%]
tests/test_tensor_general.py::test_reduce[cuda-fn2] PASSED                                   [ 77%]
tests/test_tensor_general.py::test_sum_practice PASSED                                       [ 78%]
tests/test_tensor_general.py::test_sum_practice2 PASSED                                      [ 80%]
tests/test_tensor_general.py::test_sum_practice3 PASSED                                      [ 82%]
tests/test_tensor_general.py::test_sum_practice4 PASSED                                      [ 84%]
tests/test_tensor_general.py::test_sum_practice5 PASSED                                      [ 85%]
tests/test_tensor_general.py::test_sum_practice_other_dims PASSED                            [ 87%]
tests/test_tensor_general.py::test_two_grad_broadcast[cuda-fn0] PASSED                       [ 89%]
tests/test_tensor_general.py::test_two_grad_broadcast[cuda-fn1] PASSED                       [ 91%]
tests/test_tensor_general.py::test_two_grad_broadcast[cuda-fn2] PASSED                       [ 92%]
tests/test_tensor_general.py::test_two_grad_broadcast[cuda-fn3] PASSED                       [ 94%]
tests/test_tensor_general.py::test_two_grad_broadcast[cuda-fn4] PASSED                       [ 96%]
tests/test_tensor_general.py::test_two_grad_broadcast[cuda-fn5] PASSED                       [ 98%]
tests/test_tensor_general.py::test_permute[cuda] PASSED                                      [100%]
```

### 3.4 Test Results
```
======================================= test session starts ========================================
platform linux -- Python 3.12.7, pytest-8.3.2, pluggy-1.5.0 -- /usr/bin/python3.12
cachedir: .pytest_cache
hypothesis profile 'default' -> database=DirectoryBasedExampleDatabase('/content/mod3-saliktehami/.hypothesis/examples')
rootdir: /content/mod3-saliktehami
configfile: pyproject.toml
plugins: env-1.1.4, hypothesis-6.54.0
collected 117 items / 110 deselected / 7 selected                                                  

tests/test_tensor_general.py::test_mul_practice1 PASSED                                      [ 14%]
tests/test_tensor_general.py::test_mul_practice2 PASSED                                      [ 28%]
tests/test_tensor_general.py::test_mul_practice3 PASSED                                      [ 42%]
tests/test_tensor_general.py::test_mul_practice4 PASSED                                      [ 57%]
tests/test_tensor_general.py::test_mul_practice5 PASSED                                      [ 71%]
tests/test_tensor_general.py::test_mul_practice6 PASSED                                      [ 85%]
tests/test_tensor_general.py::test_bmm[cuda] PASSED                                          [100%]
```

## 3.4 Parallelization and Performance

### **Matrix Multiplication: CPU vs. CUDA**
- **Small Matrices**: Naive CPU implementation is faster due to lower overhead.
- **Medium Matrices**: CUDA starts to outperform at sizes like `16x16` or `32x32`.
- **Large Matrices**: CUDA demonstrates significant speedup for matrices like `256x256`.

| Matrix Size  | CPU Time (s) | CUDA Time (s) | Speedup |
|--------------|--------------|---------------|---------|
| 4x4          | Fast         | Slow          | -       |
| 16x16        | Moderate     | Fast          | 1.5x    |
| 256x256      | 242.58       | 0.86          | ~282x   |

---

## 3.5 Results

### **Simple Dataset**

#### CPU Results
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

| Epoch | Loss         | Accuracy | Time per Epoch (s) |
|-------|--------------|----------|---------------------|
| 0     | 6.37         | 33       | 23.80              |
| 10    | 2.64         | 48       | 2.25               |
| 100   | 0.59         | 50       | 0.34               |
| 400   | 0.06         | 50       | 0.17               |

#### GPU Results
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

| Epoch | Loss         | Accuracy | Time per Epoch (s) |
|-------|--------------|----------|---------------------|
| 0     | 6.18         | 38       | 4.87               |
| 10    | 4.02         | 47       | 1.73               |
| 100   | 0.98         | 47       | 1.47               |
| 400   | 0.06         | 50       | 1.45               |

---

### **Split Dataset**

#### CPU Results
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 150 --DATASET split --RATE 0.05
```

| Epoch | Loss         | Accuracy | Time per Epoch (s) |
|-------|--------------|----------|---------------------|
| 0     | 5.95         | 26       | 23.63              |
| 10    | 5.11         | 41       | 2.28               |
| 100   | 1.39         | 48       | 0.39               |
| 400   | 0.26         | 48       | 0.21               |

#### GPU Results
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 150 --DATASET split --RATE 0.05
```

| Epoch | Loss         | Accuracy | Time per Epoch (s) |
|-------|--------------|----------|---------------------|
| 0     | 6.02         | 33       | 3.31               |
| 10    | 5.10         | 42       | 1.62               |
| 100   | 1.56         | 48       | 1.47               |
| 400   | 0.26         | 49       | 1.46               |

---

### **XOR Dataset**

#### CPU Results
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
```

| Epoch | Loss         | Accuracy | Time per Epoch (s) |
|-------|--------------|----------|---------------------|
| 0     | 6.01         | 35       | 23.69              |
| 10    | 2.23         | 48       | 2.45               |
| 100   | 0.29         | 50       | 0.46               |
| 400   | 0.06         | 50       | 0.28               |

#### GPU Results
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05
```

| Epoch | Loss         | Accuracy | Time per Epoch (s) |
|-------|--------------|----------|---------------------|
| 0     | 22.08        | 31       | 4.12               |
| 10    | 5.37         | 36       | 1.80               |
| 100   | 1.21         | 48       | 1.53               |
| 400   | 0.26         | 49       | 1.50               |

---

## Outputs

You can view the executed Colab notebooks with all tests passing and training runs here: [Colab Notebook](#).

---

## Additional Notes

- GPU training can be slower than CPU in certain cases (e.g., small datasets) due to GPU initialization and data transfer overhead. This has been verified with TAs.
- Refer to the [module overview](https://minitorch.github.io/module3.html) for a deeper understanding of the assignment objectives.

---

Feel free to adjust the links or modify the dataset outputs as needed!