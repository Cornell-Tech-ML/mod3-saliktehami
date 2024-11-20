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
![Microsoft Edge 2024-11-20 15 29 49](https://github.com/user-attachments/assets/5cf6a164-ea83-4f84-b260-b6febd38f0dd)

| Matrix Size | CPU Time (s) | GPU Time (s) | Speedup |
|------------|--------------|--------------|---------|
| 64x64      | 0.00373      | 0.00696      | 0.54x   |
| 128x128    | 0.01487      | 0.01513      | 0.98x   |
| 256x256    | 0.07429      | 0.05264      | 1.41x   |
| 512x512    | 0.64368      | 0.20876      | 3.08x   |
| 1024x1024  | 4.29199      | 0.85038      | 5.05x   |

*Note*: Tests run on Google Colab GPU runtime. GPU starts showing significant speedup at matrix sizes larger than 256x256, achieving a 5x speedup at 1024x1024.*

---

## 3.5 Results

### **Simple Dataset**

#### CPU Results
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

| Epoch | Loss                 | Correct | Time per Epoch (s) | Total Time (s) |
|-------|----------------------|---------|---------------------|----------------|
|   0   | 2.734408903002974    | 42      | 12.1936            | 12.1936        |
|  10   | 0.8386641889479846   | 50      | 1.3245             | 14.5698        |
|  20   | 2.6556046432178477   | 43      | 0.8074             | 16.9553        |
|  30   | 0.4332539607470506   | 50      | 0.6252             | 19.3799        |
|  40   | 1.2477259765555464   | 50      | 0.5354             | 21.9503        |
|  50   | 0.3608970349737639   | 50      | 0.477              | 24.3281        |
|  60   | 0.6643757734961746   | 50      | 0.4379             | 26.7125        |
|  70   | 0.22986442796770432  | 50      | 0.4098             | 29.0947        |
|  80   | 0.6936022678136411   | 50      | 0.3897             | 31.5691        |
|  90   | 0.16695916234333047  | 50      | 0.3745             | 34.0751        |
| 100   | 0.2335182866736362   | 50      | 0.361              | 36.462         |
| 110   | 0.11393740951961209  | 50      | 0.3499             | 38.8387        |
| 120   | 0.12952987248120912  | 50      | 0.3405             | 41.1998        |
| 130   | 0.3789719917441468   | 50      | 0.3335             | 43.6888        |
| 140   | 0.027339240712952243 | 50      | 0.3273             | 46.147         |
| 150   | 0.1041167691212132   | 50      | 0.3213             | 48.5172        |
| 160   | 0.03370029472154799  | 50      | 0.3162             | 50.9051        |
| 170   | 0.2125303968541667   | 50      | 0.3116             | 53.2765        |
| 180   | 0.04157804345972533  | 50      | 0.3083             | 55.7998        |
| 190   | 0.06757108090156756  | 50      | 0.3048             | 58.2132        |
| 200   | 0.05464735774338346  | 50      | 0.3014             | 60.572         |
| 210   | 0.01879879019059296  | 50      | 0.2984             | 62.9532        |
| 220   | 0.1273975951642429   | 50      | 0.2957             | 65.3555        |
| 230   | 0.48949461845555603  | 50      | 0.2941             | 67.9273        |
| 240   | 0.0004428012773972089| 50      | 0.2917             | 70.3075        |
| 250   | 0.09848602090741797  | 50      | 0.2896             | 72.6801        |
| 260   | 0.02589616814018968  | 50      | 0.2875             | 75.0395        |
| 270   | 0.05069919573334311  | 50      | 0.2856             | 77.4036        |
| 280   | 0.045912222488962526 | 50      | 0.2846             | 79.9742        |
| 290   | 0.12088019200991947  | 50      | 0.283              | 82.3641        |
| 300   | 0.13860168590405403  | 50      | 0.2815             | 84.7304        |
| 310   | 0.10389687886263074  | 50      | 0.2801             | 87.0969        |
| 320   | 0.1307593481228511   | 50      | 0.2787             | 89.4631        |
| 330   | 0.16352748467119393  | 50      | 0.2781             | 92.0605        |
| 340   | 0.004019359942095672 | 50      | 0.277              | 94.4418        |
| 350   | 0.08039029312174209  | 50      | 0.2759             | 96.829         |
| 360   | 0.08161928481949353  | 50      | 0.2748             | 99.1903        |
| 370   | 0.08055577795799208  | 50      | 0.2739             | 101.604        |
| 380   | 0.013613624155835455 | 50      | 0.2735             | 104.1943       |
| 390   | 0.0728627857258801   | 50      | 0.2726             | 106.5865       |
| 400   | 0.0010756297017431352| 50      | 0.2717             | 108.9634       |
| 410   | 0.0017201458932204882| 50      | 0.2711             | 111.4193       |
| 420   | 0.12432237898918862  | 50      | 0.2704             | 113.8307       |
| 430   | 0.011691465145009279 | 50      | 0.2699             | 116.3479       |
| 440   | 0.12482718384778312  | 50      | 0.2692             | 118.7199       |
| 450   | 0.1278986132882372   | 50      | 0.2685             | 121.09         |
| 460   | 0.16303173374985522  | 50      | 0.2678             | 123.4494       |
| 470   | 0.07169552488681068  | 50      | 0.2673             | 125.9123       |
| 480   | 0.05551438857932521  | 50      | 0.2669             | 128.3969       |
| 490   | 0.0865062756061068   | 50      | 0.2663             | 130.7646       |
**Total Time:** 132.9243 s
**Epoch Duration:** 0.2658 s

#### GPU Results
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```
| Epoch | Loss                 | Correct | Time per Epoch (s) | Total Time (s) |
|-------|----------------------|---------|---------------------|----------------|
|   0   | 4.1869481211619615   | 42      | 4.5153             | 4.5153         |
|  10   | 0.27690650686306467  | 50      | 2.0989             | 23.0874        |
|  20   | 1.3270359493121102   | 50      | 1.9815             | 41.6106        |
|  30   | 0.3136649202075788   | 50      | 1.9364             | 60.0272        |
|  40   | 0.4226867808872227   | 50      | 1.9165             | 78.5769        |
|  50   | 0.4875762473717512   | 50      | 1.9054             | 97.1776        |
|  60   | 0.190427259761966    | 50      | 1.9008             | 115.949        |
|  70   | 0.004122477097476195 | 50      | 1.8941             | 134.4815       |
|  80   | 0.34832713892617095  | 50      | 1.8898             | 153.0737       |
|  90   | 0.23252822350338648  | 50      | 1.8859             | 171.6187       |
| 100   | 0.07387837362575975  | 50      | 1.8824             | 190.1251       |
| 110   | 0.40128835105815164  | 50      | 1.8806             | 208.7437       |
| 120   | 0.1008096383602751   | 50      | 1.8787             | 227.3193       |
| 130   | 0.07254716289654523  | 50      | 1.8759             | 245.739        |
| 140   | 0.06383889137616554  | 50      | 1.8745             | 264.3081       |
| 150   | 0.02944508946401469  | 50      | 1.8729             | 282.8088       |
| 160   | 0.20023335282874982  | 50      | 1.871              | 301.2268       |
| 170   | 0.008169202831542127 | 50      | 1.8697             | 319.7192       |
| 180   | 0.2915271566164994   | 50      | 1.8681             | 338.1257       |
| 190   | 0.018705228056593894 | 50      | 1.8674             | 356.6793       |
| 200   | 0.15982832754319126  | 50      | 1.8663             | 375.1224       |
| 210   | 0.045690101618620284 | 50      | 1.8658             | 393.674        |
| 220   | 0.06221532074352011  | 50      | 1.8654             | 412.262        |
| 230   | 0.034999170090268705 | 50      | 1.8641             | 430.6098       |
| 240   | 0.12461725142720118  | 50      | 1.8638             | 449.1814       |
| 250   | 0.0067980333102273935| 50      | 1.8635             | 467.7437       |
| 260   | 0.02432921344258157  | 50      | 1.8632             | 486.2895       |
| 270   | 0.09306059255655627  | 50      | 1.8624             | 504.7044       |
| 280   | 0.023526790485822055 | 50      | 1.8615             | 523.0854       |
| 290   | 0.034550794452366404 | 50      | 1.8611             | 541.5783       |
| 300   | 0.041426939723997906 | 50      | 1.8606             | 560.035        |
| 310   | 0.10104636852711918  | 50      | 1.8602             | 578.5253       |
| 320   | 0.12821388126319241  | 50      | 1.8595             | 596.9082       |
| 330   | 0.02197329525986233  | 50      | 1.859              | 615.33         |
| 340   | 0.06655402925391218  | 50      | 1.858              | 633.5859       |
| 350   | 0.013161328694444134 | 50      | 1.8575             | 651.9936       |
| 360   | 0.05128927348536151  | 50      | 1.8569             | 670.3364       |
| 370   | 0.024553946338482183 | 50      | 1.8562             | 688.6428       |
| 380   | 0.024556975005765678 | 50      | 1.8561             | 707.191        |
| 390   | 0.06414115326775706  | 50      | 1.8555             | 725.4856       |
| 400   | 0.07987657812493054  | 50      | 1.8548             | 743.7621       |
| 410   | 0.01975767091615446  | 50      | 1.8542             | 762.0931       |
| 420   | 0.04254816123903718  | 50      | 1.8537             | 780.4226       |
| 430   | 0.0662270261744189   | 50      | 1.8533             | 798.7717       |
| 440   | 0.09286523138551138  | 50      | 1.8525             | 816.969        |
| 450   | 0.061404727575816435 | 50      | 1.852              | 835.2741       |
| 460   | 0.0034159446965021725| 50      | 1.8514             | 853.4951       |
| 470   | 0.0832452862551018   | 50      | 1.8508             | 871.7343       |
| 480   | 0.060078795607983466 | 50      | 1.8502             | 889.9454       |
| 490   | 0.044705273982237545 | 50      | 1.8496             | 908.1558       |
**Total Time:** 924.9793 s
**Epoch Duration:** 1.85 s

---

### **Split Dataset**

#### CPU Results
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05
```

| Epoch | Loss                 | Correct | Time per Epoch (s) | Total Time (s) |
|-------|----------------------|---------|---------------------|----------------|
|   0   | 8.39157753784894     | 31      | 12.1515            | 12.1515        |
|  10   | 4.4758392834513705   | 36      | 1.32               | 14.5195        |
|  20   | 4.8893647135456915   | 37      | 0.8041             | 16.8861        |
|  30   | 2.8359999510992306   | 48      | 0.6275             | 19.4523        |
|  40   | 1.7790201457281491   | 46      | 0.5325             | 21.8324        |
|  50   | 0.7210800405801486   | 50      | 0.4748             | 24.2155        |
|  60   | 0.8697187893683295   | 50      | 0.4356             | 26.5718        |
|  70   | 0.7419475261351631   | 50      | 0.4075             | 28.9347        |
|  80   | 2.359896946071627    | 47      | 0.3887             | 31.4854        |
|  90   | 1.409911149895089    | 48      | 0.3719             | 33.8401        |
| 100   | 0.7278198747307154   | 49      | 0.3585             | 36.2095        |
| 110   | 1.316590206021789    | 50      | 0.3475             | 38.578         |
| 120   | 1.10098026390943     | 48      | 0.3384             | 40.9484        |
| 130   | 1.3196313544834066   | 50      | 0.332              | 43.495         |
| 140   | 0.42597789036677536  | 50      | 0.3251             | 45.8455        |
| 150   | 0.5904403864376895   | 50      | 0.3193             | 48.2144        |
| 160   | 1.4439994827577072   | 50      | 0.3142             | 50.5833        |
| 170   | 0.5821702945891267   | 50      | 0.3099             | 52.9867        |
| 180   | 0.4716023521898508   | 50      | 0.3066             | 55.4974        |
| 190   | 0.8125577712575499   | 50      | 0.3029             | 57.8449        |
| 200   | 0.2554314304481932   | 50      | 0.2995             | 60.2094        |
| 210   | 0.8002124874620385   | 50      | 0.2965             | 62.569         |
| 220   | 0.6329426136189855   | 50      | 0.294              | 64.9795        |
| 230   | 0.3517793211344743   | 50      | 0.2921             | 67.4839        |
| 240   | 0.5454979677469712   | 50      | 0.2898             | 69.8307        |
| 250   | 0.3258338390860207   | 50      | 0.2876             | 72.1853        |
| 260   | 0.15078379131618053  | 50      | 0.2856             | 74.542         |
| 270   | 0.16887337975716238  | 50      | 0.284              | 76.9628        |
| 280   | 0.6991953058437892   | 50      | 0.2827             | 79.4282        |
| 290   | 0.4566440973095329   | 50      | 0.2812             | 81.815         |
| 300   | 0.35739320370439165  | 50      | 0.2797             | 84.1809        |
| 310   | 0.2062509616324841   | 50      | 0.2783             | 86.5513        |
| 320   | 0.1335484094880042   | 50      | 0.2774             | 89.0346        |
| 330   | 0.4596177178448576   | 50      | 0.2763             | 91.4557        |
| 340   | 0.38306547824761117  | 50      | 0.2751             | 93.8217        |
| 350   | 0.27309416423874666  | 50      | 0.274              | 96.172         |
| 360   | 0.5068643813183243   | 50      | 0.2729             | 98.5154        |
| 370   | 0.021609053156652607 | 50      | 0.2726             | 101.13         |
| 380   | 0.10689905654690529  | 50      | 0.2717             | 103.5344       |
| 390   | 0.06253189117081372  | 50      | 0.2709             | 105.918        |
| 400   | 0.027110472409633338 | 50      | 0.27               | 108.2846       |
| 410   | 0.2659661563452908   | 50      | 0.2692             | 110.6352       |
| 420   | 0.3680736538389301   | 50      | 0.2688             | 113.178        |
| 430   | 0.010224237834726951 | 50      | 0.2681             | 115.553        |
| 440   | 0.012807570919327124 | 50      | 0.2674             | 117.9128       |
| 450   | 0.12488751656522792  | 50      | 0.2667             | 120.2763       |
| 460   | 0.2789897164112648   | 50      | 0.2661             | 122.6667       |
| 470   | 0.3437003540175172   | 50      | 0.266              | 125.2928       |
| 480   | 0.20542885419195447  | 50      | 0.2655             | 127.6855       |
| 490   | 0.20508501412623212  | 50      | 0.2649             | 130.0862       |
**Total Time:** 132.2434 s
**Epoch Duration:** 0.2645 s

#### GPU Results
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05
```

| Epoch | Loss                 | Correct | Time per Epoch (s) | Total Time (s) |
|-------|----------------------|---------|---------------------|----------------|
|   0   | 9.281234083875603    | 17      | 3.5967             | 3.5967         |
|  10   | 4.681988161814642    | 44      | 2.0357             | 22.3932        |
|  20   | 2.66882833843097     | 44      | 1.9597             | 41.1533        |
|  30   | 4.814546762836826    | 44      | 1.93               | 59.8303        |
|  40   | 2.367555191149844    | 49      | 1.9192             | 78.687         |
|  50   | 1.8323586260501774   | 50      | 1.9154             | 97.6868        |
|  60   | 0.9931321729821994   | 48      | 1.9117             | 116.6118       |
|  70   | 1.5491043466042078   | 49      | 1.9102             | 135.6225       |
|  80   | 1.5539604718275943   | 50      | 1.9073             | 154.4947       |
|  90   | 1.2142568285129862   | 48      | 1.9065             | 173.4898       |
| 100   | 0.7513071018531501   | 50      | 1.9047             | 192.3697       |
| 110   | 1.1723046142373144   | 50      | 1.902              | 211.1174       |
| 120   | 1.5809716782915884   | 50      | 1.9                | 229.8945       |
| 130   | 0.9035815856176892   | 48      | 1.8979             | 248.6204       |
| 140   | 0.47464410423995085  | 49      | 1.8959             | 267.3215       |
| 150   | 1.638654688975262    | 49      | 1.8943             | 286.0324       |
| 160   | 1.8143197104552624   | 50      | 1.8927             | 304.732        |
| 170   | 1.5562485074627477   | 50      | 1.8912             | 323.3949       |
| 180   | 0.6284703880995919   | 50      | 1.89               | 342.0982       |
| 190   | 0.9339764106630004   | 50      | 1.8885             | 360.701        |
| 200   | 0.3538144737982994   | 50      | 1.8875             | 379.3955       |
| 210   | 0.34445691718202864  | 50      | 1.8869             | 398.1389       |
| 220   | 0.5233529455741062   | 50      | 1.8869             | 416.9956       |
| 230   | 0.3756923915258422   | 49      | 1.8853             | 435.5036       |
| 240   | 0.2035604845879015   | 50      | 1.8841             | 454.0776       |
| 250   | 0.6253398531783397   | 50      | 1.8834             | 472.721        |
| 260   | 0.5053671951151926   | 50      | 1.8829             | 491.4349       |
| 270   | 0.06178391206286859  | 50      | 1.8816             | 509.908        |
| 280   | 0.3762580080106292   | 49      | 1.8805             | 528.4222       |
| 290   | 0.13671894687214153  | 50      | 1.8797             | 546.9889       |
| 300   | 0.15645723114791912  | 50      | 1.8793             | 565.6723       |
| 310   | 0.3019748324552598   | 50      | 1.8788             | 584.3146       |
| 320   | 0.16214752493274445  | 50      | 1.8784             | 602.9552       |
| 330   | 0.6045875896011622   | 50      | 1.8783             | 621.7027       |
| 340   | 0.11171382083715001  | 50      | 1.8777             | 640.3026       |
| 350   | 0.19372648527517528  | 50      | 1.8778             | 659.0947       |
| 360   | 0.513264478950994    | 50      | 1.8775             | 677.7889       |
| 370   | 0.6684623965888898   | 50      | 1.8772             | 696.4534       |
| 380   | 0.3447022804120512   | 50      | 1.8773             | 715.2332       |
| 390   | 0.22451274000243515  | 50      | 1.8767             | 733.8044       |
| 400   | 0.07269062521940164  | 50      | 1.876              | 752.2724       |
| 410   | 0.2288878180675256   | 50      | 1.8757             | 770.9215       |
| 420   | 0.05781861427947161  | 50      | 1.8757             | 789.6498       |
| 430   | 0.01312022802005489  | 50      | 1.8752             | 808.2156       |
| 440   | 0.06617736300745422  | 50      | 1.8747             | 826.7577       |
| 450   | 0.19621055788303848  | 50      | 1.8745             | 845.3984       |
| 460   | 0.18626770289025757  | 50      | 1.8741             | 863.9758       |
| 470   | 0.2582952818320537   | 50      | 1.874              | 882.6543       |
| 480   | 0.354221962285631    | 50      | 1.8739             | 901.3304       |
| 490   | 0.6728909619212111   | 49      | 1.8734             | 919.8566       |
**Total Time:** 936.5749 s
**Epoch Duration:** 1.8731 s


---

### **XOR Dataset**

#### CPU Results
```bash
!python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET xor --RATE 0.05
```

| Epoch | Loss                 | Correct | Time per Epoch (s) | Total Time (s) |
|-------|----------------------|---------|---------------------|----------------|
|   0   | 21.32629885926305    | 31      | 11.9009            | 11.9009        |
|  10   | 3.8817057845356677   | 44      | 1.316              | 14.4759        |
|  20   | 2.4439319009813376   | 46      | 0.8028             | 16.8589        |
|  30   | 2.011776259167944    | 47      | 0.6196             | 19.2078        |
|  40   | 2.044204555292045    | 45      | 0.5258             | 21.5595        |
|  50   | 0.36443612474170217  | 49      | 0.4685             | 23.8952        |
|  60   | 1.5716815406310856   | 46      | 0.434              | 26.4763        |
|  70   | 0.7594443338245638   | 46      | 0.4063             | 28.8493        |
|  80   | 1.1553087424449096   | 49      | 0.3853             | 31.2105        |
|  90   | 0.771077916712352    | 49      | 0.3691             | 33.5855        |
| 100   | 0.964788179026314    | 49      | 0.356              | 35.9585        |
| 110   | 0.4389756125001531   | 50      | 0.3466             | 38.478         |
| 120   | 1.4802138209697298   | 50      | 0.3376             | 40.8534        |
| 130   | 1.1358898060296574   | 50      | 0.3298             | 43.2053        |
| 140   | 1.2032602708863407   | 50      | 0.3232             | 45.5672        |
| 150   | 0.959335532252897    | 50      | 0.3183             | 48.0599        |
| 160   | 0.0947610122662594   | 49      | 0.3141             | 50.5664        |
| 170   | 0.23743085910011497  | 50      | 0.3096             | 52.948         |
| 180   | 0.3396088259077795   | 50      | 0.3056             | 55.3096        |
| 190   | 0.2564721468143938   | 50      | 0.3019             | 57.6628        |
| 200   | 0.10663305828058965  | 50      | 0.2991             | 60.1229        |
| 210   | 0.7717080567218991   | 50      | 0.2968             | 62.6215        |
| 220   | 0.3334148517638334   | 50      | 0.294              | 64.9761        |
| 230   | 0.761919002261507    | 50      | 0.2915             | 67.3386        |
| 240   | 0.9428967053129902   | 50      | 0.2892             | 69.6874        |
| 250   | 0.4652854512205734   | 50      | 0.2875             | 72.156         |
| 260   | 0.2926519957157305   | 50      | 0.2859             | 74.6134        |
| 270   | 0.33682553301060464  | 50      | 0.2841             | 76.9853        |
| 280   | 0.05585910542886467  | 50      | 0.2824             | 79.3429        |
| 290   | 0.2001446455457525   | 50      | 0.2807             | 81.6898        |
| 300   | 0.20488285039847232  | 50      | 0.2798             | 84.2121        |
| 310   | 0.15716636585207366  | 50      | 0.2787             | 86.6649        |
| 320   | 0.3083156829047872   | 50      | 0.2773             | 89.0183        |
| 330   | 0.6538886821926777   | 50      | 0.2761             | 91.3983        |
| 340   | 0.49634094694348635  | 50      | 0.2751             | 93.8216        |
| 350   | 0.4270326288898577   | 50      | 0.2746             | 96.3875        |
| 360   | 0.1741426225230424   | 50      | 0.2736             | 98.7824        |
| 370   | 0.22567644538797252  | 50      | 0.2728             | 101.2077       |
| 380   | 0.26725185048379363  | 50      | 0.2718             | 103.5508       |
| 390   | 0.07467892565491815  | 50      | 0.2708             | 105.8951       |
| 400   | 0.15700441574561147  | 50      | 0.2704             | 108.4487       |
| 410   | 0.18445649443731799  | 50      | 0.2695             | 110.7806       |
| 420   | 0.4686807016093022   | 50      | 0.2687             | 113.1068       |
| 430   | 0.21591522949841901  | 50      | 0.2678             | 115.4284       |
| 440   | 0.021195528521761466 | 50      | 0.267              | 117.7601       |
| 450   | 0.15120233466964828  | 50      | 0.2669             | 120.3553       |
| 460   | 0.008670806152845623 | 50      | 0.2662             | 122.709        |
| 470   | 0.008035369640504705 | 50      | 0.2655             | 125.0518       |
| 480   | 0.08623575851032203  | 50      | 0.2651             | 127.5257       |
| 490   | 0.26207691164564917  | 50      | 0.2645             | 129.8669       |
**Total Time:** 132.1801 s
**Epoch Duration:** 0.2644 s


#### GPU Results
```bash
!python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET xor --RATE 0.05
```
| Epoch | Loss                 | Correct | Time per Epoch (s) | Total Time (s) |
|-------|----------------------|---------|---------------------|----------------|
|   0   | 6.235867686996221    | 24      | 4.4809             | 4.4809         |
|  10   | 3.696007961823046    | 44      | 2.1397             | 23.5363        |
|  20   | 3.557856710818963    | 46      | 2.022              | 42.461         |
|  30   | 2.4605180896484153   | 43      | 1.9755             | 61.2397        |
|  40   | 0.9362453492398745   | 46      | 1.9518             | 80.0222        |
|  50   | 2.2148491796612624   | 48      | 1.9362             | 98.7443        |
|  60   | 0.7397490233864751   | 47      | 1.9257             | 117.4703       |
|  70   | 2.5709406315533423   | 49      | 1.9169             | 136.0965       |
|  80   | 1.3240575136903732   | 49      | 1.9089             | 154.6218       |
|  90   | 0.8738217069661318   | 48      | 1.9064             | 173.4863       |
| 100   | 1.9022781650456748   | 48      | 1.9053             | 192.4323       |
| 110   | 0.8593841823516295   | 49      | 1.9035             | 211.2866       |
| 120   | 1.1037639264365373   | 49      | 1.9018             | 230.1174       |
| 130   | 1.2510533570163918   | 49      | 1.9013             | 249.0724       |
| 140   | 1.7331016344432324   | 49      | 1.9009             | 268.0251       |
| 150   | 0.5657280047058353   | 49      | 1.8997             | 286.8535       |
| 160   | 0.769363685880996    | 49      | 1.8969             | 305.4015       |
| 170   | 1.2632236506603054   | 50      | 1.8952             | 324.0815       |
| 180   | 0.8503222513235441   | 50      | 1.894              | 342.8207       |
| 190   | 1.0436535105137439   | 50      | 1.8916             | 361.3008       |
| 200   | 1.22208573200736     | 50      | 1.8889             | 379.6661       |
| 210   | 0.3862200595874152   | 50      | 1.8868             | 398.1152       |
| 220   | 0.4662469571590653   | 50      | 1.8846             | 416.5044       |
| 230   | 0.40760399293647515  | 50      | 1.8823             | 434.8108       |
| 240   | 0.5612950259827852   | 50      | 1.8805             | 453.1896       |
| 250   | 0.22710276948187993  | 50      | 1.8789             | 471.6094       |
| 260   | 0.6210342951718517   | 50      | 1.8776             | 490.0511       |
| 270   | 0.3697050239530079   | 50      | 1.8763             | 508.4765       |
| 280   | 0.1780144498117433   | 50      | 1.8749             | 526.8559       |
| 290   | 0.4225442327167754   | 50      | 1.8736             | 545.2275       |
| 300   | 0.5429488356151115   | 50      | 1.872              | 563.4809       |
| 310   | 0.30032698959841964  | 50      | 1.8712             | 581.9301       |
| 320   | 0.3341210814458028   | 50      | 1.8707             | 600.4928       |
| 330   | 0.09179193197670936  | 50      | 1.8693             | 618.7406       |
| 340   | 0.07404058313433108  | 50      | 1.8681             | 637.0213       |
| 350   | 0.4522361702320395   | 50      | 1.8672             | 655.3865       |
| 360   | 0.42727552688513337  | 50      | 1.8663             | 673.7501       |
| 370   | 0.07196898177711031  | 50      | 1.8655             | 692.0939       |
| 380   | 0.4389195937466251   | 50      | 1.8647             | 710.4415       |
| 390   | 0.2833606612804548   | 50      | 1.8639             | 728.7966       |
| 400   | 0.21165229743381067  | 50      | 1.8631             | 747.1231       |
| 410   | 0.21651993800514963  | 50      | 1.8627             | 765.5509       |
| 420   | 0.3186020278212222   | 50      | 1.8619             | 783.8588       |
| 430   | 0.2686358786964569   | 50      | 1.8611             | 802.1554       |
| 440   | 0.28096468194877416  | 50      | 1.8603             | 820.4133       |
| 450   | 0.3389989484234791   | 50      | 1.8599             | 838.8002       |
| 460   | 0.3795132728438986   | 50      | 1.8595             | 857.2148       |
| 470   | 0.17648797691426427  | 50      | 1.8589             | 875.5262       |
| 480   | 0.27062309522894645  | 50      | 1.8587             | 894.0453       |
| 490   | 0.2742576090676533   | 50      | 1.8582             | 912.3883       |
**Total Time:** 928.8481 s
**Epoch Duration:** 1.8577 s
---

## Outputs

Here is the link to the Colab Notebook: [Notebook](https://colab.research.google.com/drive/1ZHBRGdYs7AfHbMkOM33E2pk2_4PgLlFH?usp=sharing).
