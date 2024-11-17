# MiniTorch Module 3

## Parallel Performance Analysis
(.venv) salik@Saliks-MacBook-Pro mod3-saliktehami % python project/parallel_check.py

### MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (163) 
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
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (173) 
is hoisted out of the parallel loop labelled #0 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (174) 
is hoisted out of the parallel loop labelled #0 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None

### ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (207)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (207) 
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
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (220) 
is hoisted out of the parallel loop labelled #1 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (221) 
is hoisted out of the parallel loop labelled #1 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (222) 
is hoisted out of the parallel loop labelled #1 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
### REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (256)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (256) 
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
    # Parallelize the outer loop over output elements            | 
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
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (271) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (272) 
is hoisted out of the parallel loop labelled #3 (it will be performed before the
 loop is executed and reused inside the loop):
   Allocation:: local_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
### MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (289)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (289) 
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
                                                                                        | 
    # Basic compatibility check                                                         | 
    assert a_shape[-1] == b_shape[-2]                                                   | 
                                                                                        | 
    # Get batch strides (0 if not batched)                                              | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                              | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                              | 
                                                                                        | 
    # Get key dimensions                                                                | 
    batch_size = max(out_shape[0], 1)    # Number of batches                            | 
    M = out_shape[1]                     # Rows in output                               | 
    N = out_shape[2]                     # Cols in output                               | 
    K = a_shape[-1]                      # Shared dimension (cols in A, rows in B)      | 
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

