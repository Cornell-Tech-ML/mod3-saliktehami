# MiniTorch Module 3

## Parallel Performance Analysis

Command executed:
```bash
(.venv) salik@Saliks-MacBook-Pro mod3-saliktehami % python project/parallel_check.py

================================================================================
Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (163)
================================================================================

Parallel loop listing for Function tensor_map.<locals>._map:
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (163)
-----------------------------------------------------------------------|loop #ID
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

Fusing loops
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops, there are 1 parallel for-loop(s) (originating from loops labeled: #0).

Before Optimization
The loop structure before optimization is displayed.

After Optimization
The parallel structure is already optimal.

Loop invariant code motion
The following allocations are hoisted out of the parallel loop labeled #0:

Allocation for out_index:
```out_index = np.empty(MAX_DIMS, np.int32)

Allocation for in_index:
```in_index = np.empty(MAX_DIMS, np.int32)

================================================================================
Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (207)
================================================================================

Parallel loop listing for Function tensor_zip.<locals>._zip:
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (207)
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

Fusing loops
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops, there are 1 parallel for-loop(s) (originating from loops labeled: #1).

Loop invariant code motion
The following allocations are hoisted out of the parallel loop labeled #1:

Allocation for out_index:
```out_index = np.empty(MAX_DIMS, np.int32)

Allocation for a_index:
```a_index = np.empty(MAX_DIMS, np.int32)

Allocation for b_index:
```b_index = np.empty(MAX_DIMS, np.int32)

================================================================================
Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (256)
================================================================================

Parallel loop listing for Function tensor_reduce.<locals>._reduce:
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (256)
-----------------------------------------------------------------------|loop #ID
def _reduce(                                                         |
    out: Storage,                                                    |
    out_shape: Shape,                                                |
    out_strides: Strides,                                            |
    a_storage: Storage,                                              |
    a_shape: Shape,                                                  |
    a_strides: Strides,                                              |
    reduce_dim: int,                                                 |
) -> None:                                                           |
    out_index = np.zeros(MAX_DIMS, np.int32)-------------------------| #2
    reduce_size = a_shape[reduce_dim]                                |
                                                                     |
    for i in prange(len(out)):---------------------------------------| #3
        # Thread-local indices                                       |
        out_index = np.empty(MAX_DIMS, np.int32)                     |
        local_index = np.empty(MAX_DIMS, np.int32)                   |
        # ... Remaining Code                                         |


Fusing loops
Attempting fusion of parallel loops...
Following the attempted fusion, there are 1 parallel for-loop(s) (originating from loops labeled: #3).

================================================================================
Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (289)
================================================================================

Parallel loop listing for Function _tensor_matrix_multiply:
/Users/salik/Coding/workspace-mle/mod3-saliktehami/minitorch/fast_ops.py (289)
-----------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                         |
    out: Storage,                                                    |
    out_shape: Shape,                                                |
    out_strides: Strides,                                            |
    a_storage: Storage,                                              |
    a_shape: Shape,                                                  |
    a_strides: Strides,                                              |
    b_storage: Storage,                                              |
    b_shape: Shape,                                                  |
    b_strides: Strides,                                              |
) -> None:                                                           |
    for batch in prange(batch_size):---------------------------------| #5
        # Calculate batch offsets once per batch                    |
        a_batch_offset = batch * a_batch_stride                     |
        b_batch_offset = batch * b_batch_stride                     |
        out_batch_offset = batch * out_strides[0]                   |
        # Remaining Code                                            |

Loop Nest Optimizations
Parallel loop #5 remains parallel.
Loop #4 rewritten as a serial loop for optimization.

```

# hello
