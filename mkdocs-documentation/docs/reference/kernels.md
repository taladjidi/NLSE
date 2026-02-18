# Kernels

Kernels are backend-specific implementations of the numerical operations used by the solvers. They are not meant to be called directly; they serve as the computational backend for the solver methods.

All backends implement the same set of kernel functions: `nl_prop`, `nl_prop_without_V`, `nl_prop_c`, `nl_prop_without_V_c`, `square_mod`, `square_mod_nl_prop`, `square_mod_nl_prop_v`, `apply_propagator`, and `rabi_coupling`.

## CPU (Numba)

CPU kernels use [Numba](https://numba.readthedocs.io/en/stable/user/index.html) JIT compilation for performance. They use multithreading internally and benefit from higher core counts.

CPU kernels do not support broadcasting (parallel simulations).

::: NLSE.kernels.cpu

## CUPY (CUDA)

GPU kernels use the [`cupy.fuse`](https://docs.cupy.dev/en/stable/reference/generated/cupy.fuse.html) API to JIT-compile array operations into a single GPU kernel, maximizing GPU occupancy and minimizing memory bandwidth overhead.

Arrays are mutated in place to avoid costly memory allocations:

```python
@cp.fuse
def kernel(A, ...):
    A += ...
    A *= ...
```

::: NLSE.kernels.cupy

## OpenCL

The OpenCL backend uses native C kernels defined in `NLSE/kernels/cl_source/kernels.cl`. These kernels are template-substituted at runtime for single or double precision and compiled with `-cl-fast-relaxed-math -cl-mad-enable` flags.

Native C kernels replace PyOpenCL array expressions to avoid implicit kernel launches and temporary buffer allocations. For example, the `rabi_coupling` kernel performs all coupling operations in a single kernel launch instead of 6 implicit launches.

Compiled programs are cached by `(context_hash, precision)` to avoid recompilation.

::: NLSE.kernels.cl

## MLX (Apple Silicon)

The MLX backend uses Apple's MLX framework for Metal-accelerated computation on Apple Silicon. Kernels use `mx.compile` for fused Metal kernels and return modified arrays via donation.

::: NLSE.kernels.mlx_kernels
