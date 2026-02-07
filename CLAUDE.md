# NLSE - Claude Code Guidelines

## Project Overview
Split-step Fourier solver for Nonlinear Schrodinger / Gross-Pitaevskii equations.
Published in JOSS (DOI: 10.21105/joss.06607). GPLv3 licensed. Version 2.3.0.

## Architecture
- **Language:** Python 3.8+
- **Package:** `NLSE/` (source), `tests/` (pytest), `examples/`
- **Build:** `setup.py` (setuptools), no pyproject.toml yet
- **Backends:** CPU (pyfftw + numba), GPU (cupy + pyvkfft), OpenCL (pyopencl + pyvkfft)

### Class Hierarchy
```
NLSE (2D spatial, base class)
├── NLSE_1d
├── NLSE_3d (spatio-temporal with GVD)
├── GPE (Gross-Pitaevskii, atomic units)
└── CNLSE (coupled 2-component)
    ├── CNLSE_1d
    └── DDGPE (driven-dissipative polaritons)
```

### Key Files
- `NLSE/nlse.py` - Base class with `out_field()`, `split_step()`, `split_step_RK4()`
- `NLSE/kernels_cpu.py` - Numba JIT kernels (`nl_prop`, `rabi_coupling`, etc.)
- `NLSE/kernels_gpu.py` - CuPy fused kernels (same API)
- `NLSE/kernels_cl.py` - PyOpenCL kernels (same API, incomplete)
- `NLSE/callbacks.py` - `sample()`, `norm()`, `adapt_delta_z()`, `evaluate_delta_n()`
- `NLSE/utils.py` - Backend detection, constants

### Key Patterns
- Backend selected at import time via `utils.__BACKEND__`, switchable per instance
- Template method: `out_field()` is the main loop, calls overridden `split_step()` / `_build_propagator()`
- Callbacks: `callback(self, A, z, i)` signature, called each propagation step
- Broadcasting: GPU supports numpy-style broadcasting over batch dimensions

## Running Tests
```bash
pytest tests/          # all tests (CPU + GPU if available)
pytest tests/ -k cpu   # CPU only
```

## Conventions
- SI units throughout ("God given units")
- Field arrays are complex64 (single) or complex128 (double)
- `precision="single"` = O(dz), `precision="double"` = O(dz^3) (Strang splitting)
- `delta_z` auto-computed from Rayleigh length and nonlinear length
