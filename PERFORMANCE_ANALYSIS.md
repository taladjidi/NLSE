# NLSE Performance Analysis
## Based on Profiling at Realistic Grid Sizes (256x256 to 4096x4096)

**Date:** 2026-02-08
**Baseline:** 2048x2048 (typical production size)
**Test:** 1mm propagation (single split_step)

---

## Executive Summary

### Key Findings

1. **OpenCL is 1.54x FASTER than CPU at 2048x2048** (49ms vs 76ms)
2. **Initialization dominates single-shot performance** (57.6% in `_build_propagator`)
3. **FFT operations are 90% of split_step time** (fundamental limit)
4. **Small grid overhead is extreme** (256x256 is 38x slower per element than 2048x2048)
5. **Previous assumptions about GPU overhead were based on pathologically small grids (N=64)**

### Performance at Scale

| Grid Size | Time (CPU) | ns/element | Notes |
|-----------|------------|------------|-------|
| 256×256   | 0.125s     | 1912.69    | Initialization overhead dominates |
| 512×512   | 0.013s     | 49.31      | Sweet spot begins |
| 1024×1024 | 0.041s     | 39.50      | Good scaling |
| **2048×2048** | **0.173s** | **41.14** | **Baseline** |
| 4096×4096 | 0.725s     | 43.20      | Excellent scaling |

**Scaling:** Nearly O(N²) from 512×512 to 4096×4096 (ideal for FFT-based solver)

---

## Detailed Profiling Results (2048×2048, CPU)

### Time Breakdown - `out_field()` (176ms total)

| Component | Time | % | Calls | Notes |
|-----------|------|---|-------|-------|
| `_build_propagator` | 101.5ms | **57.6%** | 1 | **BOTTLENECK #1** |
| `split_step` | 43.9ms | **24.9%** | 1 | **BOTTLENECK #2** |
| `_prepare_output_array` | 23.0ms | **13.1%** | 1 | One-time cost |
| `_build_fft_plan` | 7.7ms | 4.4% | 1 | One-time cost |
| Other | 0.0ms | 0.0% | - | Negligible |

### Time Breakdown - `split_step()` (44ms total)

| Operation | Time | % | Notes |
|-----------|------|---|-------|
| `ifft` | 21.5ms | **49.0%** | FFTW, fundamental limit |
| `fft` | 18.2ms | **41.6%** | FFTW, fundamental limit |
| `nl_prop_without_V` | 2.1ms | 4.8% | Numba JIT kernel |
| Propagator multiply | 1.3ms | 3.0% | Pure numpy |
| `square_mod` | 0.6ms | 1.5% | Numba JIT kernel |

**Key Insight:** FFT operations account for 90.6% of split_step time. This is fundamental - optimization opportunities are limited.

### Time Breakdown - `_prepare_output_array()` (23ms total)

| Operation | Time | % | Issue |
|-----------|------|---|-------|
| `arr * delta_X * delta_Y` + `astype` | 6.5ms | 28.4% | DoubleDowncastWarning |
| `arr = E.real² + E.imag²` | 5.7ms | 25.0% | Creates temp array |
| `allocate_field` | 4.8ms | 20.8% | pyfftw byte alignment |
| Broadcasting `A[:] = (E_00 * E_in)` | 3.8ms | 16.7% | Necessary |
| `allocate_real_field` | 1.4ms | 6.2% | pyfftw byte alignment |
| `np.sum` | 0.7ms | 3.0% | Fast |

---

## Backend Comparison (2048×2048)

### Single Propagation (1mm, 1 split_step)

| Backend | Time | Speedup | Notes |
|---------|------|---------|-------|
| CPU     | 76ms | 1.00x   | Baseline |
| OpenCL  | **49ms** | **1.54x** | ✅ Faster! |

**Reversal of Previous Assessment:** At N=64, OpenCL was 2.4x slower. At N=2048, OpenCL is 1.54x faster!

### OpenCL Bottlenecks (cProfile, 180ms total)

| Operation | Time | % | Notes |
|-----------|------|---|-------|
| `enqueue_copy` (CPU↔GPU transfers) | 48ms | 26.7% | Data movement |
| `to_numpy` (GPU→CPU) | 29ms | 16.1% | Getting results back |
| `from_numpy` (CPU→GPU) | 19ms | 10.6% | Sending input |
| `_build_propagator` | 17ms | 9.4% | Still significant |
| `split_step` kernel work | ~7ms | ~4% | Fast! |

**Key Insight:** For single-shot propagation, transfers dominate (47%). For long propagations (many steps), compute would dominate → larger GPU advantage.

---

## Critical Bottlenecks & Solutions

### 1. `_build_propagator` - 57.6% of single-shot time ⚠️⚠️⚠️

**Problem:** Exponential calculation of propagator matrix.

```python
# NLSE/solvers/nlse.py:171
def _build_propagator(self, precision: str = "single") -> np.ndarray:
    dtype = np.complex128 if precision == "double" else np.complex64
    propagator = np.exp(
        -1j * 0.5 * (self.Kxx**2 + self.Kyy**2) / self.k * self.delta_z,
        dtype=dtype,
    )
    return propagator
```

**Why slow:** `np.exp()` on 2048×2048 complex array is expensive.

**Current behavior:** Propagator cached after first call → one-time cost.

**Solutions:**
- ✅ **Already optimal** - Caching means this only happens once per `out_field` call
- For multiple `out_field` calls with same parameters, could cache at class level
- For parametric sweeps, consider `lru_cache` on propagator function

**Impact:**
- Single propagation: HIGH (57.6%)
- Long propagation (100 steps): LOW (2.2%, amortized over steps)

**Recommendation:** Add class-level propagator caching for repeated calls.

---

### 2. FFT Operations - 90% of `split_step` time 🔒

**Problem:** FFT + IFFT take 39.7ms out of 44ms per step.

**Why slow:** FFT is O(N² log N) - this is the fundamental algorithmic cost.

**Solutions:**
- ✅ **Already using FFTW** (fastest CPU FFT library)
- ✅ **Already using wisdom** (optimizes FFT plans)
- ❌ **Cannot significantly improve** - this is fundamental

**Impact:** Dominates long propagations (e.g., 100 steps = 4 seconds of FFT)

**Recommendation:**
- Accept this as fundamental limit
- Use GPU backends for long propagations (VkFFT is highly optimized)
- Consider domain decomposition for multi-GPU scaling

---

### 3. GPU Data Transfers - 47% of OpenCL time (single propagation) ⚠️

**Problem:** Copying data to/from GPU dominates single-shot performance.

**Current behavior:**
```python
# NLSE/solvers/nlse.py:517-518
if self._backend.name in ["CUPY", "CL"]:
    self._send_arrays_to_gpu()  # Every out_field call!
```

**Solutions:**

**Option A: Persistent GPU arrays** (for multiple propagations)
```python
class NLSE:
    def __init__(self, ...):
        self._gpu_persistent = False
        self._arrays_on_gpu = False

    def enable_gpu_persistence(self):
        """Keep arrays on GPU between out_field calls."""
        self._gpu_persistent = True
        if self._backend.name in ["CUPY", "CL"]:
            self._send_arrays_to_gpu()
            self._arrays_on_gpu = True

    def out_field(self, E_in, z, ...):
        if self._backend.name in ["CUPY", "CL"]:
            if not self._arrays_on_gpu:
                self._send_arrays_to_gpu()
                self._arrays_on_gpu = True
        # ... rest of method
```

**Option B: Batch interface** (propagate multiple fields at once)
```python
def out_field_batch(self, E_in_batch, z, ...):
    """Propagate batch of fields simultaneously.

    Args:
        E_in_batch: shape (batch_size, NX, NY)
    Returns:
        E_out_batch: shape (batch_size, NX, NY)
    """
    # Send batch once, process all, return batch
    # Amortizes transfer cost over batch
```

**Impact:**
- Single propagation: HIGH (47% → ~5% with persistence)
- Long single propagation: MEDIUM (amortized, but still relevant)
- Multiple fields: HIGH (batch processing)

**Recommendation:**
1. Implement persistent GPU mode (high priority)
2. Consider batch interface for parameter sweeps

---

### 4. Normalization Array Operations - 13% ⚠️

**Problem:** Creating temporary arrays, DoubleDowncastWarning.

```python
# NLSE/solvers/nlse.py:240-242
arr = E_in.real * E_in.real + E_in.imag * E_in.imag  # Temp array 1
arr = (arr * self.delta_X * self.delta_Y).astype(E_in.real.dtype)  # Temp array 2 + warning
```

**Solutions:**

**Fix 1: Use contiguous formula**
```python
arr = (E_in * E_in.conj()).real  # Already contiguous
arr = arr * np.float32(self.delta_X * self.delta_Y)  # Pre-cast to avoid upcast
```

**Fix 2: Pre-compute normalization constant**
```python
# In __init__:
self._norm_factor = np.float32(self.delta_X * self.delta_Y * c * epsilon_0 / 2)

# In _prepare_output_array:
arr = (E_in * E_in.conj()).real
integral = np.sum(arr, axis=self._last_axes) * self._norm_factor
E_00 = (self.power / integral) ** 0.5
```

**Impact:**
- Saves ~5ms (3% of total)
- Removes warning
- Cleaner code

**Recommendation:** Implement both fixes (low effort, removes warning).

---

### 5. Small Grid Overhead - 38x slower per element at 256×256 🔍

**Problem:** 256×256 takes 1912 ns/element vs 50 ns/element for larger grids.

**Analysis:**
```
Overhead-dominated regime: N < 512
Scaling regime: N ≥ 512
```

**Why:**
- Initialization costs (propagator, FFT plan) are fixed
- For small grids, initialization >> computation
- Not a bug, just physics of fixed costs

**Solutions:**
- None needed - users should use larger grids for performance
- Document minimum recommended size: **512×512**

**Recommendation:** Add warning for small grids in documentation.

---

## Optimization Priority Matrix

| Issue | Impact (1 propagation) | Impact (100 propagations) | Effort | Priority |
|-------|------------------------|---------------------------|--------|----------|
| GPU persistent mode | High (47% → 5%) | Medium | Medium | **HIGH** |
| Normalize optimization | Low (3%) | Low (3%) | Low | **HIGH** (removes warning) |
| Class-level propagator cache | Medium (58% → 0% for repeated calls) | Low | Low | **MEDIUM** |
| Batch interface | N/A | High (for param sweeps) | High | **MEDIUM** |
| FFT optimization | None possible | None possible | N/A | **LOW** (already optimal) |

---

## Revised Recommendations

### Quick Wins (< 1 hour)

1. **Fix normalization** (Priority: HIGH)
   - Use `(E_in * E_in.conj()).real`
   - Pre-compute `delta_X * delta_Y` as float32
   - Pre-compute norm_factor in `__init__`
   - **Impact:** Removes DoubleDowncastWarning, saves 5ms, cleaner code

2. **Document minimum grid size** (Priority: HIGH)
   - Add to docs: "For performance, use N ≥ 512"
   - Explain initialization overhead
   - **Impact:** Prevents user confusion about small-grid performance

### Medium Priority (2-4 hours)

3. **Implement GPU persistence mode** (Priority: HIGH for repeated use)
   - Add `enable_gpu_persistence()` method
   - Track `_arrays_on_gpu` state
   - Skip redundant transfers
   - **Impact:** 47% speedup for repeated calls on same instance

4. **Add class-level propagator caching** (Priority: MEDIUM)
   - Cache propagator by `(NX, NY, delta_z, precision)` key
   - Use `lru_cache` or manual dict
   - **Impact:** Eliminates 57.6% cost for repeated calls with same params

### Long-term (1+ days)

5. **Implement batch interface** (Priority: MEDIUM for sweeps)
   - `out_field_batch(E_in_batch, z, ...)` method
   - Process multiple fields simultaneously
   - Amortizes GPU transfer costs
   - **Impact:** Huge for parameter sweeps (10-100x speedup)

6. **Multi-GPU support** (Priority: LOW, research level)
   - Domain decomposition
   - Requires significant architecture changes
   - **Impact:** Scales to very large problems (8192×8192+)

---

## Benchmark Recommendations

### Add realistic benchmarks

```python
# tests/benchmarks/test_benchmark_profile.py

# Add grid size parametrization
@pytest.mark.parametrize("N", [512, 1024, 2048])
@pytest.mark.parametrize("backend", BACKENDS)
def test_nlse_2d_realistic(benchmark, backend, N):
    """Benchmark realistic grid sizes."""
    ...

# Add long propagation test
@pytest.mark.parametrize("backend", BACKENDS)
def test_nlse_2d_long_propagation(benchmark, backend):
    """Benchmark 100-step propagation (2048x2048)."""
    # Tests amortized cost structure
    ...

# Add batch test
@pytest.mark.parametrize("backend", ["CUPY", "CL"])
def test_nlse_2d_batch(benchmark, backend):
    """Benchmark batch processing (10 fields, 2048x2048)."""
    # Tests transfer amortization
    ...
```

---

## Conclusions

### Performance is Good

- CPU performance scales well O(N²) from 512×512 to 4096×4096
- OpenCL provides 1.54x speedup at realistic sizes
- FFT (90% of compute) is already optimal (FFTW/VkFFT)
- Kernels are fast (<2ms) - no need for fusion

### Main Opportunities

1. **GPU persistence** - Biggest impact for repeated use
2. **Normalization cleanup** - Removes warning, small speedup
3. **Batch processing** - Enables efficient parameter sweeps
4. **Propagator caching** - Helps repeated calls

### Non-Issues (Contrary to Initial Assessment)

- ❌ FFT wisdom I/O is NOT a bottleneck (1.5% of plan time)
- ❌ Kernel fusion is NOT needed (kernels are <2ms)
- ❌ GPU overhead at small grids was misleading (N=64 is pathological)
- ❌ Multiple kernel launches are fine (overhead is negligible)

### Critical Insight

**For production workloads (N=2048, multiple propagations), the main bottleneck is GPU data transfers, not computation.**

Implementing GPU persistence would provide ~2x speedup for typical workflows.
