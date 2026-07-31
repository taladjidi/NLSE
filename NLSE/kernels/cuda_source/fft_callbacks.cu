// cuFFT store callbacks that fold the propagator multiply into the transform
// whose output it consumes.
//
// A split step touches the field four times: the transform pair, the
// propagator multiply, and the nonlinear step. cuFFT calls a store callback
// as it writes each element of a transform's output, so the multiply can
// happen there rather than in a pass of its own -- a pass that read the
// field, read the propagator and wrote the field back becomes one extra read
// of the propagator inside a write that was happening anyway.
//
// This file is compiled by nvrtc to LTO-IR and linked into the plan by cuFFT,
// which is why it cannot include <cufftXt.h>: nvrtc has no cuFFT headers on
// its include path. Nothing is lost by spelling the element type out --
// cufftComplex is float2 and cufftDoubleComplex is double2, so {{FP2_TYPE}}
// names it at either width.
//
// The arithmetic is deliberately the same expression as the apply_propagator
// kernel in kernels.cu, in the same order, so that folding the multiply in
// changes when it happens and not what it computes.

// callerInfo does not point at the propagator. It points at this block, which
// points at the propagator, and the indirection is the reason the plan is
// built once per run instead of once per step.
//
// cuFFT binds callerInfo when the plan is created, and no propagator lives
// that long: an adaptive step rebuilds it, Yoshida splitting cycles three of
// them, and the solver's cache hands back a different array each time the
// step length changes. Bound directly, every one of those would cost a new
// plan -- nvrtc, nvJitLink and cuFFT planning, tens of milliseconds -- and
// the fusion would lose more than it saves. Bound through the block, the
// pointer is rewritten by a kernel launch that a CUDA graph can record.
struct NlsePropagator {
    const {{FP2_TYPE}} *values;
    unsigned long long size;
};

// The field and the propagator have the same shape: the transform's flat
// output offset indexes both.
__device__ void nlse_store_propagator(void *dataOut, unsigned long long offset,
                                      {{FP2_TYPE}} element, void *callerInfo,
                                      void *sharedPointer) {
    const NlsePropagator *info = (const NlsePropagator *)callerInfo;
    {{FP2_TYPE}} p = info->values[offset];
    {{FP2_TYPE}} out;
    out.x = element.x * p.x - element.y * p.y;
    out.y = element.x * p.y + element.y * p.x;
    (({{FP2_TYPE}} *)dataOut)[offset] = out;
}

// A batch of fields sharing one propagator, which is how a parameter sweep
// propagates: the offset runs over the whole batch, the propagator only over
// one member of it.
//
// A separate function rather than a remainder in the one above, because the
// remainder is the identity in every non-batched case and a 64-bit remainder
// per element is not free.
__device__ void nlse_store_propagator_batched(void *dataOut,
                                              unsigned long long offset,
                                              {{FP2_TYPE}} element,
                                              void *callerInfo,
                                              void *sharedPointer) {
    const NlsePropagator *info = (const NlsePropagator *)callerInfo;
    {{FP2_TYPE}} p = info->values[offset % info->size];
    {{FP2_TYPE}} out;
    out.x = element.x * p.x - element.y * p.y;
    out.y = element.x * p.y + element.y * p.x;
    (({{FP2_TYPE}} *)dataOut)[offset] = out;
}
