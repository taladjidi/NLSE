// cuFFT store callbacks that fold the propagator multiply into the forward
// transform and the nonlinear step into the inverse one, the transforms whose
// output they consume. nonlinear_step.cuh is prepended to this file.
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

// The nonlinear step's arguments, read through callerInfo for the reason
// NlsePropagator is. The fields and the bound are rendered from
// NONLINEAR_FIELDS and INFO_WORDS in kernels/templating.py, whose
// nonlinear_words packs the words _WRITE_INFO writes from the same tuple.
struct NlseNonlinear {
    const void *V;
    {{NONLINEAR_FIELDS}}
};
static_assert(sizeof(NlseNonlinear) <= {{INFO_WORDS}} * sizeof(unsigned long long),
              "NlseNonlinear outgrew INFO_WORDS in kernels/templating.py");

// The nonlinear step as the inverse transform writes each element; the flat
// offset indexes the field and the potential alike. Called out of line:
// inlined, cuFFT's kernel spills its sixteen elements per thread to the stack
// and writes 36 bytes per point for 8 (docs/optimization-log.md, CUDA).
// {{VBLOCK}}
__device__ __noinline__ {{FP2_TYPE}} nlse_nonlinear_out_of_line(
    const {{FP2_TYPE}} element,
    V_ARG(V)
    const unsigned long long offset,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    {{FP_TYPE}} A_sq_val = element.x * element.x + element.y * element.y;
    return nlse_nonlinear_element(
        element, A_sq_val, V_PASS(V) offset, dz, alpha, g, Isat);
}

__device__ void nlse_store_nonlinear(void *dataOut, unsigned long long offset,
                                     {{FP2_TYPE}} element, void *callerInfo,
                                     void *sharedPointer) {
    const NlseNonlinear *info = (const NlseNonlinear *)callerInfo;
    (({{FP2_TYPE}} *)dataOut)[offset] = nlse_nonlinear_out_of_line(
        element, V_PASS(info->V) offset,
        info->dz, info->alpha, info->g, info->Isat);
}
// {{END_VBLOCK}}

// The propagator of a dispersion that is one term per axis is one factor per
// axis, rows[iy] * columns[ix], read from one value per row and per column
// rather than per element. rows carries the 1/N and repeats over a batch.
struct NlseSeparablePropagator {
    const {{FP2_TYPE}} *rows;
    const {{FP2_TYPE}} *columns;
    unsigned long long nx;
};

__device__ void nlse_store_propagator_separable(void *dataOut,
                                                unsigned long long offset,
                                                {{FP2_TYPE}} element,
                                                void *callerInfo,
                                                void *sharedPointer) {
    const NlseSeparablePropagator *info =
        (const NlseSeparablePropagator *)callerInfo;
    unsigned long long iy = offset / info->nx;
    unsigned long long ix = offset - iy * info->nx;
    {{FP2_TYPE}} a = info->rows[iy];
    {{FP2_TYPE}} b = info->columns[ix];
    {{FP2_TYPE}} p;
    p.x = a.x * b.x - a.y * b.y;
    p.y = a.x * b.y + a.y * b.x;
    {{FP2_TYPE}} out;
    out.x = element.x * p.x - element.y * p.y;
    out.y = element.x * p.y + element.y * p.x;
    (({{FP2_TYPE}} *)dataOut)[offset] = out;
}
