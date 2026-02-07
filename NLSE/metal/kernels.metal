#include <metal_stdlib>
using namespace metal;

// complex64 is represented as float2: (real, imag)
typedef float2 cfloat;

// Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
inline cfloat cmul(cfloat a, cfloat b) {
    return cfloat(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex exponential: exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
inline cfloat cexp(cfloat z) {
    float r = exp(z.x);
    return cfloat(r * cos(z.y), r * sin(z.y));
}

// |z|^2 = re^2 + im^2
inline float cabs2(cfloat z) {
    return z.x * z.x + z.y * z.y;
}


// ============================================================
// square_mod: A_sq[i] = |A[i]|^2
// ============================================================
kernel void square_mod(
    device const cfloat* A [[buffer(0)]],
    device float* A_sq [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    A_sq[id] = cabs2(A[id]);
}


// ============================================================
// nl_prop: nonlinear propagation with potential
// CPU convention: arg = -alpha*sat + 1j*g*A_sq*sat + 1j*V
// so 1j*V = -V.imag + 1j*V.real
// ============================================================
kernel void nl_prop(
    device cfloat* A [[buffer(0)]],
    device const float* A_sq [[buffer(1)]],
    device const cfloat* V [[buffer(2)]],
    constant float& dz [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant float& g [[buffer(5)]],
    constant float& Isat [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    float sat = 1.0f / (1.0f + A_sq[id] / Isat);
    // 1j * V = (-V.imag, +V.real)
    cfloat arg = cfloat(-alpha * sat - V[id].y,
                        g * A_sq[id] * sat + V[id].x);
    A[id] = cmul(A[id], cexp(dz * arg));
}


// ============================================================
// nl_prop_without_V: nonlinear propagation without potential
// ============================================================
kernel void nl_prop_without_V(
    device cfloat* A [[buffer(0)]],
    device const float* A_sq [[buffer(1)]],
    constant float& dz [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant float& g [[buffer(4)]],
    constant float& Isat [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    float sat = 1.0f / (1.0f + A_sq[id] / Isat);
    cfloat arg = cfloat(-alpha * sat, g * A_sq[id] * sat);
    A[id] = cmul(A[id], cexp(dz * arg));
}


// ============================================================
// nl_prop_c: coupled nonlinear propagation with potential
// ============================================================
kernel void nl_prop_c(
    device cfloat* A1 [[buffer(0)]],
    device const float* A_sq_1 [[buffer(1)]],
    device const float* A_sq_2 [[buffer(2)]],
    device const cfloat* V [[buffer(3)]],
    constant float& dz [[buffer(4)]],
    constant float& alpha [[buffer(5)]],
    constant float& g11 [[buffer(6)]],
    constant float& g12 [[buffer(7)]],
    constant float& Isat1 [[buffer(8)]],
    constant float& Isat2 [[buffer(9)]],
    uint id [[thread_position_in_grid]])
{
    float sat = 1.0f / (1.0f + A_sq_1[id] / Isat1 + A_sq_2[id] / Isat2);
    // 1j * V = (-V.imag, +V.real)
    float re = -alpha * sat - V[id].y;
    float im = (g11 * A_sq_1[id] * sat + g12 * A_sq_2[id] * sat) + V[id].x;
    cfloat arg = cfloat(re, im);
    A1[id] = cmul(A1[id], cexp(dz * arg));
}


// ============================================================
// nl_prop_without_V_c: coupled nonlinear propagation, no potential
// ============================================================
kernel void nl_prop_without_V_c(
    device cfloat* A1 [[buffer(0)]],
    device const float* A_sq_1 [[buffer(1)]],
    device const float* A_sq_2 [[buffer(2)]],
    constant float& dz [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant float& g11 [[buffer(5)]],
    constant float& g12 [[buffer(6)]],
    constant float& Isat1 [[buffer(7)]],
    constant float& Isat2 [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    float sat = 1.0f / (1.0f + A_sq_1[id] / Isat1 + A_sq_2[id] / Isat2);
    float re = -alpha * sat;
    float im = g11 * A_sq_1[id] * sat + g12 * A_sq_2[id] * sat;
    cfloat arg = cfloat(re, im);
    A1[id] = cmul(A1[id], cexp(dz * arg));
}


// ============================================================
// rabi_coupling: Rabi hopping between two components
// needs two passes since A1_old is needed for A2 update
// ============================================================
kernel void rabi_coupling_A1(
    device cfloat* A1 [[buffer(0)]],
    device const cfloat* A2 [[buffer(1)]],
    device cfloat* A1_old [[buffer(2)]],
    constant float& cos_val [[buffer(3)]],
    constant float& sin_val [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    // Save A1 to A1_old, then update A1
    A1_old[id] = A1[id];
    cfloat a1 = A1[id];
    cfloat a2 = A2[id];
    // A1 = cos * A1 - i*sin * A2
    // -i * sin * A2 = cfloat(sin * A2.y, -sin * A2.x)
    A1[id] = cfloat(
        cos_val * a1.x + sin_val * a2.y,
        cos_val * a1.y - sin_val * a2.x
    );
}

kernel void rabi_coupling_A2(
    device cfloat* A2 [[buffer(0)]],
    device const cfloat* A1_old [[buffer(1)]],
    constant float& cos_val [[buffer(2)]],
    constant float& sin_val [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    cfloat a2 = A2[id];
    cfloat a1_old = A1_old[id];
    // A2 = cos * A2 - i*sin * A1_old
    A2[id] = cfloat(
        cos_val * a2.x + sin_val * a1_old.y,
        cos_val * a2.y - sin_val * a1_old.x
    );
}


// ============================================================
// vortex: generate vortex phase pattern
// Only supports charge 1 and -1 for now (most common)
// ============================================================
kernel void vortex(
    device float* im [[buffer(0)]],
    device const float* ii [[buffer(1)]],
    device const float* jj [[buffer(2)]],
    constant float& i_pos [[buffer(3)]],
    constant float& j_pos [[buffer(4)]],
    constant int& ll [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    float dy = ii[id] - i_pos;
    float dx = jj[id] - j_pos;
    // For charge 1: angle = atan2(dx, dy)
    // For higher charges: angle = ll * atan2(dx, dy) (not exact but common approx)
    im[id] += float(ll) * atan2(dx, dy);
}


// ============================================================
// element-wise complex multiply (for propagator application)
// ============================================================
kernel void complex_multiply_inplace(
    device cfloat* A [[buffer(0)]],
    device const cfloat* B [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    A[id] = cmul(A[id], B[id]);
}
