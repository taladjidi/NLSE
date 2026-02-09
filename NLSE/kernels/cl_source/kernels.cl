// OpenCL C kernels for NLSE operations
// Precision-agnostic template using {{FP_TYPE}}, {{FP2_TYPE}}, {{FP_SUFFIX}}

// Nonlinear propagation with potential
__kernel void nl_prop_fused(
    __global {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq[idx] * sat + V[idx]) * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A_val = A[idx];
    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}

// Nonlinear propagation without potential
__kernel void nl_prop_without_v_fused(
    __global {{FP2_TYPE}}* A,
    __global const {{FP_TYPE}}* A_sq,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq[idx] / Isat);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = g * A_sq[idx] * sat * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A_val = A[idx];
    A[idx] = ({{FP2_TYPE}})(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}

// Coupled nonlinear propagation with potential
__kernel void nl_prop_c_fused(
    __global {{FP2_TYPE}}* A1,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    __global const {{FP_TYPE}}* V,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat + V[idx]) * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A1_val = A1[idx];
    A1[idx] = ({{FP2_TYPE}})(
        A1_val.x * exp_arg.x - A1_val.y * exp_arg.y,
        A1_val.x * exp_arg.y + A1_val.y * exp_arg.x
    );
}

// Coupled nonlinear propagation without potential
__kernel void nl_prop_c_without_v_fused(
    __global {{FP2_TYPE}}* A1,
    __global const {{FP_TYPE}}* A_sq_1,
    __global const {{FP_TYPE}}* A_sq_2,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g11,
    const {{FP_TYPE}} g12,
    const {{FP_TYPE}} Isat1,
    const {{FP_TYPE}} Isat2
) {
    int idx = get_global_id(0);
    {{FP_TYPE}} sat = 1.0{{FP_SUFFIX}} / (1.0{{FP_SUFFIX}} + A_sq_1[idx] / Isat1 + A_sq_2[idx] / Isat2);
    {{FP_TYPE}} arg_real = -alpha * sat * dz;
    {{FP_TYPE}} arg_imag = (g11 * A_sq_1[idx] * sat + g12 * A_sq_2[idx] * sat) * dz;
    {{FP_TYPE}} exp_real_part = exp(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    sin_imag = sincos(arg_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = ({{FP2_TYPE}})(exp_real_part * cos_imag, exp_real_part * sin_imag);
    {{FP2_TYPE}} A1_val = A1[idx];
    A1[idx] = ({{FP2_TYPE}})(
        A1_val.x * exp_arg.x - A1_val.y * exp_arg.y,
        A1_val.x * exp_arg.y + A1_val.y * exp_arg.x
    );
}

// Square modulus computation
__kernel void square_mod_fused(
    __global const {{FP2_TYPE}}* A,
    __global {{FP_TYPE}}* A_sq
) {
    int idx = get_global_id(0);
    {{FP2_TYPE}} A_val = A[idx];
    A_sq[idx] = A_val.x * A_val.x + A_val.y * A_val.y;
}
