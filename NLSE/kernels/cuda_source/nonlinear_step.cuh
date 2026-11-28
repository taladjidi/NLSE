// The real-space part of a split step for one element: interaction, potential
// and loss. Prepended to kernels.cu and to fft_callbacks.cu, so the kernels
// and the store callback that replaces them on CUPY run one expression, in one
// order, and fold the step into the inverse transform without changing it.

// The saturation P the interaction sees over a lossy step, u = 2*alpha*dz: the
// step applies g*y0*P*dz for the phase and sqrt(1 - P*u) for the amplitude.
// Three passes give fourth order; derivation in docs/numerical_methods.md.
// u == 0 returns early, so a lossless run is unchanged to the bit.
__device__ inline {{FP_TYPE}} nlse_loss_factor(
    const {{FP_TYPE}} sat,
    const {{FP_TYPE}} u
) {
    if (u == ({{FP_TYPE}})0.0) return sat;
    {{FP_TYPE}} P = sat;
    for (int pass = 0; pass < 3; ++pass) {
        {{FP_TYPE}} Pu = P * u;
        P = sat * (({{FP_TYPE}})1.0 - Pu * P * (({{FP_TYPE}})0.5
            + Pu * ({{FP_TYPE}})(1.0 / 3.0) + Pu * Pu * ({{FP_TYPE}})0.25));
    }
    return P;
}

// Largest |u| = |2*alpha*dz| the iteration is used at: past it the iteration
// stops contracting and the kernels use the frozen step. A propagation stays
// under a quarter of it (LOSS_PER_STEP_LIMIT in solvers/step_size.py).
#define NLSE_LOSS_SOLVED_LIMIT 0.1

// One element of the step. A_sq_val is |A_val|^2, or the non-locally smoothed
// intensity nl_prop is given. The three twins are overloads, told apart by V.
// {{VBLOCK}}
__device__ inline {{FP2_TYPE}} nlse_nonlinear_element(
    const {{FP2_TYPE}} A_val,
    const {{FP_TYPE}} A_sq_val,
    V_ARG(V)
    const unsigned long long idx,
    const {{FP_TYPE}} dz,
    const {{FP_TYPE}} alpha,
    const {{FP_TYPE}} g,
    const {{FP_TYPE}} Isat
) {
    // The loss has left arg_real for exp_real_part, where the step is solved
    // rather than frozen. What remains there is a complex potential's own
    // absorption, which nlse_loss_factor does not cover.
    {{FP_TYPE}} u = ({{FP_TYPE}})2.0 * alpha * dz;
    {{FP_TYPE}} sat = ({{FP_TYPE}})1.0 / (({{FP_TYPE}})1.0 + A_sq_val / Isat);
    // Solved, or frozen as before where solving it is out of reach. Both
    // branches are uniform across the grid: u is a scalar.
    bool solved = u <= ({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT
        && u >= -({{FP_TYPE}})NLSE_LOSS_SOLVED_LIMIT;
    {{FP_TYPE}} P = solved ? nlse_loss_factor(sat, u) : sat;
    {{FP_TYPE}} alpha_left = solved ? ({{FP_TYPE}})0.0 : alpha;
    {{FP_TYPE}} amp = solved ? sqrt{{FP_SUFFIX}}(({{FP_TYPE}})1.0 - P * u) : ({{FP_TYPE}})1.0;
    {{FP_TYPE}} arg_real = -(alpha_left * sat V_LOSS(V, idx)) * dz;
    {{FP_TYPE}} arg_imag = (g * A_sq_val * P V_PHASE(V, idx)) * dz;
    {{FP_TYPE}} exp_real_part = amp * exp{{FP_SUFFIX}}(arg_real);
    {{FP_TYPE}} cos_imag, sin_imag;
    {{SINCOS_FUNC}}(arg_imag, &sin_imag, &cos_imag);
    {{FP2_TYPE}} exp_arg = make_{{FP2_TYPE}}(exp_real_part * cos_imag, exp_real_part * sin_imag);

    return make_{{FP2_TYPE}}(
        A_val.x * exp_arg.x - A_val.y * exp_arg.y,
        A_val.x * exp_arg.y + A_val.y * exp_arg.x
    );
}
// {{END_VBLOCK}}
