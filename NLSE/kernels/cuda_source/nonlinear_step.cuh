// The real-space part of a split step for one element: interaction, potential
// and loss. Prepended to kernels.cu and to fft_callbacks.cu, so the kernels
// and the store callback that replaces them on CUPY run one expression, in one
// order, and fold the step into the inverse transform without changing it.

// The saturation the interaction should be evaluated at over a lossy step.
//
// A real-space step that freezes |A|^2 is exact only where |A|^2 is constant
// through it -- true of a pure rotation, false the moment there is loss, since
// the amplitude decays inside the step while the interaction goes on turning
// the phase at the rate the step began with. Frozen, the sub-step is O(dz^2)
// locally and every composition over it is first order: Lie, Strang and
// Yoshida alike, measured.
//
// With y = |A|^2, s(y) = 1/(1 + y/Isat) and u = 2*alpha*dz, two exact facts,
//
//     dy/dz = -2*alpha*s*y,  dphi/dz = g*y*s   =>   dphi/dy = -g/(2*alpha)
//
// give the phase over the step as (g/(2*alpha))*(y0 - y_end) whatever the
// saturation does in between, with y_end fixed by
// ln y + y/Isat = ln y0 + y0/Isat - u.
//
// Returned is P = (1 - y_end/y0)/u, that phase written as a saturation: the
// step applies g*y0*P*dz where it applied g*y0*sat*dz, and scales the
// amplitude by sqrt(1 - P*u) where it multiplied by exp(-alpha*sat*dz). Three
// passes put the composition back at fourth order, which is all Yoshida can
// use. The bracket is the series of (-log1p(-P*u) - P*u)/(P*u)^2, and it
// contracts while a step loses well under half the intensity -- which is what
// LOSS_PER_STEP_LIMIT in solvers/step_size.py keeps it to.
//
// u == 0 returns early, so a lossless run computes what it always did, to the
// bit, and pays one comparison for it.
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

// Largest |u| = |2*alpha*dz| the iteration above is used at, and why there is a
// limit: it contracts only while the step takes out a small fraction of the
// intensity, and the fraction it sees is sat*u, so the bound has to hold at
// sat = 1. Measured against a stiff solve there it beats the frozen step by
// 2.6x at u = 0.1 and loses to it by u = 0.3; near 1 it walks away and returns
// a larger field than it was given. Above this the kernels compute what they
// computed before -- see LOSS_PER_STEP_LIMIT in solvers/step_size.py, which
// caps a propagation at half of it, so only a direct kernel call gets here.
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
