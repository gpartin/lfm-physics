"""
CUDA Kernel Source Strings
==========================

Production CUDA kernels for LFM leapfrog evolution.
These are the exact kernels from the canonical universe simulator,
extracted verbatim for use by the CuPy backend.

Three kernels:
- EVOLUTION_KERNEL_SRC: Full GOV-01 + GOV-02 (3-color complex Ψₐ)
- PHASE1_KERNEL_SRC: Parametric resonance with oscillating χ
- EVOLUTION_REAL_KERNEL_SRC: Simplified real-E gravity-only kernel
"""

# ---------------------------------------------------------------------------
# Full 3-color complex evolution kernel (Level 2 — all four forces)
# ---------------------------------------------------------------------------
EVOLUTION_KERNEL_SRC = r"""
extern "C" __global__ __launch_bounds__(256)
void evolve_gov01_gov02(
    // Input arrays -- 3-color complex Psi_a, packed [3*N^3]
    const float* __restrict__ Psi_r,
    const float* __restrict__ Psi_r_prev,
    const float* __restrict__ Psi_i,
    const float* __restrict__ Psi_i_prev,
    const float* __restrict__ chi,
    const float* __restrict__ chi_prev,
    const float* __restrict__ boundary_mask,
    // Output arrays
    float* __restrict__ Psi_r_next,
    float* __restrict__ Psi_r_prev_next,
    float* __restrict__ Psi_i_next,
    float* __restrict__ Psi_i_prev_next,
    float* __restrict__ chi_next,
    float* __restrict__ chi_prev_next,
    // Parameters
    const int N,
    const float dt2,
    const float kappa,
    const float lam,
    const float chi0,
    const float E0_sq,
    const float eps_w,
    const float kappa_c,
    const float eps_cc,
    const float* __restrict__ Sa_in,
    const float kappa_string,
    const float kappa_tube)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N * N * N;
    if (idx >= total) return;

    int i = idx / (N * N);
    int j = (idx / N) % N;
    int k = idx % N;

    // 19-point stencil indices (faces + edges)
    int row_p = ((i + 1) % N) * N * N;
    int row_m = ((i - 1 + N) % N) * N * N;
    int row_c = i * N * N;
    int col_p = ((j + 1) % N) * N;
    int col_m = ((j - 1 + N) % N) * N;
    int col_c = j * N;
    int dep_p = (k + 1) % N;
    int dep_m = (k - 1 + N) % N;
    int ip = row_p + col_c + k;
    int im = row_m + col_c + k;
    int jp = row_c + col_p + k;
    int jm = row_c + col_m + k;
    int kp = row_c + col_c + dep_p;
    int km = row_c + col_c + dep_m;
    int ipjp = row_p + col_p + k;
    int ipjm = row_p + col_m + k;
    int imjp = row_m + col_p + k;
    int imjm = row_m + col_m + k;
    int ipkp = row_p + col_c + dep_p;
    int ipkm = row_p + col_c + dep_m;
    int imkp = row_m + col_c + dep_p;
    int imkm = row_m + col_c + dep_m;
    int jpkp = row_c + col_p + dep_p;
    int jpkm = row_c + col_p + dep_m;
    int jmkp = row_c + col_m + dep_p;
    int jmkm = row_c + col_m + dep_m;

    float chi_c = chi[idx];
    float chi_sq = chi_c * chi_c;
    float mask = boundary_mask[idx];
    float absorb = 1.0f - mask;  // 1 inside, 0 at boundary (smooth taper)

    // Accumulate colorblind sources for GOV-02
    float psi_sq_total = 0.0f;
    float j_total = 0.0f;

    // Per-color energy densities for f_c (v14)
    float ea[3];

    // Per-color per-direction currents for CCV (v15 GOV-02)
    float j_color_x[3] = {0.0f, 0.0f, 0.0f};
    float j_color_y[3] = {0.0f, 0.0f, 0.0f};
    float j_color_z[3] = {0.0f, 0.0f, 0.0f};

    // v15: compute color average Psi_bar for cross-color coupling
    float Pr_avg = 0.0f;
    float Pi_avg = 0.0f;
    if (eps_cc > 0.0f) {
        for (int a = 0; a < 3; a++) {
            int off = a * total;
            Pr_avg += Psi_r[off + idx];
            Pi_avg += Psi_i[off + idx];
        }
        Pr_avg *= (1.0f / 3.0f);
        Pi_avg *= (1.0f / 3.0f);
    }

    // Loop over 3 color components (a=0,1,2)
    #pragma unroll
    for (int a = 0; a < 3; a++) {
        int off = a * total;
        int aidx = off + idx;

        float Pr = Psi_r[aidx];
        float Pi_val = Psi_i[aidx];

        // 19-point Laplacian: w_face=1/3, w_edge=1/6, center=-4
        float lap_Pr = (1.0f/3.0f) * (Psi_r[off+ip] + Psi_r[off+im] + Psi_r[off+jp] + Psi_r[off+jm] + Psi_r[off+kp] + Psi_r[off+km])
                     + (1.0f/6.0f) * (Psi_r[off+ipjp] + Psi_r[off+ipjm] + Psi_r[off+imjp] + Psi_r[off+imjm]
                                     + Psi_r[off+ipkp] + Psi_r[off+ipkm] + Psi_r[off+imkp] + Psi_r[off+imkm]
                                     + Psi_r[off+jpkp] + Psi_r[off+jpkm] + Psi_r[off+jmkp] + Psi_r[off+jmkm])
                     - 4.0f * Pr;

        float lap_Pi = (1.0f/3.0f) * (Psi_i[off+ip] + Psi_i[off+im] + Psi_i[off+jp] + Psi_i[off+jm] + Psi_i[off+kp] + Psi_i[off+km])
                     + (1.0f/6.0f) * (Psi_i[off+ipjp] + Psi_i[off+ipjm] + Psi_i[off+imjp] + Psi_i[off+imjm]
                                     + Psi_i[off+ipkp] + Psi_i[off+ipkm] + Psi_i[off+imkp] + Psi_i[off+imkm]
                                     + Psi_i[off+jpkp] + Psi_i[off+jpkm] + Psi_i[off+jmkp] + Psi_i[off+jmkm])
                     - 4.0f * Pi_val;

        // GOV-01 leapfrog
        float Pr_new = 2.0f * Pr - Psi_r_prev[aidx] + dt2 * (lap_Pr - chi_sq * Pr);
        float Pi_new = 2.0f * Pi_val - Psi_i_prev[aidx] + dt2 * (lap_Pi - chi_sq * Pi_val);

        // v15: cross-color coupling -eps_cc * chi^2 * (Psi_a - Psi_bar)
        if (eps_cc > 0.0f) {
            Pr_new -= dt2 * eps_cc * chi_sq * (Pr - Pr_avg);
            Pi_new -= dt2 * eps_cc * chi_sq * (Pi_val - Pi_avg);
        }

        // Absorbing boundary — damp both new and prev to prevent leapfrog reflection.
        Psi_r_next[aidx] = Pr_new * absorb;
        Psi_r_prev_next[aidx] = Pr * absorb;      // damp prev too
        Psi_i_next[aidx] = Pi_new * absorb;
        Psi_i_prev_next[aidx] = Pi_val * absorb;  // damp prev too

        // Per-color energy density
        float e_a = Pr * Pr + Pi_val * Pi_val;
        ea[a] = e_a;

        // Colorblind energy density: Sum_a |Psi_a|^2
        psi_sq_total += e_a;

        // Momentum density: Sum_a Im(Psi_a* . nabla(Psi_a))
        float j_x = Pr * (Psi_i[off+ip] - Psi_i[off+im]) - Pi_val * (Psi_r[off+ip] - Psi_r[off+im]);
        float j_y = Pr * (Psi_i[off+jp] - Psi_i[off+jm]) - Pi_val * (Psi_r[off+jp] - Psi_r[off+jm]);
        float j_z = Pr * (Psi_i[off+kp] - Psi_i[off+km]) - Pi_val * (Psi_r[off+kp] - Psi_r[off+km]);
        j_total += 0.5f * (j_x + j_y + j_z);
        // Store per-color currents for CCV
        j_color_x[a] = j_x;
        j_color_y[a] = j_y;
        j_color_z[a] = j_z;
    }  // end color loop
    // CCV = Sum_d [ Sum_a j_{a,d}^2 - (1/3)(Sum_a j_{a,d})^2 ]
    float ccv = 0.0f;
    if (kappa_string > 0.0f) {
        float jx_sum = j_color_x[0] + j_color_x[1] + j_color_x[2];
        float jy_sum = j_color_y[0] + j_color_y[1] + j_color_y[2];
        float jz_sum = j_color_z[0] + j_color_z[1] + j_color_z[2];
        float sum_jxsq = j_color_x[0]*j_color_x[0] + j_color_x[1]*j_color_x[1] + j_color_x[2]*j_color_x[2];
        float sum_jysq = j_color_y[0]*j_color_y[0] + j_color_y[1]*j_color_y[1] + j_color_y[2]*j_color_y[2];
        float sum_jzsq = j_color_z[0]*j_color_z[0] + j_color_z[1]*j_color_z[1] + j_color_z[2]*j_color_z[2];
        ccv = (sum_jxsq - (1.0f/3.0f)*jx_sum*jx_sum)
            + (sum_jysq - (1.0f/3.0f)*jy_sum*jy_sum)
            + (sum_jzsq - (1.0f/3.0f)*jz_sum*jz_sum);
    }

    // v16: smoothed color variance (SCV) from S_a auxiliary fields
    // SCV = Sum_a S_a^2 - (1/3)(Sum_a S_a)^2
    float scv = 0.0f;
    if (kappa_tube > 0.0f) {
        float sa_sum = 0.0f;
        float sa_sq_sum = 0.0f;
        for (int a = 0; a < 3; a++) {
            float sa = Sa_in[a * total + idx];
            sa_sum += sa;
            sa_sq_sum += sa * sa;
        }
        scv = sa_sq_sum - (1.0f/3.0f) * sa_sum * sa_sum;
    }

    // v14: normalized color variance f_c = [Sum_a |Psi_a|^4 / (Sum_a |Psi_a|^2)^2] - 1/3
    float color_var_term = 0.0f;
    if (kappa_c > 0.0f) {
        float total_sq = psi_sq_total * psi_sq_total;
        if (total_sq > 1e-30f) {
            float sum_sq = ea[0]*ea[0] + ea[1]*ea[1] + ea[2]*ea[2];
            float f_c = sum_sq / total_sq - (1.0f / 3.0f);
            color_var_term = kappa_c * f_c * psi_sq_total;
        }
    }

    // 19-point Laplacian for chi
    float lap_chi = (1.0f/3.0f) * (chi[ip] + chi[im] + chi[jp] + chi[jm] + chi[kp] + chi[km])
                  + (1.0f/6.0f) * (chi[ipjp] + chi[ipjm] + chi[imjp] + chi[imjm]
                                  + chi[ipkp] + chi[ipkm] + chi[imkp] + chi[imkm]
                                  + chi[jpkp] + chi[jpkm] + chi[jmkp] + chi[jmkm])
                  - 4.0f * chi_c;

    // Mexican hat: -4*lam*chi*(chi^2 - chi0^2)
    float chi_self = -4.0f * lam * chi_c * (chi_sq - chi0 * chi0);

    // GOV-02 v28.0: colorblind gravity + color variance + CCV + SCV
    float chi_new = 2.0f * chi_c - chi_prev[idx] + dt2 * (
        lap_chi - (kappa / chi0) * chi_c * (psi_sq_total + eps_w * j_total - E0_sq)
        - color_var_term + chi_self
        - kappa_string * ccv - kappa_tube * scv);

    // BH excision: clamp to Z2 second vacuum
    if (chi_new < -chi0) chi_new = -chi0;

    // Frozen boundary
    chi_new = mask * chi0 + absorb * chi_new;

    chi_next[idx] = chi_new;
    chi_prev_next[idx] = chi_c;
}
"""

# ---------------------------------------------------------------------------
# Parametric resonance kernel (Phase 1 — matter creation via Mathieu eq.)
# ---------------------------------------------------------------------------
PHASE1_KERNEL_SRC = r"""
extern "C" __global__ __launch_bounds__(256)
void phase1_parametric(
    const float* __restrict__ Psi_r,
    const float* __restrict__ Psi_r_prev,
    const float* __restrict__ Psi_i,
    const float* __restrict__ Psi_i_prev,
    float* __restrict__ Psi_r_next,
    float* __restrict__ Psi_r_prev_next,
    float* __restrict__ Psi_i_next,
    float* __restrict__ Psi_i_prev_next,
    const int N,
    const float dt2,
    const float chi_sq)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N * N * N;
    if (idx >= total) return;

    int i = idx / (N * N);
    int j = (idx / N) % N;
    int k = idx % N;

    int row_p = ((i + 1) % N) * N * N;
    int row_m = ((i - 1 + N) % N) * N * N;
    int row_c = i * N * N;
    int col_p = ((j + 1) % N) * N;
    int col_m = ((j - 1 + N) % N) * N;
    int col_c = j * N;
    int dep_p = (k + 1) % N;
    int dep_m = (k - 1 + N) % N;
    int ip = row_p + col_c + k;
    int im = row_m + col_c + k;
    int jp = row_c + col_p + k;
    int jm = row_c + col_m + k;
    int kp = row_c + col_c + dep_p;
    int km = row_c + col_c + dep_m;
    int ipjp = row_p + col_p + k;
    int ipjm = row_p + col_m + k;
    int imjp = row_m + col_p + k;
    int imjm = row_m + col_m + k;
    int ipkp = row_p + col_c + dep_p;
    int ipkm = row_p + col_c + dep_m;
    int imkp = row_m + col_c + dep_p;
    int imkm = row_m + col_c + dep_m;
    int jpkp = row_c + col_p + dep_p;
    int jpkm = row_c + col_p + dep_m;
    int jmkp = row_c + col_m + dep_p;
    int jmkm = row_c + col_m + dep_m;

    #pragma unroll
    for (int a = 0; a < 3; a++) {
        int off = a * total;
        int aidx = off + idx;

        float Pr = Psi_r[aidx];
        float Pi_val = Psi_i[aidx];

        float lap_Pr = (1.0f/3.0f) * (Psi_r[off+ip] + Psi_r[off+im] + Psi_r[off+jp] + Psi_r[off+jm] + Psi_r[off+kp] + Psi_r[off+km])
                     + (1.0f/6.0f) * (Psi_r[off+ipjp] + Psi_r[off+ipjm] + Psi_r[off+imjp] + Psi_r[off+imjm]
                                     + Psi_r[off+ipkp] + Psi_r[off+ipkm] + Psi_r[off+imkp] + Psi_r[off+imkm]
                                     + Psi_r[off+jpkp] + Psi_r[off+jpkm] + Psi_r[off+jmkp] + Psi_r[off+jmkm])
                     - 4.0f * Pr;
        float lap_Pi = (1.0f/3.0f) * (Psi_i[off+ip] + Psi_i[off+im] + Psi_i[off+jp] + Psi_i[off+jm] + Psi_i[off+kp] + Psi_i[off+km])
                     + (1.0f/6.0f) * (Psi_i[off+ipjp] + Psi_i[off+ipjm] + Psi_i[off+imjp] + Psi_i[off+imjm]
                                     + Psi_i[off+ipkp] + Psi_i[off+ipkm] + Psi_i[off+imkp] + Psi_i[off+imkm]
                                     + Psi_i[off+jpkp] + Psi_i[off+jpkm] + Psi_i[off+jmkp] + Psi_i[off+jmkm])
                     - 4.0f * Pi_val;

        float Pr_new = 2.0f * Pr - Psi_r_prev[aidx] + dt2 * (lap_Pr - chi_sq * Pr);
        float Pi_new = 2.0f * Pi_val - Psi_i_prev[aidx] + dt2 * (lap_Pi - chi_sq * Pi_val);

        Psi_r_next[aidx] = Pr_new;
        Psi_r_prev_next[aidx] = Pr;
        Psi_i_next[aidx] = Pi_new;
        Psi_i_prev_next[aidx] = Pi_val;
    }
}
"""

# ---------------------------------------------------------------------------
# Real-E gravity-only kernel (Level 0 — cosmology, structure formation)
# ---------------------------------------------------------------------------
EVOLUTION_REAL_KERNEL_SRC = r"""
extern "C" __global__ __launch_bounds__(256)
void evolve_real(
    const float* __restrict__ E,
    const float* __restrict__ E_prev,
    const float* __restrict__ chi,
    const float* __restrict__ chi_prev,
    const float* __restrict__ boundary_mask,
    float* __restrict__ E_next,
    float* __restrict__ E_prev_next,
    float* __restrict__ chi_next,
    float* __restrict__ chi_prev_next,
    const int N,
    const float dt2,
    const float kappa,
    const float lam,
    const float chi0,
    const float E0_sq)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N * N * N;
    if (idx >= total) return;

    int i = idx / (N * N);
    int j = (idx / N) % N;
    int k = idx % N;

    int row_p = ((i + 1) % N) * N * N;
    int row_m = ((i - 1 + N) % N) * N * N;
    int row_c = i * N * N;
    int col_p = ((j + 1) % N) * N;
    int col_m = ((j - 1 + N) % N) * N;
    int col_c = j * N;
    int dep_p = (k + 1) % N;
    int dep_m = (k - 1 + N) % N;
    int ip = row_p + col_c + k;
    int im = row_m + col_c + k;
    int jp = row_c + col_p + k;
    int jm = row_c + col_m + k;
    int kp = row_c + col_c + dep_p;
    int km = row_c + col_c + dep_m;
    int ipjp = row_p + col_p + k;
    int ipjm = row_p + col_m + k;
    int imjp = row_m + col_p + k;
    int imjm = row_m + col_m + k;
    int ipkp = row_p + col_c + dep_p;
    int ipkm = row_p + col_c + dep_m;
    int imkp = row_m + col_c + dep_p;
    int imkm = row_m + col_c + dep_m;
    int jpkp = row_c + col_p + dep_p;
    int jpkm = row_c + col_p + dep_m;
    int jmkp = row_c + col_m + dep_p;
    int jmkm = row_c + col_m + dep_m;

    float E_c = E[idx];
    float chi_c = chi[idx];
    float chi_sq = chi_c * chi_c;

    // 19-point Laplacian for E
    float lap_E = (1.0f/3.0f) * (E[ip] + E[im] + E[jp] + E[jm] + E[kp] + E[km])
                + (1.0f/6.0f) * (E[ipjp] + E[ipjm] + E[imjp] + E[imjm]
                                + E[ipkp] + E[ipkm] + E[imkp] + E[imkm]
                                + E[jpkp] + E[jpkm] + E[jmkp] + E[jmkm])
                - 4.0f * E_c;

    // 19-point Laplacian for chi
    float lap_chi = (1.0f/3.0f) * (chi[ip] + chi[im] + chi[jp] + chi[jm] + chi[kp] + chi[km])
                  + (1.0f/6.0f) * (chi[ipjp] + chi[ipjm] + chi[imjp] + chi[imjm]
                                  + chi[ipkp] + chi[ipkm] + chi[imkp] + chi[imkm]
                                  + chi[jpkp] + chi[jpkm] + chi[jmkp] + chi[jmkm])
                  - 4.0f * chi_c;

    // GOV-01
    float E_new = 2.0f * E_c - E_prev[idx] + dt2 * (lap_E - chi_sq * E_c);

    // Mexican hat self-interaction
    float chi_self = -4.0f * lam * chi_c * (chi_sq - chi0 * chi0);

    // GOV-02 v28.0
    float chi_new = 2.0f * chi_c - chi_prev[idx] + dt2 * (
        lap_chi - (kappa / chi0) * chi_c * (E_c * E_c - E0_sq) + chi_self);

    // BH excision
    if (chi_new < -chi0) chi_new = -chi0;

    // Absorbing boundary — damp both new and prev to prevent leapfrog reflection.
    float mask = boundary_mask[idx];
    float absorb = 1.0f - mask;
    E_new = absorb * E_new;
    chi_new = mask * chi0 + absorb * chi_new;

    E_next[idx] = E_new;
    E_prev_next[idx] = absorb * E_c;   // damp prev too
    chi_next[idx] = chi_new;
    chi_prev_next[idx] = chi_c;
}
"""

# ---------------------------------------------------------------------------
# Complex single-component kernel (Level 1 — gravity + EM)
# ---------------------------------------------------------------------------
EVOLUTION_COMPLEX_KERNEL_SRC = r"""
extern "C" __global__ __launch_bounds__(256)
void evolve_complex(
    const float* __restrict__ Psi_r,
    const float* __restrict__ Psi_r_prev,
    const float* __restrict__ Psi_i,
    const float* __restrict__ Psi_i_prev,
    const float* __restrict__ chi,
    const float* __restrict__ chi_prev,
    const float* __restrict__ boundary_mask,
    float* __restrict__ Psi_r_next,
    float* __restrict__ Psi_r_prev_next,
    float* __restrict__ Psi_i_next,
    float* __restrict__ Psi_i_prev_next,
    float* __restrict__ chi_next,
    float* __restrict__ chi_prev_next,
    const int N,
    const float dt2,
    const float kappa,
    const float lam,
    const float chi0,
    const float E0_sq,
    const float eps_w)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N * N * N;
    if (idx >= total) return;

    int i = idx / (N * N);
    int j = (idx / N) % N;
    int k = idx % N;

    int row_p = ((i + 1) % N) * N * N;
    int row_m = ((i - 1 + N) % N) * N * N;
    int row_c = i * N * N;
    int col_p = ((j + 1) % N) * N;
    int col_m = ((j - 1 + N) % N) * N;
    int col_c = j * N;
    int dep_p = (k + 1) % N;
    int dep_m = (k - 1 + N) % N;
    int ip = row_p + col_c + k;
    int im = row_m + col_c + k;
    int jp = row_c + col_p + k;
    int jm = row_c + col_m + k;
    int kp = row_c + col_c + dep_p;
    int km = row_c + col_c + dep_m;
    int ipjp = row_p + col_p + k;
    int ipjm = row_p + col_m + k;
    int imjp = row_m + col_p + k;
    int imjm = row_m + col_m + k;
    int ipkp = row_p + col_c + dep_p;
    int ipkm = row_p + col_c + dep_m;
    int imkp = row_m + col_c + dep_p;
    int imkm = row_m + col_c + dep_m;
    int jpkp = row_c + col_p + dep_p;
    int jpkm = row_c + col_p + dep_m;
    int jmkp = row_c + col_m + dep_p;
    int jmkm = row_c + col_m + dep_m;

    float Pr = Psi_r[idx];
    float Pi_val = Psi_i[idx];
    float chi_c = chi[idx];
    float chi_sq = chi_c * chi_c;

    // 19-point Laplacians
    float lap_Pr = (1.0f/3.0f) * (Psi_r[ip] + Psi_r[im] + Psi_r[jp] + Psi_r[jm] + Psi_r[kp] + Psi_r[km])
                 + (1.0f/6.0f) * (Psi_r[ipjp] + Psi_r[ipjm] + Psi_r[imjp] + Psi_r[imjm]
                                 + Psi_r[ipkp] + Psi_r[ipkm] + Psi_r[imkp] + Psi_r[imkm]
                                 + Psi_r[jpkp] + Psi_r[jpkm] + Psi_r[jmkp] + Psi_r[jmkm])
                 - 4.0f * Pr;
    float lap_Pi = (1.0f/3.0f) * (Psi_i[ip] + Psi_i[im] + Psi_i[jp] + Psi_i[jm] + Psi_i[kp] + Psi_i[km])
                 + (1.0f/6.0f) * (Psi_i[ipjp] + Psi_i[ipjm] + Psi_i[imjp] + Psi_i[imjm]
                                 + Psi_i[ipkp] + Psi_i[ipkm] + Psi_i[imkp] + Psi_i[imkm]
                                 + Psi_i[jpkp] + Psi_i[jpkm] + Psi_i[jmkp] + Psi_i[jmkm])
                 - 4.0f * Pi_val;
    float lap_chi = (1.0f/3.0f) * (chi[ip] + chi[im] + chi[jp] + chi[jm] + chi[kp] + chi[km])
                  + (1.0f/6.0f) * (chi[ipjp] + chi[ipjm] + chi[imjp] + chi[imjm]
                                  + chi[ipkp] + chi[ipkm] + chi[imkp] + chi[imkm]
                                  + chi[jpkp] + chi[jpkm] + chi[jmkp] + chi[jmkm])
                  - 4.0f * chi_c;

    // GOV-01
    float Pr_new = 2.0f * Pr - Psi_r_prev[idx] + dt2 * (lap_Pr - chi_sq * Pr);
    float Pi_new = 2.0f * Pi_val - Psi_i_prev[idx] + dt2 * (lap_Pi - chi_sq * Pi_val);

    // |Psi|^2 and momentum density j
    float psi_sq = Pr * Pr + Pi_val * Pi_val;
    float j_x = Pr * (Psi_i[ip] - Psi_i[im]) - Pi_val * (Psi_r[ip] - Psi_r[im]);
    float j_y = Pr * (Psi_i[jp] - Psi_i[jm]) - Pi_val * (Psi_r[jp] - Psi_r[jm]);
    float j_z = Pr * (Psi_i[kp] - Psi_i[km]) - Pi_val * (Psi_r[kp] - Psi_r[km]);
    float j_scalar = 0.5f * (j_x + j_y + j_z);

    // Mexican hat
    float chi_self = -4.0f * lam * chi_c * (chi_sq - chi0 * chi0);

    // GOV-02
    float chi_new = 2.0f * chi_c - chi_prev[idx] + dt2 * (
        lap_chi - (kappa / chi0) * chi_c * (psi_sq + eps_w * j_scalar - E0_sq) + chi_self);

    // BH excision
    if (chi_new < -chi0) chi_new = -chi0;

    // Absorbing boundary — damp both new and prev to prevent leapfrog reflection.
    float mask = boundary_mask[idx];
    float absorb = 1.0f - mask;
    Pr_new = absorb * Pr_new;
    Pi_new = absorb * Pi_new;
    chi_new = mask * chi0 + absorb * chi_new;

    Psi_r_next[idx] = Pr_new;
    Psi_r_prev_next[idx] = absorb * Pr;      // damp prev too
    Psi_i_next[idx] = Pi_new;
    Psi_i_prev_next[idx] = absorb * Pi_val;  // damp prev too
    chi_next[idx] = chi_new;
    chi_prev_next[idx] = chi_c;
}
"""

# ---------------------------------------------------------------------------
# S_a auxiliary field diffusion kernel (v16 — confinement flux tube)
# ---------------------------------------------------------------------------
SA_DIFFUSION_KERNEL_SRC = r"""
extern "C" __global__ __launch_bounds__(256)
void evolve_sa_diffusion(
    // Input: S_a fields, packed [3*N^3]
    const float* __restrict__ Sa_in,
    // Input: |Ψ_a|^2 source for each color, packed [3*N^3]
    const float* __restrict__ psi_sq_colors,
    // Output: S_a fields after Euler step
    float* __restrict__ Sa_out,
    // Grid params
    const int N,
    const float dt,
    const float sa_d,
    const float sa_gamma)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = N * N * N;
    if (idx >= total) return;

    int i = idx / (N * N);
    int j = (idx / N) % N;
    int k = idx % N;

    // 6-point neighbours for standard Laplacian (sufficient for diffusion)
    int ip = ((i + 1) % N) * N * N + j * N + k;
    int im = ((i - 1 + N) % N) * N * N + j * N + k;
    int jp = i * N * N + ((j + 1) % N) * N + k;
    int jm = i * N * N + ((j - 1 + N) % N) * N + k;
    int kp = i * N * N + j * N + (k + 1) % N;
    int km = i * N * N + j * N + (k - 1 + N) % N;

    // Euler update: dS_a/dt = D * Lap(S_a) + gamma * (|Psi_a|^2 - S_a)
    // gamma-normalised source ensures equilibrium S_a -> |Psi_a|^2
    #pragma unroll
    for (int a = 0; a < 3; a++) {
        int off = a * total;
        float sa = Sa_in[off + idx];

        // 6-pt Laplacian (no edges needed — diffusion tolerates lower isotropy)
        float lap_sa = Sa_in[off + ip] + Sa_in[off + im]
                     + Sa_in[off + jp] + Sa_in[off + jm]
                     + Sa_in[off + kp] + Sa_in[off + km]
                     - 6.0f * sa;

        float psi_sq_a = psi_sq_colors[off + idx];
        float sa_new = sa + dt * (sa_d * lap_sa + sa_gamma * (psi_sq_a - sa));

        // S_a must be non-negative
        if (sa_new < 0.0f) sa_new = 0.0f;

        Sa_out[off + idx] = sa_new;
    }
}
"""
