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
EVOLUTION_KERNEL_SRC = r'''
extern "C" __global__
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
    const float eps_cc)
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

    // Accumulate colorblind sources for GOV-02
    float psi_sq_total = 0.0f;
    float j_total = 0.0f;

    // Per-color energy densities for f_c (v14)
    float ea[3];

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

        Psi_r_next[aidx] = Pr_new;
        Psi_r_prev_next[aidx] = Pr;
        Psi_i_next[aidx] = Pi_new;
        Psi_i_prev_next[aidx] = Pi_val;

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
    }

    // v14: normalized color variance f_c = [Sum_a |Psi_a|^4 / (Sum_a |Psi_a|^2)^2] - 1/3
    float color_var_term = 0.0f;
    if (kappa_c > 0.0f && psi_sq_total > 1e-30f) {
        float sum_sq = ea[0]*ea[0] + ea[1]*ea[1] + ea[2]*ea[2];
        float f_c = sum_sq / (psi_sq_total * psi_sq_total) - (1.0f / 3.0f);
        color_var_term = kappa_c * f_c * psi_sq_total;
    }

    // 19-point Laplacian for chi
    float lap_chi = (1.0f/3.0f) * (chi[ip] + chi[im] + chi[jp] + chi[jm] + chi[kp] + chi[km])
                  + (1.0f/6.0f) * (chi[ipjp] + chi[ipjm] + chi[imjp] + chi[imjm]
                                  + chi[ipkp] + chi[ipkm] + chi[imkp] + chi[imkm]
                                  + chi[jpkp] + chi[jpkm] + chi[jmkp] + chi[jmkm])
                  - 4.0f * chi_c;

    // Mexican hat: -4*lam*chi*(chi^2 - chi0^2)
    float chi_self = -4.0f * lam * chi_c * (chi_sq - chi0 * chi0);

    // GOV-02 v14+: colorblind gravity + color variance
    float chi_new = 2.0f * chi_c - chi_prev[idx] + dt2 * (
        lap_chi - kappa * (psi_sq_total + eps_w * j_total - E0_sq)
        - color_var_term + chi_self);

    // BH excision: clamp to Z2 second vacuum
    if (chi_new < -chi0) chi_new = -chi0;

    // Frozen boundary
    float mask = boundary_mask[idx];
    chi_new = mask * chi0 + (1.0f - mask) * chi_new;

    chi_next[idx] = chi_new;
    chi_prev_next[idx] = chi_c;
}
'''

# ---------------------------------------------------------------------------
# Parametric resonance kernel (Phase 1 — matter creation via Mathieu eq.)
# ---------------------------------------------------------------------------
PHASE1_KERNEL_SRC = r'''
extern "C" __global__
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
'''

# ---------------------------------------------------------------------------
# Real-E gravity-only kernel (Level 0 — cosmology, structure formation)
# ---------------------------------------------------------------------------
EVOLUTION_REAL_KERNEL_SRC = r'''
extern "C" __global__
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

    // GOV-02
    float chi_new = 2.0f * chi_c - chi_prev[idx] + dt2 * (
        lap_chi - kappa * (E_c * E_c - E0_sq) + chi_self);

    // BH excision
    if (chi_new < -chi0) chi_new = -chi0;

    // Frozen boundary
    float mask = boundary_mask[idx];
    E_new = (1.0f - mask) * E_new;
    chi_new = mask * chi0 + (1.0f - mask) * chi_new;

    E_next[idx] = E_new;
    E_prev_next[idx] = E_c;
    chi_next[idx] = chi_new;
    chi_prev_next[idx] = chi_c;
}
'''

# ---------------------------------------------------------------------------
# Complex single-component kernel (Level 1 — gravity + EM)
# ---------------------------------------------------------------------------
EVOLUTION_COMPLEX_KERNEL_SRC = r'''
extern "C" __global__
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
        lap_chi - kappa * (psi_sq + eps_w * j_scalar - E0_sq) + chi_self);

    // BH excision
    if (chi_new < -chi0) chi_new = -chi0;

    // Frozen boundary
    float mask = boundary_mask[idx];
    Pr_new = (1.0f - mask) * Pr_new;
    Pi_new = (1.0f - mask) * Pi_new;
    chi_new = mask * chi0 + (1.0f - mask) * chi_new;

    Psi_r_next[idx] = Pr_new;
    Psi_r_prev_next[idx] = Pr;
    Psi_i_next[idx] = Pi_new;
    Psi_i_prev_next[idx] = Pi_val;
    chi_next[idx] = chi_new;
    chi_prev_next[idx] = chi_c;
}
'''
