import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Navier plate solution (simply supported) - DEFLECTION ONLY
# Uniform pressure load, odd-odd terms only
#
# This script:
#  1) Builds convergence surface Wsurf(N,M) = w_max(M,N)
#  2) Finds first (M,N) meeting rel_tol vs reference at (M_limit,N_limit)
#  3) Plots TWO 2D curves in ONE figure (two subplots):
#       (a) w_max vs M  (with N fixed)
#       (b) w_max vs N  (with M fixed)
#     - y-axis in mm
#     - integer ticks on x-axes
#     - red convergence marker
# ============================================================

# -----------------------------
# 1) INPUT: Plate and material
# -----------------------------
p0 = 10e3          # Uniform pressure [N/m^2] = 10 kN/m^2
a  = 2.8           # [m]
b  = 0.67          # [m]
t  = 0.0065        # [m]
E  = 210e9         # [Pa]
nu = 0.30          # [-]

# Grid used to evaluate w_max over the plate
nx = 81
ny = 81

# Limits for convergence scan
M_limit = 20
N_limit = 20

# Convergence tolerance (relative error vs reference)
rel_tol = 1e-4

# -----------------------------
# 2) Plate theory helpers
# -----------------------------
def plate_rigidity(E_, nu_, t_):
    return E_ * t_**3 / (12.0 * (1.0 - nu_**2))

def Pmn_uniform(p0_, m, n):
    """
    Uniform pressure Fourier coefficient.
    If m OR n is even -> return 0.
    Only odd-odd terms contribute.
    """
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0
    return 16.0 * p0_ / (np.pi**2 * m * n)

def Wmn(Pmn_, D_, a_, b_, m, n):
    denom = D_ * np.pi**4 * ((m/a_)**2 + (n/b_)**2)**2
    return Pmn_ / denom

# -----------------------------
# 3) Compute w_max for (Mmax,Nmax)
# -----------------------------
def compute_w_max(Mmax, Nmax, nx_=81, ny_=81):
    D = plate_rigidity(E, nu, t)
    x1 = np.linspace(0.0, a, nx_)
    x2 = np.linspace(0.0, b, ny_)
    w = np.zeros((ny_, nx_), dtype=float)

    # Precompute sine arrays (only odd indices needed)
    sin_mx1 = {m: np.sin(m * np.pi * x1 / a) for m in range(1, Mmax + 1) if m % 2 == 1}
    sin_nx2 = {n: np.sin(n * np.pi * x2 / b) for n in range(1, Nmax + 1) if n % 2 == 1}

    for m in range(1, Mmax + 1):
        for n in range(1, Nmax + 1):
            P = Pmn_uniform(p0, m, n)
            if P == 0.0:
                continue
            coeff = Wmn(P, D, a, b, m, n)
            w += coeff * np.outer(sin_nx2[n], sin_mx1[m])

    return float(np.max(np.abs(w)))  # [m]

# -----------------------------
# 4) Build convergence surface Wsurf(N,M) = w_max(M,N)
# -----------------------------
print("Computing convergence surface Wsurf(Mmax,Nmax)...")

Wsurf = np.zeros((N_limit, M_limit), dtype=float)

for Mmax in range(1, M_limit + 1):
    for Nmax in range(1, N_limit + 1):
        Wsurf[Nmax - 1, Mmax - 1] = compute_w_max(Mmax, Nmax, nx_=nx, ny_=ny)

# Reference value at the largest (M,N)
w_ref = Wsurf[N_limit - 1, M_limit - 1]
print(f"Reference w_ref = w_max(M={M_limit}, N={N_limit}) = {w_ref:.6e} m  = {w_ref*1000:.6e} mm")

# -----------------------------
# 5) Find (Mconv, Nconv) for convergence
# -----------------------------
Err = np.abs(Wsurf - w_ref) / (abs(w_ref) if abs(w_ref) > 0 else 1.0)

Mconv = None
Nconv = None

# Choose the smallest (M,N) meeting rel_tol using budget k = max(M,N)
for k in range(1, max(M_limit, N_limit) + 1):
    candidates = []

    # candidates where N=k
    for M in range(1, min(M_limit, k) + 1):
        N = k
        if N <= N_limit and Err[N - 1, M - 1] <= rel_tol:
            candidates.append((M, N))

    # candidates where M=k
    for N in range(1, min(N_limit, k) + 1):
        M = k
        if M <= M_limit and Err[N - 1, M - 1] <= rel_tol:
            candidates.append((M, N))

    if candidates:
        # pick smallest (M+N), then M, then N
        Mconv, Nconv = min(candidates, key=lambda tup: (tup[0] + tup[1], tup[0], tup[1]))
        break

if Mconv is None:
    print(f"Convergence NOT reached within (M,N)=({M_limit},{N_limit}) for rel_tol={rel_tol}.")
    M_use, N_use = M_limit, N_limit
else:
    w_conv = Wsurf[Nconv - 1, Mconv - 1]
    rel_err_conv = Err[Nconv - 1, Mconv - 1]
    print("\n===== Convergence found =====")
    print(f"rel_tol = {rel_tol}")
    print(f"Converged at: Mconv = {Mconv}, Nconv = {Nconv}")
    print(f"w_max(Mconv,Nconv) = {w_conv*1000:.3e} mm")
    print(f"relative error vs reference = {rel_err_conv:.3e}")
    M_use, N_use = Mconv, Nconv

# -----------------------------
# 6) Two 2D plots in ONE figure (mm) with integer ticks + red marker
# -----------------------------
M_vals = np.arange(1, M_limit + 1)
N_vals = np.arange(1, N_limit + 1)

Wsurf_mm = Wsurf * 1000.0  # m -> mm
w_ref_mm = w_ref * 1000.0

# Data for curves
w_vs_M = Wsurf_mm[N_use - 1, :]      # w_max vs M at fixed N_use
w_vs_N = Wsurf_mm[:, M_use - 1]      # w_max vs N at fixed M_use

# Convergence point y-value (even if not converged, it's the chosen point)
w_point_mm = Wsurf_mm[N_use - 1, M_use - 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# ---- Left: w_max vs M (N fixed)
ax1.plot(M_vals, w_vs_M, marker="o", markersize=3, linewidth=1)
ax1.axhline(w_ref_mm, linestyle="--", linewidth=1, label=f"w_ref (M={M_limit}, N={N_limit})")
ax1.scatter([M_use], [w_point_mm], s=90, color="red", edgecolors="black", zorder=5,
            label=f"Point (M,N)=({M_use},{N_use})")
ax1.set_xlabel("Mmax")
ax1.set_ylabel("w_max [mm]")
ax1.set_title(f"w_max vs Mmax (Nmax = {N_use})")
ax1.set_xticks(np.arange(1, M_limit + 1, 1))   # integer ticks only
ax1.grid(True, alpha=0.3)
ax1.legend()

# ---- Right: w_max vs N (M fixed)
ax2.plot(N_vals, w_vs_N, marker="o", markersize=3, linewidth=1)
ax2.axhline(w_ref_mm, linestyle="--", linewidth=1, label=f"w_ref (M={M_limit}, N={N_limit})")
ax2.scatter([N_use], [w_point_mm], s=90, color="red", edgecolors="black", zorder=5,
            label=f"Point (M,N)=({M_use},{N_use})")
ax2.set_xlabel("Nmax")
ax2.set_title(f"w_max vs Nmax (Mmax = {M_use})")
ax2.set_xticks(np.arange(1, N_limit + 1, 1))   # integer ticks only
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
