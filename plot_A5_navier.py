import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# A5 – Simply supported plate (Navier-type) with PATCH LOAD
#
# This version matches what you wanted (same style as earlier):
#  1) Builds convergence surface Wsurf(N,M) = w_max(M,N)
#  2) Finds first (M,N) meeting tol vs reference at (M_max,N_max)
#  3) NO 3D plot, NO relative error map
#  4) Makes TWO 2D plots in ONE figure (two subplots):
#       (a) w_max vs M  (with N fixed)
#       (b) w_max vs N  (with M fixed)
#     - y-axis in mm
#     - integer ticks on x-axes
#     - red convergence marker
# ============================================================

# -----------------------------
# 1) INPUT (A5 – patch load in center)
# -----------------------------
P_total = 20000.0       # N (20 kN total force)
a = 2.8                # m
b = 0.67               # m

# Patch dimensions (u along x1, v along x2)
u = 0.25               # m (250 mm) along x1
v = 0.20               # m (200 mm) along x2

t  = 0.0065            # m
E  = 210e9             # Pa
nu = 0.30

# Patch is centered
xi1 = a / 2.0
xi2 = b / 2.0

# Convergence settings
M_max = 20
N_max = 20
tol = 1e-4

# Field grid for evaluating max deflection
nx = 81
ny = 81

# Toggle:
#   False = robust (recommended)
#   True  = uses only odd-odd terms (only safe if patch truly centered & symmetric)
ONLY_ODD = False

# -----------------------------
# 2) Derived quantities
# -----------------------------
D = E * t**3 / (12.0 * (1.0 - nu**2))

# Uniform pressure on patch
A_patch = u * v
p0 = P_total / A_patch  # N/m^2

print("=== Load sanity check ===")
print(f"A_patch = {A_patch:.6f} m^2")
print(f"p0 = {p0:.3f} N/m^2  (= {p0/1e3:.3f} kN/m^2 = {p0/1e6:.3f} MPa)")
print(f"P_check = p0*A_patch = {p0*A_patch:.3f} N")
print("=========================\n")

# -----------------------------
# 3) Pmn for rectangular patch load (general m,n)
# -----------------------------
def _int_sin_over_centered_interval(m, L, center, width):
    """
    Computes: ∫_{center-width/2}^{center+width/2} sin(m*pi*x/L) dx
    closed form. Works for any m.
    """
    k = m * np.pi / L
    x1 = center - width / 2.0
    x2 = center + width / 2.0
    return (-np.cos(k * x2) + np.cos(k * x1)) / k

def Pmn_patch(m, n):
    """
    Fourier coefficient for patch pressure on simply supported plate:
      Pmn = (4/(a*b)) ∬_patch p0 sin(mπx/a) sin(nπy/b) dA
    """
    Ix = _int_sin_over_centered_interval(m, a, xi1, u)
    Iy = _int_sin_over_centered_interval(n, b, xi2, v)
    return (4.0 / (a * b)) * p0 * Ix * Iy

def Wmn(m, n):
    denom = D * np.pi**4 * ((m / a)**2 + (n / b)**2)**2
    return Pmn_patch(m, n) / denom

# -----------------------------
# 4) Compute w-field and max|w| for given (M,N)
# -----------------------------
def compute_w_max(Mmax, Nmax, only_odd=False):
    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)
    w = np.zeros((ny, nx))

    if only_odd:
        m_list = range(1, Mmax + 1, 2)
        n_list = range(1, Nmax + 1, 2)
    else:
        m_list = range(1, Mmax + 1)
        n_list = range(1, Nmax + 1)

    sin_m = {m: np.sin(m * np.pi * x1 / a) for m in m_list}
    sin_n = {n: np.sin(n * np.pi * x2 / b) for n in n_list}

    for m in m_list:
        for n in n_list:
            w += Wmn(m, n) * np.outer(sin_n[n], sin_m[m])

    return float(np.max(np.abs(w)))  # [m]

# -----------------------------
# 5) Convergence surface + convergence point
# -----------------------------
print("Computing convergence surface...")
Wsurf = np.zeros((N_max, M_max), dtype=float)

for M in range(1, M_max + 1):
    for N in range(1, N_max + 1):
        Wsurf[N - 1, M - 1] = compute_w_max(M, N, only_odd=ONLY_ODD)

w_ref = float(Wsurf[N_max - 1, M_max - 1])  # [m]
Err = np.abs(Wsurf - w_ref) / (abs(w_ref) if abs(w_ref) > 0 else 1.0)

# Find first (M,N) meeting tol using budget k=max(M,N)
Mconv = None
Nconv = None

for k in range(1, max(M_max, N_max) + 1):
    candidates = []

    # N = k candidates
    for M in range(1, min(M_max, k) + 1):
        N = k
        if N <= N_max and Err[N - 1, M - 1] <= tol:
            candidates.append((M, N))

    # M = k candidates
    for N in range(1, min(N_max, k) + 1):
        M = k
        if M <= M_max and Err[N - 1, M - 1] <= tol:
            candidates.append((M, N))

    if candidates:
        Mconv, Nconv = min(candidates, key=lambda t: (t[0] + t[1], t[0], t[1]))
        break

print("\n===== Deflection convergence (patch load) =====")
print(f"Reference w_max at (M_max,N_max)=({M_max},{N_max}) = {w_ref*1e3:.4f} mm")
if Mconv is None:
    print(f"NOT converged for tol={tol} within limits.")
    M_use, N_use = M_max, N_max
else:
    print(f"tol = {tol}")
    print(f"Converged at: Mconv={Mconv}, Nconv={Nconv}")
    print(f"w_max(Mconv,Nconv) = {Wsurf[Nconv-1, Mconv-1]*1e3:.4f} mm")
    print(f"relative error = {Err[Nconv-1, Mconv-1]:.3e}")
    M_use, N_use = Mconv, Nconv

# -----------------------------
# 6) TWO 2D plots in ONE figure (mm) + red convergence marker
# -----------------------------
M_vals = np.arange(1, M_max + 1)
N_vals = np.arange(1, N_max + 1)

Wsurf_mm = Wsurf * 1e3
w_ref_mm = w_ref * 1e3

# Curves
w_vs_M = Wsurf_mm[N_use - 1, :]       # fixed N_use
w_vs_N = Wsurf_mm[:, M_use - 1]       # fixed M_use

# Point value
w_point_mm = Wsurf_mm[N_use - 1, M_use - 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# ---- Left: w_max vs M (N fixed)
ax1.plot(M_vals, w_vs_M, marker="o", markersize=3, linewidth=1)
ax1.axhline(w_ref_mm, linestyle="--", linewidth=1,
            label=f"w_ref (M={M_max}, N={N_max})")
ax1.scatter([M_use], [w_point_mm], s=90, color="red", edgecolors="black", zorder=5,
            label=f"Punkt (M,N)=({M_use},{N_use})")
ax1.set_xlabel("Mmax")
ax1.set_ylabel("w_max [mm]")
ax1.set_title(f"w_max vs Mmax (Nmax = {N_use})   |  ONLY_ODD={ONLY_ODD}")
ax1.set_xticks(np.arange(1, M_max + 1, 1))  # integers only
ax1.grid(True, alpha=0.3)
ax1.legend()

# ---- Right: w_max vs N (M fixed)
ax2.plot(N_vals, w_vs_N, marker="o", markersize=3, linewidth=1)
ax2.axhline(w_ref_mm, linestyle="--", linewidth=1,
            label=f"w_ref (M={M_max}, N={N_max})")
ax2.scatter([N_use], [w_point_mm], s=90, color="red", edgecolors="black", zorder=5,
            label=f"Punkt (M,N)=({M_use},{N_use})")
ax2.set_xlabel("Nmax")
ax2.set_title(f"w_max vs Nmax (Mmax = {M_use})   |  ONLY_ODD={ONLY_ODD}")
ax2.set_xticks(np.arange(1, N_max + 1, 1))  # integers only
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()
