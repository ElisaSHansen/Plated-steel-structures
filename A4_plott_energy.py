import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Rayleigh–Ritz (Energy method) – CCSS
# Uavhengig konvergens i M og N, med plott som matcher Navier-stilen:
# - to subplot ved siden av hverandre
# - samme aksetikk (heltall)
# - samme markør, linjetykkelse, grid, legend
# - referanselinje (stiplet) og rød markør med svart kant
# ============================================================

# -----------------------------
# 1) Inputs
# -----------------------------
p0 = 10e3          # N/m^2
a  = 2.8           # m
b  = 0.67          # m
t  = 0.0065        # m
E  = 210e9         # Pa
nu = 0.30          # -
D  = E * t**3 / (12.0 * (1.0 - nu**2))

M_limit = 20
N_limit = 20


# ============================================================
# 2) Basis functions
# ============================================================
def basis_SS(k, L, x):
    return np.sin(k * np.pi * x / L)

def basis_SS_d1(k, L, x):
    return (k * np.pi / L) * np.cos(k * np.pi * x / L)

def basis_SS_d2(k, L, x):
    return - (k * np.pi / L)**2 * np.sin(k * np.pi * x / L)

def basis_CC(k, L, x):
    # admissible clamped-clamped: w=0 and dw/dx=0 at x=0,L
    return np.sin(np.pi * x / L) * np.sin(k * np.pi * x / L)

def basis_CC_d1(k, L, x):
    s1 = np.sin(np.pi * x / L)
    c1 = np.cos(np.pi * x / L)
    sk = np.sin(k * np.pi * x / L)
    ck = np.cos(k * np.pi * x / L)
    return (np.pi / L) * c1 * sk + (k * np.pi / L) * s1 * ck

def basis_CC_d2(k, L, x):
    s1 = np.sin(np.pi * x / L)
    c1 = np.cos(np.pi * x / L)
    sk = np.sin(k * np.pi * x / L)
    ck = np.cos(k * np.pi * x / L)
    term1 = (np.pi / L) * (-(np.pi / L) * s1 * sk + c1 * (k * np.pi / L) * ck)
    term2 = (k * np.pi / L) * ((np.pi / L) * c1 * ck + s1 * (-(k * np.pi / L)) * sk)
    return term1 + term2


# ============================================================
# 3) Quadrature utilities
# ============================================================
def gauss_interval(n, x1, x2):
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (xi + 1.0) * (x2 - x1) + x1
    w = 0.5 * (x2 - x1) * wi
    return x, w

def int_mat(A, B, w):
    return (A * w) @ B.T


# ============================================================
# 4) Rayleigh–Ritz solver for uniform load (efficient Kronecker)
#    CC i x-retning, SS i y-retning  (CCSS)
# ============================================================
def ritz_solve_uniform(M, N):
    f0x, f1x, f2x = basis_CC, basis_CC_d1, basis_CC_d2
    f0y, f1y, f2y = basis_SS, basis_SS_d1, basis_SS_d2

    nqx = max(60, 3 * M)
    nqy = max(60, 3 * N)

    xq, wx = gauss_interval(nqx, 0.0, a)
    yq, wy = gauss_interval(nqy, 0.0, b)

    X0 = np.array([f0x(m, a, xq) for m in range(1, M + 1)])
    X1 = np.array([f1x(m, a, xq) for m in range(1, M + 1)])
    X2 = np.array([f2x(m, a, xq) for m in range(1, M + 1)])

    Y0 = np.array([f0y(n, b, yq) for n in range(1, N + 1)])
    Y1 = np.array([f1y(n, b, yq) for n in range(1, N + 1)])
    Y2 = np.array([f2y(n, b, yq) for n in range(1, N + 1)])

    Ix_00 = int_mat(X0, X0, wx)
    Ix_11 = int_mat(X1, X1, wx)
    Ix_20 = int_mat(X2, X0, wx)
    Ix_02 = Ix_20.T
    Ix_22 = int_mat(X2, X2, wx)

    Iy_00 = int_mat(Y0, Y0, wy)
    Iy_11 = int_mat(Y1, Y1, wy)
    Iy_20 = int_mat(Y2, Y0, wy)
    Iy_02 = Iy_20.T
    Iy_22 = int_mat(Y2, Y2, wy)

    A = D * (
        np.kron(Iy_00, Ix_22)
        + np.kron(Iy_22, Ix_00)
        + nu * (np.kron(Iy_02, Ix_20) + np.kron(Iy_20, Ix_02))
        + 2.0 * (1.0 - nu) * np.kron(Iy_11, Ix_11)
    )

    Ix_0 = (X0 * wx).sum(axis=1)  # (M,)
    Iy_0 = (Y0 * wy).sum(axis=1)  # (N,)
    B = (p0 * np.outer(Ix_0, Iy_0)).reshape(M * N)  # idx(m,n)=m*N+n

    W = np.linalg.solve(A, B).reshape(M, N)
    return W


# ============================================================
# 5) Compute w at center from Wmn
# ============================================================
def w_center_from_W(W):
    M, N = W.shape
    xc, yc = a / 2.0, b / 2.0

    Xc = np.array([basis_CC(m, a, xc) for m in range(1, M + 1)])
    Yc = np.array([basis_SS(n, b, yc) for n in range(1, N + 1)])

    return float(Xc @ W @ Yc)  # meters


# ============================================================
# 6) Data: to kurver med fast N=20 og fast M=20
# ============================================================
M_vals = np.arange(1, M_limit + 1)
N_vals = np.arange(1, N_limit + 1)

N_use = N_limit
M_use = M_limit

w_vs_M_mm = []
for Mmax in M_vals:
    W = ritz_solve_uniform(int(Mmax), int(N_use))
    w_vs_M_mm.append(w_center_from_W(W) * 1e3)

w_vs_N_mm = []
for Nmax in N_vals:
    W = ritz_solve_uniform(int(M_use), int(Nmax))
    w_vs_N_mm.append(w_center_from_W(W) * 1e3)

w_vs_M_mm = np.array(w_vs_M_mm)
w_vs_N_mm = np.array(w_vs_N_mm)

# Referanse ved (20,20)
w_ref_mm = float(w_vs_M_mm[-1])  # samme som w_vs_N_mm[-1]

# Punkt som markeres (M_use,N_use) = (20,20)
w_point_mm = w_ref_mm


# ============================================================
# 7) Plot: matcher Navier-utseende
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

# ---- Venstre: w_center vs M (N fast)
ax1.plot(M_vals, w_vs_M_mm, marker="o", markersize=3, linewidth=1)
ax1.axhline(w_ref_mm, linestyle="--", linewidth=1, label=f"w_ref (M={M_limit}, N={N_limit})")
ax1.scatter([M_use], [w_point_mm], s=90, color="red", edgecolors="black", zorder=5,
            label=f"Point (M,N)=({M_use},{N_use})")
ax1.set_xlabel("Mmax")
ax1.set_ylabel("w_center [mm]")
ax1.set_title(f"w_center vs Mmax (Nmax = {N_use})")
ax1.set_xticks(np.arange(1, M_limit + 1, 1))
ax1.grid(True, alpha=0.3)
ax1.legend()

# ---- Høyre: w_center vs N (M fast)
ax2.plot(N_vals, w_vs_N_mm, marker="o", markersize=3, linewidth=1)
ax2.axhline(w_ref_mm, linestyle="--", linewidth=1, label=f"w_ref (M={M_limit}, N={N_limit})")
ax2.scatter([N_use], [w_point_mm], s=90, color="red", edgecolors="black", zorder=5,
            label=f"Point (M,N)=({M_use},{N_use})")
ax2.set_xlabel("Nmax")
ax2.set_title(f"w_center vs Nmax (Mmax = {M_use})")
ax2.set_xticks(np.arange(1, N_limit + 1, 1))
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()