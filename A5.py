import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# A5 – Simply supported plate (Navier-type) with PATCH LOAD
# (Your approach with Pmn for a rectangular loaded area)
#
# This version is written "robustly":
#   - Uses the correct uniform pressure: p0 = P_total / (u*v)
#   - Does NOT assume "only odd" terms (you can enable that later)
#     because the odd-only shortcut is easy to get wrong if u/v or
#     the centering is mismatched.
#   - Computes w(x1,x2) and finds max|w| over the grid.
#   - Performs convergence study vs (M,N) and reports (Mconv,Nconv).
#
# NOTE: This is for SSSS (Navier-like) only, matching your code structure.
# ============================================================

# -----------------------------
# 1) INPUT (A5 – patch load in center)
# -----------------------------
P_total = 20000.0       # N (20 kN total force)
a = 2.8                # m
b = 0.67               # m

# Patch dimensions (u along x1, v along x2)
u = 0.25               # m  (250 mm) along x1
v = 0.20               # m  (200 mm) along x2

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

# -----------------------------
# 2) Derived quantities
# -----------------------------
D = E * t**3 / (12.0 * (1.0 - nu**2))

# Uniform pressure on patch (IMPORTANT)
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
    in closed form. Works for any m (not only odd).
    """
    k = m * np.pi / L
    x1 = center - width/2.0
    x2 = center + width/2.0
    # ∫ sin(kx) dx = -cos(kx)/k
    return (-np.cos(k*x2) + np.cos(k*x1)) / k

def Pmn_patch(m, n):
    """
    Fourier coefficient for patch pressure on simply supported plate:
      Pmn = (4/(a*b)) ∬_patch p0 sin(mπx/a) sin(nπy/b) dA

    Because separable:
      ∬ = (∫ sin(mπx/a) dx)(∫ sin(nπy/b) dy)
    """
    Ix = _int_sin_over_centered_interval(m, a, xi1, u)
    Iy = _int_sin_over_centered_interval(n, b, xi2, v)
    return (4.0/(a*b)) * p0 * Ix * Iy

def Wmn(m, n):
    denom = D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2
    return Pmn_patch(m, n) / denom

# -----------------------------
# 4) Compute w-field and max|w| for given (M,N)
# -----------------------------
def compute_w_max(Mmax, Nmax, only_odd=False):
    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)
    w = np.zeros((ny, nx))

    # choose index sets
    if only_odd:
        m_list = range(1, Mmax+1, 2)
        n_list = range(1, Nmax+1, 2)
    else:
        m_list = range(1, Mmax+1)
        n_list = range(1, Nmax+1)

    sin_m = {m: np.sin(m*np.pi*x1/a) for m in m_list}
    sin_n = {n: np.sin(n*np.pi*x2/b) for n in n_list}

    for m in m_list:
        for n in n_list:
            w += Wmn(m, n) * np.outer(sin_n[n], sin_m[m])

    return float(np.max(np.abs(w)))

# Toggle this:
#   False = robust (recommended)
#   True  = uses only odd-odd terms (only safe if patch truly centered & symmetric)
ONLY_ODD = False

# -----------------------------
# 5) Convergence surface
# -----------------------------
print("Computing convergence surface...")
Wsurf = np.zeros((N_max, M_max))

for M in range(1, M_max+1):
    for N in range(1, N_max+1):
        Wsurf[N-1, M-1] = compute_w_max(M, N, only_odd=ONLY_ODD)

w_ref = float(Wsurf[N_max-1, M_max-1])
Err = np.abs(Wsurf - w_ref) / (abs(w_ref) if abs(w_ref) > 0 else 1.0)

# find first (M,N) meeting tol using budget k=max(M,N)
Mconv = None
Nconv = None

for k in range(1, max(M_max, N_max)+1):
    candidates = []
    for M in range(1, min(M_max, k)+1):
        N = k
        if N <= N_max and Err[N-1, M-1] <= tol:
            candidates.append((M, N))
    for N in range(1, min(N_max, k)+1):
        M = k
        if M <= M_max and Err[N-1, M-1] <= tol:
            candidates.append((M, N))
    if candidates:
        Mconv, Nconv = min(candidates, key=lambda t: (t[0]+t[1], t[0], t[1]))
        break

print("\n===== Deflection convergence (patch load) =====")
print(f"Reference w_max at (M_max,N_max)=({M_max},{N_max}) = {w_ref*1e3:.4f} mm")
if Mconv is None:
    print(f"NOT converged for tol={tol} within limits.")
else:
    print(f"tol = {tol}")
    print(f"Converged at: Mconv={Mconv}, Nconv={Nconv}")
    print(f"w_max(Mconv,Nconv) = {Wsurf[Nconv-1, Mconv-1]*1e3:.4f} mm")
    print(f"relative error = {Err[Nconv-1, Mconv-1]:.3e}")

# -----------------------------
# 6) 3D plot of convergence surface
# -----------------------------
M_vals = np.arange(1, M_max+1)
N_vals = np.arange(1, N_max+1)
M_mesh, N_mesh = np.meshgrid(M_vals, N_vals)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    M_mesh, N_mesh, Wsurf*1e3,  # show mm
    cmap="plasma",
    edgecolor="k",
    linewidth=0.2,
    antialiased=True
)

cbar = fig.colorbar(surf, ax=ax, shrink=0.75, aspect=18, pad=0.15)
cbar.set_label("w_max [mm]", rotation=90, labelpad=15)
cbar.ax.yaxis.set_label_position("left")
cbar.ax.yaxis.tick_left()

ax.set_xlabel("Mmax")
ax.set_ylabel("Nmax", labelpad=20)
ax.set_zlabel("")
ax.set_title(f"Convergence for patch load (ONLY_ODD={ONLY_ODD})")

zmin = float(np.min(Wsurf*1e3))
zmax = float(np.max(Wsurf*1e3))
ax.set_zlim(zmin, zmax * 1.15)

if Mconv is not None:
    Zc = float(Wsurf[Nconv-1, Mconv-1]*1e3)
    eps = 1e-4 * (zmax - zmin)
    ax.scatter(
        Mconv, Nconv, Zc + eps,
        color="red", s=120,
        edgecolors="black", linewidths=1,
        depthshade=False
    )

ax.view_init(elev=30, azim=220)
plt.tight_layout()
plt.show()
