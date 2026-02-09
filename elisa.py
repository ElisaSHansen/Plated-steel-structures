import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 3D convergence plot for Navier plate deflection
# z = w_max(Mmax, Nmax)
# ============================================================

# -------------------------------
# 1) INPUT: Plate + material data
# -------------------------------
p0 = 10e3          # Uniform pressure [N/m^2] = 10 kN/m^2
a  = 2.8           # Plate length in x1-direction [m]
b  = 0.67          # Plate length in x2-direction [m]
t  = 0.0065        # Thickness [m]
E  = 210e9         # Young's modulus [Pa]
nu = 0.30          # Poisson's ratio [-]

# Grid used to evaluate w_max = max |w(x1,x2)| for each (M,N)
nx = 101
ny = 101

# (M,N) grid for convergence surface (odd values only)
# Keep this moderate for speed; increase if needed.
M_grid = list(range(1, 20, 2))   # 1,3,...,79
N_grid = list(range(1, 20, 2))   # 1,3,...,79


# -------------------------------
# 2) Plate theory functions
# -------------------------------
def plate_rigidity(E, nu, t):
    """Flexural rigidity of isotropic Kirchhoff plate."""
    return E * t**3 / (12.0 * (1.0 - nu**2))

def Pmn_uniform(p0, m, n):
    """
    For uniform load p=p0:
      Pmn = 16 p0 / (pi^2 m n)  for m,n odd
      Pmn = 0 otherwise
    """
    if (m % 2 == 1) and (n % 2 == 1):
        return 16.0 * p0 / (np.pi**2 * m * n)
    return 0.0

def Wmn_isotropic(Pmn, D, a, b, m, n):
    """Navier coefficient for deflection."""
    denom = D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2
    return Pmn / denom


# -------------------------------
# 3) Compute w_max for given (Mmax,Nmax)
# -------------------------------
def w_max_for_MN(a, b, p0, E, nu, t, Mmax, Nmax, nx=101, ny=101):
    """
    Computes w(x1,x2) on a grid for truncation (Mmax,Nmax)
    and returns max absolute deflection.
    """
    D = plate_rigidity(E, nu, t)

    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)
    w = np.zeros((ny, nx), dtype=float)

    m_list = range(1, Mmax + 1, 2)  # odd
    n_list = range(1, Nmax + 1, 2)  # odd

    # Precompute sines (speed)
    sin_mx1 = {m: np.sin(m * np.pi * x1 / a) for m in m_list}
    sin_nx2 = {n: np.sin(n * np.pi * x2 / b) for n in n_list}

    for m in m_list:
        for n in n_list:
            P = Pmn_uniform(p0, m, n)
            W = Wmn_isotropic(P, D, a, b, m, n)
            w += W * np.outer(sin_nx2[n], sin_mx1[m])

    return float(np.max(np.abs(w)))


# -------------------------------
# 4) Build the convergence surface Wmax(M,N)
# -------------------------------
print("Computing w_max(M,N) surface... (may take a bit)")

Wsurf = np.zeros((len(N_grid), len(M_grid)), dtype=float)

for i, Nmax in enumerate(N_grid):
    for j, Mmax in enumerate(M_grid):
        Wsurf[i, j] = w_max_for_MN(a, b, p0, E, nu, t, Mmax, Nmax, nx=nx, ny=ny)

M_ref = max(M_grid)
N_ref = max(N_grid)
w_ref = Wsurf[-1, -1]
print(f"Reference: w_max at (M,N)=({M_ref},{N_ref}) is {w_ref:.6e} m")


# -------------------------------
# 5) 3D surface plot: x=M, y=N, z=w_max
# -------------------------------
M_mesh, N_mesh = np.meshgrid(M_grid, N_grid)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(M_mesh, N_mesh, Wsurf, rstride=1, cstride=1, linewidth=0)

ax.set_xlabel("Mmax (odd terms)")
ax.set_ylabel("Nmax (odd terms)")
ax.set_zlabel("w_max [m]")
ax.set_title("Convergence surface: w_max(Mmax, Nmax)")

# Mark the reference point (largest M,N)
ax.scatter([M_ref], [N_ref], [w_ref], marker="o")
plt.tight_layout()
plt.show()


# -------------------------------
# 6) 2D contour plot (often clearer for reports)
# -------------------------------
plt.figure(figsize=(7, 5))
cs = plt.contourf(M_mesh, N_mesh, Wsurf, levels=30)
plt.colorbar(cs, label="w_max [m]")
plt.xlabel("Mmax (odd terms)")
plt.ylabel("Nmax (odd terms)")
plt.title("Convergence map: w_max(Mmax, Nmax)")
plt.scatter([M_ref], [N_ref], marker="o")  # reference point
plt.tight_layout()
plt.show()


# -------------------------------
# 7) Optional: show convergence error relative to reference
# -------------------------------
# This makes it very obvious when you're "close enough".
Err = np.abs(Wsurf - w_ref) / np.abs(w_ref)

plt.figure(figsize=(7, 5))
cs2 = plt.contourf(M_mesh, N_mesh, Err, levels=30)
plt.colorbar(cs2, label="Relative error |w_max - w_ref| / |w_ref|")
plt.xlabel("Mmax (odd terms)")
plt.ylabel("Nmax (odd terms)")
plt.title("Relative convergence error vs (Mmax, Nmax)")
plt.scatter([M_ref], [N_ref], marker="o")
plt.tight_layout()
plt.show()
