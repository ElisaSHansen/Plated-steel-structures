import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# 3D convergence plot for Navier plate solution (simply supported)
# z = w_max(Mmax, Nmax) where w_max = max_{plate} |w(x1,x2)|
#
# Includes: if m OR n is EVEN -> Pmn = 0 (no contribution)
# Adds: convergence marker and prints/returns (Mconv, Nconv)
#
# Convergence definition used:
#   rel_err(M,N) = |w_max(M,N) - w_ref| / |w_ref|
# where w_ref = w_max(M_limit, N_limit)
# Converged if rel_err <= rel_tol
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

# Limits for the convergence surface axes
M_limit = 20
N_limit = 20

# Convergence tolerance (e.g. 1e-3 = 0.1%)
rel_tol = 1e-4

# -----------------------------
# 2) Plate theory
# -----------------------------
def plate_rigidity(E, nu, t):
    return E * t**3 / (12.0 * (1.0 - nu**2))

def Pmn_uniform(p0, m, n):
    """
    Uniform pressure Fourier coefficient.
    If m OR n is even -> return 0.
    Only odd-odd terms contribute.
    """
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0
    return 16.0 * p0 / (np.pi**2 * m * n)

def Wmn(Pmn, D, a, b, m, n):
    denom = D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2
    return Pmn / denom

# -----------------------------
# 3) Compute w_max for (Mmax,Nmax)
# -----------------------------
def compute_w_max(Mmax, Nmax, nx=81, ny=81):
    D = plate_rigidity(E, nu, t)
    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)
    w = np.zeros((ny, nx), dtype=float)

    # Precompute sine arrays (speed)
    sin_mx1 = {m: np.sin(m * np.pi * x1 / a) for m in range(1, Mmax + 1) if m % 2 == 1}
    sin_nx2 = {n: np.sin(n * np.pi * x2 / b) for n in range(1, Nmax + 1) if n % 2 == 1}

    for m in range(1, Mmax + 1):
        for n in range(1, Nmax + 1):
            P = Pmn_uniform(p0, m, n)
            if P == 0.0:
                continue

            coeff = Wmn(P, D, a, b, m, n)
            w += coeff * np.outer(sin_nx2[n], sin_mx1[m])

    return float(np.max(np.abs(w)))

# -----------------------------
# 4) Build convergence surface Wsurf(N,M) = w_max(M,N)
# -----------------------------
print("Computing convergence surface Wsurf(Mmax,Nmax)...")

Wsurf = np.zeros((N_limit, M_limit), dtype=float)

for Mmax in range(1, M_limit + 1):
    for Nmax in range(1, N_limit + 1):
        Wsurf[Nmax - 1, Mmax - 1] = compute_w_max(Mmax, Nmax, nx=nx, ny=ny)

# Reference value at the largest (M,N)
w_ref = Wsurf[N_limit - 1, M_limit - 1]
print(f"Reference w_ref = w_max(M={M_limit}, N={N_limit}) = {w_ref:.6e} m")

# -----------------------------
# 5) Find (Mconv, Nconv) for convergence
# -----------------------------
# Relative error surface
Err = np.abs(Wsurf - w_ref) / (abs(w_ref) if abs(w_ref) > 0 else 1.0)

# Choose the *smallest* (M,N) meeting rel_tol, using "budget" k = max(M,N)
Mconv = None
Nconv = None

for k in range(1, max(M_limit, N_limit) + 1):
    candidates = []
    for M in range(1, min(M_limit, k) + 1):
        N = k  # keep max(M,N)=k by setting N=k (then also check swapped)
        if N <= N_limit and Err[N - 1, M - 1] <= rel_tol:
            candidates.append((M, N))
    for N in range(1, min(N_limit, k) + 1):
        M = k
        if M <= M_limit and Err[N - 1, M - 1] <= rel_tol:
            candidates.append((M, N))

    if candidates:
        # pick the smallest M+N among candidates at this k
        Mconv, Nconv = min(candidates, key=lambda tup: (tup[0] + tup[1], tup[0], tup[1]))
        break

if Mconv is None:
    print(f"Convergence NOT reached within (M,N)=({M_limit},{N_limit}) for rel_tol={rel_tol}.")
else:
    w_conv = Wsurf[Nconv - 1, Mconv - 1]
    rel_err_conv = Err[Nconv - 1, Mconv - 1]
    print("\n===== Convergence found =====")
    print(f"rel_tol = {rel_tol}")
    print(f"Converged at: Mconv = {Mconv}, Nconv = {Nconv}")
    print(f"w_max(Mconv,Nconv) = {w_conv*1000:.3e} mm")
    print(f"relative error vs reference = {rel_err_conv:.3e}")

# -----------------------------
# 6) Improved 3D surface plot + convergence marker
# -----------------------------
M_vals = np.arange(1, M_limit + 1)
N_vals = np.arange(1, N_limit + 1)
M_mesh, N_mesh = np.meshgrid(M_vals, N_vals)

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

# Better colormap + visible mesh lines
surf = ax.plot_surface(
    M_mesh,
    N_mesh,
    Wsurf,
    cmap="plasma",          # more contrast than default
    edgecolor="k",          # black grid lines
    linewidth=0.2,
    antialiased=True,
    alpha=0.95
)

# Add colorbar
cbar = fig.colorbar(surf, shrink=0.6, aspect=12)
cbar.set_label("w_max [m]")

# Make view clearer
ax.view_init(elev=30, azim=220)

ax.set_xlabel("Mmax")
ax.set_ylabel("Nmax")
ax.set_zlabel("w_max [m]")
ax.set_title(
    f"3D Convergence Surface\n"
    f"Convergence when relative error ≤ {rel_tol:g}"
)

# Convergence marker (bright red, larger)
if Mconv is not None:
    ax.scatter(
        Mconv,
        Nconv,
        Wsurf[Nconv - 1, Mconv - 1],
        color="red",
        s=120,
        edgecolors="black",
        label="Convergence point"
    )
    ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 7) Optional: 2D contour of relative error (very clear for reports)
# -----------------------------
plt.figure(figsize=(7, 5))
cs = plt.contourf(M_mesh, N_mesh, Err, levels=30)
plt.colorbar(cs, label="Relative error |w_max - w_ref| / |w_ref|")
plt.xlabel("Mmax")
plt.ylabel("Nmax")
plt.title(f"Relative error map (converged region: ≤ {rel_tol:g})")
if Mconv is not None:
    plt.scatter([Mconv], [Nconv], marker="o")
plt.tight_layout()
plt.show()

# -----------------------------
# 8) Return values (as variables)
# -----------------------------
# You can use these in the rest of your script/report:
#   Mconv, Nconv, w_ref


# ============================================================
# Stress from Navier deflection using DIRECT stress formulas:
#   sigma1 = -(E*z/(1-nu^2)) (w_x1x1 + nu*w_x2x2)
#   sigma2 = -(E*z/(1-nu^2)) (w_x2x2 + nu*w_x1x1)
#   tau12  = -2*G*z*w_x1x2,  G = E/(2(1+nu))
#
# Using Mconv=19, Nconv=5 (as requested)
# ============================================================

# -----------------------------
# 1) Inputs (Project A.4)
# -----------------------------
p0 = 10e3          # N/m^2
a  = 2.8           # m
b  = 0.67          # m
t  = 0.0065        # m
E  = 210e9         # Pa
nu = 0.30          # -
G  = E/(2*(1+nu))  # Pa

Mconv = 19
Nconv = 5

# Grid for evaluating stresses
nx = 121
ny = 121

# Top surface coordinate (x3 = z)
z = t/2

# -----------------------------
# 2) Navier coefficients
# -----------------------------
def plate_rigidity(E, nu, t):
    return E * t**3 / (12.0 * (1.0 - nu**2))

def Pmn_uniform(p0, m, n):
    # uniform load => only odd-odd terms are non-zero
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0
    return 16.0 * p0 / (np.pi**2 * m * n)

def Wmn(Pmn, D, a, b, m, n):
    return Pmn / (D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2)

# -----------------------------
# 3) Compute second derivatives of w on a grid
# -----------------------------
def w_second_derivatives_grid(a, b, p0, E, nu, t, Mmax, Nmax, nx=121, ny=121):
    """
    Returns x1, x2 arrays and fields:
      w_x1x1, w_x2x2, w_x1x2   (ny x nx)
    computed from Navier series.
    """
    D = plate_rigidity(E, nu, t)

    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)

    w_x1x1 = np.zeros((ny, nx))
    w_x2x2 = np.zeros((ny, nx))
    w_x1x2 = np.zeros((ny, nx))

    # Precompute trig arrays
    sin_mx1, cos_mx1 = {}, {}
    for m in range(1, Mmax + 1):
        sin_mx1[m] = np.sin(m*np.pi*x1/a)
        cos_mx1[m] = np.cos(m*np.pi*x1/a)

    sin_nx2, cos_nx2 = {}, {}
    for n in range(1, Nmax + 1):
        sin_nx2[n] = np.sin(n*np.pi*x2/b)
        cos_nx2[n] = np.cos(n*np.pi*x2/b)

    for m in range(1, Mmax + 1):
        for n in range(1, Nmax + 1):
            P = Pmn_uniform(p0, m, n)
            if P == 0.0:
                continue
            W = Wmn(P, D, a, b, m, n)

            S = np.outer(sin_nx2[n], sin_mx1[m])  # sin(y)*sin(x)
            C = np.outer(cos_nx2[n], cos_mx1[m])  # cos(y)*cos(x)

            # Second derivatives of the sine shape:
            w_x1x1 += W * (-(m*np.pi/a)**2) * S
            w_x2x2 += W * (-(n*np.pi/b)**2) * S
            w_x1x2 += W * ((m*np.pi/a)*(n*np.pi/b)) * C

    return x1, x2, w_x1x1, w_x2x2, w_x1x2

# -----------------------------
# 4) Stress fields from the provided formulas
# -----------------------------
x1, x2, w_x1x1, w_x2x2, w_x1x2 = w_second_derivatives_grid(
    a, b, p0, E, nu, t, Mconv, Nconv, nx=nx, ny=ny
)

# Direct stress formulas (your screenshot)
sigma1 = -(E*z/(1-nu**2)) * (w_x1x1 + nu*w_x2x2)
sigma2 = -(E*z/(1-nu**2)) * (w_x2x2 + nu*w_x1x1)
tau12  = -(2*G*z)         * (w_x1x2)

# von Mises (plane stress)
sigma_vm = np.sqrt(sigma1**2 - sigma1*sigma2 + sigma2**2 + 3*tau12**2)

# -----------------------------
# 5) Find maxima + locations
# -----------------------------
def max_abs_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    return field[iy, ix], x1[ix], x2[iy]

def max_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(field), field.shape)
    return field[iy, ix], x1[ix], x2[iy]

s1_max, x_s1, y_s1 = max_abs_with_location(sigma1, x1, x2)
s2_max, x_s2, y_s2 = max_abs_with_location(sigma2, x1, x2)
t12_max, x_t12, y_t12 = max_abs_with_location(tau12, x1, x2)
svm_max, x_vm, y_vm = max_with_location(sigma_vm, x1, x2)

print("\n=== TOP SURFACE MAXIMA (Direct stress formulas) ===")
print(f"Mmax={Mconv}, Nmax={Nconv}")
print(f"Max |sigma1| = {abs(s1_max)/1e6:.3f} MPa at x1={x_s1:.3f} m, x2={y_s1:.3f} m")
print(f"Max |sigma2| = {abs(s2_max)/1e6:.3f} MPa at x1={x_s2:.3f} m, x2={y_s2:.3f} m")
print(f"Max |tau12|  = {abs(t12_max)/1e6:.3f} MPa at x1={x_t12:.3f} m, x2={y_t12:.3f} m")
print(f"Max von Mises= {svm_max/1e6:.3f} MPa at x1={x_vm:.3f} m, x2={y_vm:.3f} m")

# -----------------------------
# 6) Plot von Mises (MPa)
# -----------------------------
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, sigma_vm/1e6, levels=40)
plt.colorbar(cs, label="von Mises stress [MPa]")
plt.scatter([x_vm], [y_vm], marker="x", s=80, color="black", label="Max von Mises")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.title(r"von Mises stress on top surface (Direct formulas, Navier)")
plt.legend()
plt.tight_layout()
plt.show()

