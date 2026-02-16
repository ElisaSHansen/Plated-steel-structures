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


# ============================================================
# Simply supported rectangular plate, uniform pressure load
# Navier series solution (odd-odd terms for uniform load)
#
# Deflection:
#   w(x1,x2) = sum_{m=1..M} sum_{n=1..N} Wmn sin(mπx1/a) sin(nπx2/b)
#
# Wmn = Pmn / (D π^4 ((m/a)^2 + (n/b)^2)^2)
# Pmn = 16 p0 / (π^2 m n)   for m,n odd, else 0
#
# Stresses (top surface z=t/2) from curvatures:
#   sigma1 = -(E z/(1-ν^2)) (w_x1x1 + ν w_x2x2)
#   sigma2 = -(E z/(1-ν^2)) (w_x2x2 + ν w_x1x1)
#   tau12  = -(2 G z) w_x1x2
#   von Mises = sqrt(s1^2 - s1*s2 + s2^2 + 3*t12^2)
#
# This script:
#  1) Builds convergence surface for max von Mises vs (M,N)
#  2) Finds first (M,N) meeting rel_tol vs reference at (M_limit,N_limit)
#  3) Prints max stresses (direct, shear, von Mises) + locations
#  4) Plots convergence surface and final von Mises contour (only plots required)
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
G  = E/(2*(1+nu))  # Pa

# top surface coordinate
z = t / 2.0

# Grid used to evaluate maxima over the plate
nx = 81
ny = 81

# Convergence surface limits (moderate -> speed)
M_limit = 50
N_limit = 50

# Convergence tolerance (relative error vs reference)
rel_tol = 1e-4

# -----------------------------
# 2) Navier helpers
# -----------------------------
def plate_rigidity(E_, nu_, t_):
    return E_ * t_**3 / (12.0 * (1.0 - nu_**2))

def Pmn_uniform(p0_, m, n):
    # Uniform load -> only odd-odd contribute
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0
    return 16.0 * p0_ / (np.pi**2 * m * n)

def Wmn_from_Pmn(Pmn_, D_, a_, b_, m, n):
    return Pmn_ / (D_ * np.pi**4 * ((m/a_)**2 + (n/b_)**2)**2)

# -----------------------------
# 3) Field computation (w-derivatives -> stresses -> von Mises)
# -----------------------------
def compute_stress_fields(Mmax, Nmax, nx_, ny_):
    """
    Compute curvature derivatives and stresses on a (ny_ x nx_) grid.
    Returns x1, x2, sigma1, sigma2, tau12, svm
    """
    D = plate_rigidity(E, nu, t)
    x1 = np.linspace(0.0, a, nx_)
    x2 = np.linspace(0.0, b, ny_)

    # second derivatives of w
    w_x1x1 = np.zeros((ny_, nx_))
    w_x2x2 = np.zeros((ny_, nx_))
    w_x1x2 = np.zeros((ny_, nx_))

    # precompute trig
    sin_mx1 = {m: np.sin(m*np.pi*x1/a) for m in range(1, Mmax+1)}
    cos_mx1 = {m: np.cos(m*np.pi*x1/a) for m in range(1, Mmax+1)}
    sin_nx2 = {n: np.sin(n*np.pi*x2/b) for n in range(1, Nmax+1)}
    cos_nx2 = {n: np.cos(n*np.pi*x2/b) for n in range(1, Nmax+1)}

    for m in range(1, Mmax+1):
        for n in range(1, Nmax+1):
            P = Pmn_uniform(p0, m, n)
            if P == 0.0:
                continue
            W = Wmn_from_Pmn(P, D, a, b, m, n)

            S = np.outer(sin_nx2[n], sin_mx1[m])  # sin(nπx2/b)*sin(mπx1/a)
            C = np.outer(cos_nx2[n], cos_mx1[m])  # cos(nπx2/b)*cos(mπx1/a)

            w_x1x1 += W * (-(m*np.pi/a)**2) * S
            w_x2x2 += W * (-(n*np.pi/b)**2) * S
            w_x1x2 += W * ((m*np.pi/a)*(n*np.pi/b)) * C

    sigma1 = -(E*z/(1.0 - nu**2)) * (w_x1x1 + nu*w_x2x2)
    sigma2 = -(E*z/(1.0 - nu**2)) * (w_x2x2 + nu*w_x1x1)
    tau12  = -(2.0*G*z) * w_x1x2

    svm = np.sqrt(sigma1**2 - sigma1*sigma2 + sigma2**2 + 3.0*tau12**2)
    return x1, x2, sigma1, sigma2, tau12, svm

def max_von_mises(Mmax, Nmax):
    _, _, _, _, _, svm = compute_stress_fields(Mmax, Nmax, nx, ny)
    return float(np.max(svm))

def max_stresses_with_locations(Mmax, Nmax):
    """
    Returns maxima (absolute) of sigma1, sigma2, tau12 and max von Mises + locations.
    Location is based on the discrete evaluation grid (nx, ny).
    """
    x1, x2, sigma1, sigma2, tau12, svm = compute_stress_fields(Mmax, Nmax, nx, ny)

    idx_s1  = np.unravel_index(np.argmax(np.abs(sigma1)), sigma1.shape)
    idx_s2  = np.unravel_index(np.argmax(np.abs(sigma2)), sigma2.shape)
    idx_t12 = np.unravel_index(np.argmax(np.abs(tau12)),  tau12.shape)
    idx_vm  = np.unravel_index(np.argmax(svm),            svm.shape)  # svm >= 0

    out = {
        "sigma1_max_abs": float(np.abs(sigma1[idx_s1])),
        "sigma2_max_abs": float(np.abs(sigma2[idx_s2])),
        "tau12_max_abs":  float(np.abs(tau12[idx_t12])),
        "svm_max":        float(svm[idx_vm]),

        "sigma1_val": float(sigma1[idx_s1]),
        "sigma2_val": float(sigma2[idx_s2]),
        "tau12_val":  float(tau12[idx_t12]),

        "sigma1_loc": (float(x1[idx_s1[1]]),  float(x2[idx_s1[0]])),
        "sigma2_loc": (float(x1[idx_s2[1]]),  float(x2[idx_s2[0]])),
        "tau12_loc":  (float(x1[idx_t12[1]]), float(x2[idx_t12[0]])),
        "svm_loc":    (float(x1[idx_vm[1]]),  float(x2[idx_vm[0]])),
    }
    return out

# -----------------------------
# 4) Build convergence surface for max von Mises
# -----------------------------
print("Computing convergence surface for max von Mises...")
SVMsurf = np.zeros((N_limit, M_limit))

for Mmax in range(1, M_limit+1):
    for Nmax in range(1, N_limit+1):
        SVMsurf[Nmax-1, Mmax-1] = max_von_mises(Mmax, Nmax)

svm_ref = SVMsurf[N_limit-1, M_limit-1]
Err = np.abs(SVMsurf - svm_ref) / (abs(svm_ref) if abs(svm_ref) > 0 else 1.0)

# Find first (M,N) meeting tolerance using budget k=max(M,N)
Mconv = None
Nconv = None
for k in range(1, max(M_limit, N_limit)+1):
    candidates = []

    # candidates where N=k
    for M in range(1, min(M_limit, k)+1):
        N = k
        if N <= N_limit and Err[N-1, M-1] <= rel_tol:
            candidates.append((M, N))

    # candidates where M=k
    for N in range(1, min(N_limit, k)+1):
        M = k
        if M <= M_limit and Err[N-1, M-1] <= rel_tol:
            candidates.append((M, N))

    if candidates:
        # choose smallest (M+N), then M, then N
        Mconv, Nconv = min(candidates, key=lambda tup: (tup[0] + tup[1], tup[0], tup[1]))
        break

print("\n===== Stress convergence (max von Mises) =====")
print(f"Reference max von Mises at (M_limit,N_limit)=({M_limit},{N_limit}) = {svm_ref/1e6:.3f} MPa")
if Mconv is None:
    print(f"NOT converged for rel_tol={rel_tol} within limits.")
else:
    print(f"rel_tol = {rel_tol}")
    print(f"Converged at: Mconv = {Mconv}, Nconv = {Nconv}")
    print(f"max von Mises at (Mconv,Nconv) = {SVMsurf[Nconv-1, Mconv-1]/1e6:.3f} MPa")
    print(f"relative error = {Err[Nconv-1, Mconv-1]:.3e}")

# -----------------------------
# 5) 3D plot of convergence surface + marker
# -----------------------------
M_vals = np.arange(1, M_limit+1)
N_vals = np.arange(1, N_limit+1)
M_mesh, N_mesh = np.meshgrid(M_vals, N_vals)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(M_mesh, N_mesh, SVMsurf/1e6, cmap="plasma", edgecolor="k", linewidth=0.2)

cbar = fig.colorbar(surf, shrink=0.65, aspect=12)
cbar.set_label("max von Mises [MPa]")

ax.set_xlabel("Mmax")
ax.set_ylabel("Nmax")
ax.set_zlabel("max von Mises [MPa]")
ax.set_title(f"Convergence surface of max von Mises (rel_tol={rel_tol:g})")
ax.view_init(elev=28, azim=225)

if Mconv is not None:
    ax.scatter(Mconv, Nconv, SVMsurf[Nconv-1, Mconv-1]/1e6, s=120, color="red", edgecolors="black")

plt.tight_layout()
plt.show()

# -----------------------------
# 6) Choose final truncation for plotting + print max stresses
# -----------------------------
M_use = Mconv if Mconv is not None else M_limit
N_use = Nconv if Nconv is not None else N_limit

res = max_stresses_with_locations(M_use, N_use)

print("\n===== Maximum stresses (top surface, based on grid) =====")
print(f"(M,N)=({M_use},{N_use}), p0={p0/1e3:.1f} kN/m^2, z=t/2={z:.6f} m\n")

print(f"max |sigma1| = {res['sigma1_max_abs']/1e6:.3f} MPa  "
      f"(value={res['sigma1_val']/1e6:.3f} MPa) at (x1,x2)=({res['sigma1_loc'][0]:.4f}, {res['sigma1_loc'][1]:.4f}) m")

print(f"max |sigma2| = {res['sigma2_max_abs']/1e6:.3f} MPa  "
      f"(value={res['sigma2_val']/1e6:.3f} MPa) at (x1,x2)=({res['sigma2_loc'][0]:.4f}, {res['sigma2_loc'][1]:.4f}) m")

print(f"max |tau12|  = {res['tau12_max_abs']/1e6:.3f} MPa  "
      f"(value={res['tau12_val']/1e6:.3f} MPa) at (x1,x2)=({res['tau12_loc'][0]:.4f}, {res['tau12_loc'][1]:.4f}) m")

print(f"max von Mises = {res['svm_max']/1e6:.3f} MPa at (x1,x2)=({res['svm_loc'][0]:.4f}, {res['svm_loc'][1]:.4f}) m")

# -----------------------------
# 7) Final contour plot of von Mises (only required plot)
# -----------------------------
x1, x2, sigma1, sigma2, tau12, svm = compute_stress_fields(M_use, N_use, nx, ny)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, svm/1e6, levels=40)
plt.colorbar(cs, label="von Mises [MPa]")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.title(f"von Mises on top surface using (M,N)=({M_use},{N_use})")
plt.tight_layout()
plt.show()

