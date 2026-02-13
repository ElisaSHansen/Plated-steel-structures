import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Compare plate stresses computed by:
# (A) Moment method (Mij -> stresses)
# (B) Direct stress formulas (from screenshot)
# Using same Navier truncation: Mmax=19, Nmax=5
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

Mmax = 19
Nmax = 5

nx = 121
ny = 121

# Top surface coordinate (x3 = z)
z = t/2

# -----------------------------
# 2) Shared Navier helpers
# -----------------------------
def plate_rigidity(E, nu, t):
    return E * t**3 / (12.0 * (1.0 - nu**2))

def Pmn_uniform(p0, m, n):
    # Uniform load => only odd-odd contribute
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0
    return 16.0 * p0 / (np.pi**2 * m * n)

def Wmn(Pmn, D, a, b, m, n):
    return Pmn / (D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2)

def w_second_derivatives_grid(a, b, p0, E, nu, t, Mmax, Nmax, nx=121, ny=121):
    """
    Compute w_x1x1, w_x2x2, w_x1x2 on a grid using Navier series.
    Returns: x1, x2, w_x1x1, w_x2x2, w_x1x2  (each ny x nx)
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

            w_x1x1 += W * (-(m*np.pi/a)**2) * S
            w_x2x2 += W * (-(n*np.pi/b)**2) * S
            w_x1x2 += W * ((m*np.pi/a)*(n*np.pi/b)) * C

    return x1, x2, w_x1x1, w_x2x2, w_x1x2

# -----------------------------
# 3) Method A: Moments -> stresses
# -----------------------------
def stresses_from_moments(w_x1x1, w_x2x2, w_x1x2, E, nu, t):
    """
    Uses:
      M11 = -D (w_x1x1 + nu w_x2x2)
      M22 = -D (w_x2x2 + nu w_x1x1)
      M12 = -D (1-nu) w_x1x2
    Stresses at top surface: sigma = (6/t^2) * M
    """
    D = plate_rigidity(E, nu, t)
    M11 = -D * (w_x1x1 + nu*w_x2x2)
    M22 = -D * (w_x2x2 + nu*w_x1x1)
    M12 = -D * (1-nu) * w_x1x2

    factor = 6.0/(t**2)  # z=t/2 already built-in
    s11 = factor * M11
    s22 = factor * M22
    t12 = factor * M12

    svm = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*t12**2)
    return s11, s22, t12, svm

# -----------------------------
# 4) Method B: Direct stress formulas (your screenshot)
# -----------------------------
def stresses_direct(w_x1x1, w_x2x2, w_x1x2, E, nu, G, z):
    """
    Uses:
      sigma1 = -(E*z/(1-nu^2)) (w_x1x1 + nu*w_x2x2)
      sigma2 = -(E*z/(1-nu^2)) (w_x2x2 + nu*w_x1x1)
      tau12  = -(2*G*z) * w_x1x2
    """
    s11 = -(E*z/(1-nu**2)) * (w_x1x1 + nu*w_x2x2)
    s22 = -(E*z/(1-nu**2)) * (w_x2x2 + nu*w_x1x1)
    t12 = -(2*G*z)         * (w_x1x2)

    svm = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*t12**2)
    return s11, s22, t12, svm

# -----------------------------
# 5) Compute derivatives once, then both stress methods
# -----------------------------
x1, x2, w_x1x1, w_x2x2, w_x1x2 = w_second_derivatives_grid(
    a, b, p0, E, nu, t, Mmax, Nmax, nx=nx, ny=ny
)

# Moment method fields
s11_m, s22_m, t12_m, svm_m = stresses_from_moments(w_x1x1, w_x2x2, w_x1x2, E, nu, t)

# Direct method fields
s11_d, s22_d, t12_d, svm_d = stresses_direct(w_x1x1, w_x2x2, w_x1x2, E, nu, G, z)

# -----------------------------
# 6) Print comparison statistics
# -----------------------------
print("\n=== FIELD COMPARISON (Moment vs Direct) ===")
print("Max abs difference sigma11 [MPa]:", np.max(np.abs(s11_m - s11_d))/1e6)
print("Max abs difference sigma22 [MPa]:", np.max(np.abs(s22_m - s22_d))/1e6)
print("Max abs difference tau12   [MPa]:", np.max(np.abs(t12_m - t12_d))/1e6)

mask = np.abs(t12_m) > 1e-12
if np.any(mask):
    ratio = np.median(t12_d[mask] / t12_m[mask])
    print("Median ratio tau12_direct / tau12_moment:", ratio)

# -----------------------------
# 7) Maxima + locations (for each method)
# -----------------------------
def max_abs_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    return field[iy, ix], x1[ix], x2[iy]

def max_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(field), field.shape)
    return field[iy, ix], x1[ix], x2[iy]

def print_maxima(tag, s11, s22, t12, svm, x1, x2):
    s11_max, xs11, ys11 = max_abs_with_location(s11, x1, x2)
    s22_max, xs22, ys22 = max_abs_with_location(s22, x1, x2)
    t12_max, xt12, yt12 = max_abs_with_location(t12, x1, x2)
    svm_max, xvm, yvm   = max_with_location(svm, x1, x2)

    print(f"\n=== TOP SURFACE MAXIMA ({tag}) ===")
    print(f"Max |sigma11| = {abs(s11_max)/1e6:.3f} MPa at x1={xs11:.3f} m, x2={ys11:.3f} m")
    print(f"Max |sigma22| = {abs(s22_max)/1e6:.3f} MPa at x1={xs22:.3f} m, x2={ys22:.3f} m")
    print(f"Max |tau12|   = {abs(t12_max)/1e6:.3f} MPa at x1={xt12:.3f} m, x2={yt12:.3f} m")
    print(f"Max von Mises = {svm_max/1e6:.3f} MPa at x1={xvm:.3f} m, x2={yvm:.3f} m")

print_maxima("Moment method", s11_m, s22_m, t12_m, svm_m, x1, x2)
print_maxima("Direct formulas", s11_d, s22_d, t12_d, svm_d, x1, x2)

# -----------------------------
# 8) Plot von Mises (Moment vs Direct)
# -----------------------------
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, svm_m/1e6, levels=40)
plt.colorbar(cs, label="von Mises [MPa]")
plt.title("von Mises (Moment method)")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, svm_d/1e6, levels=40)
plt.colorbar(cs, label="von Mises [MPa]")
plt.title("von Mises (Direct formulas)")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.tight_layout()
plt.show()
