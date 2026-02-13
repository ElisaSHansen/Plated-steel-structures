import numpy as np
import matplotlib.pyplot as plt

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

