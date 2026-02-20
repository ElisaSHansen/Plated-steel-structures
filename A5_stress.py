import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Inputs (Project A5)
# -----------------------------
P  = 20e3          # N (total kraft)
a  = 2.8           # m
b  = 0.67          # m
v  = 0.200         # m
u  = 0.250         # m

# Lastsenter i senter av platen
xi1 = a / 2.0
xi2 = b / 2.0

t  = 0.0065        # m
E  = 210e9         # Pa
nu = 0.30          # -

Mmax = 49
Nmax = 29

# Grid for contour/stress
nx = 121
ny = 121

# -----------------------------
# 2) Plate theory
# -----------------------------
def plate_rigidity(E, nu, t):
    return E * t**3 / (12.0 * (1.0 - nu**2))

def Pmn_patch(P, a, b, u, v, xi1, xi2, m, n):
    # For senterlast blir partall null -> hopp over for fart
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0

    p0 = P / (u * v)  # N/m^2
    return (
        16.0 * p0 / (np.pi**2 * m * n)
        * np.sin(m * np.pi * xi1 / a) * np.sin(m * np.pi * u / (2.0 * a))
        * np.sin(n * np.pi * xi2 / b) * np.sin(n * np.pi * v / (2.0 * b))
    )

def Wmn(Pmn, D, a, b, m, n):
    return Pmn / (D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2)

# -----------------------------
# 3) Stress fields on TOP surface
# -----------------------------
def stress_fields_top_surface_A5(a, b, t, E, nu, P, u, v, xi1, xi2, Mmax, Nmax, nx=121, ny=121):
    D = plate_rigidity(E, nu, t)

    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)

    w_x1x1 = np.zeros((ny, nx), dtype=float)
    w_x2x2 = np.zeros((ny, nx), dtype=float)
    w_x1x2 = np.zeros((ny, nx), dtype=float)

    # Precompute sin/cos ONLY for odd indices (stor hastighetsgevinst)
    sin_mx1 = {}
    cos_mx1 = {}
    for m in range(1, Mmax + 1, 2):
        sin_mx1[m] = np.sin(m * np.pi * x1 / a)
        cos_mx1[m] = np.cos(m * np.pi * x1 / a)

    sin_nx2 = {}
    cos_nx2 = {}
    for n in range(1, Nmax + 1, 2):
        sin_nx2[n] = np.sin(n * np.pi * x2 / b)
        cos_nx2[n] = np.cos(n * np.pi * x2 / b)

    for m in range(1, Mmax + 1, 2):
        for n in range(1, Nmax + 1, 2):
            Pmn = Pmn_patch(P, a, b, u, v, xi1, xi2, m, n)
            if Pmn == 0.0:
                continue

            W = Wmn(Pmn, D, a, b, m, n)

            S = np.outer(sin_nx2[n], sin_mx1[m])
            C = np.outer(cos_nx2[n], cos_mx1[m])

            w_x1x1 += W * (-(m*np.pi/a)**2) * S
            w_x2x2 += W * (-(n*np.pi/b)**2) * S
            w_x1x2 += W * ((m*np.pi/a)*(n*np.pi/b)) * C

    # Moments
    M11 = -D * (w_x1x1 + nu * w_x2x2)
    M22 = -D * (w_x2x2 + nu * w_x1x1)
    M12 = -D * (1.0 - nu) * w_x1x2

    # Top surface stresses: z = +t/2 => factor = 6/t^2
    factor = 6.0 / (t**2)
    sigma11 = factor * M11
    sigma22 = factor * M22
    tau12   = factor * M12

    # von Mises (plane stress)
    sigma_vm = np.sqrt(sigma11**2 - sigma11*sigma22 + sigma22**2 + 3.0*tau12**2)

    return x1, x2, sigma11, sigma22, tau12, sigma_vm

x1, x2, s11, s22, t12, svm = stress_fields_top_surface_A5(
    a, b, t, E, nu, P, u, v, xi1, xi2, Mmax, Nmax, nx=nx, ny=ny
)

# -----------------------------
# 4) Maxima + location
# -----------------------------
def max_abs_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    return field[iy, ix], x1[ix], x2[iy]

def max_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(field), field.shape)
    return field[iy, ix], x1[ix], x2[iy]

s11_max, x_s11, y_s11 = max_abs_with_location(s11, x1, x2)
s22_max, x_s22, y_s22 = max_abs_with_location(s22, x1, x2)
t12_max, x_t12, y_t12 = max_abs_with_location(t12, x1, x2)
svm_max, x_vm,  y_vm  = max_with_location(svm, x1, x2)

print("\n=== TOP SURFACE MAXIMA (Navier, A5 lokal last) ===")
print(f"Mmax={Mmax}, Nmax={Nmax}")
print(f"Max |sigma11| = {abs(s11_max)/1e6:.3f} MPa at x1={x_s11:.3f} m, x2={y_s11:.3f} m")
print(f"Max |sigma22| = {abs(s22_max)/1e6:.3f} MPa at x1={x_s22:.3f} m, x2={y_s22:.3f} m")
print(f"Max |tau12|   = {abs(t12_max)/1e6:.3f} MPa at x1={x_t12:.3f} m, x2={y_t12:.3f} m")
print(f"Max von Mises = {svm_max/1e6:.3f} MPa at x1={x_vm:.3f} m, x2={y_vm:.3f} m")

# -----------------------------
# 5) Contour plot of von Mises + mark maximum
# -----------------------------
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, svm/1e6, levels=40)
plt.colorbar(cs, label="von Mises stress [MPa]")
plt.scatter([x_vm], [y_vm], marker="x", s=90, color="black", label="Max von Mises")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.legend()
plt.tight_layout()
plt.show()
