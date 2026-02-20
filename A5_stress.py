import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# A5 – Navier (SSSS) plate with RECTANGULAR PATCH LOAD (centered)
# Robust version:
#   - Uses the *definition* of P_mn from the Fourier integral (no risk of missing factors)
#   - Uses stresses directly from w-derivatives (same as you did in A4):
#       sigma11 = -(E*z/(1-nu^2)) (w_xx + nu*w_yy)
#       sigma22 = -(E*z/(1-nu^2)) (w_yy + nu*w_xx)
#       tau12   = -(2*G*z) w_xy
#   - Does NOT rely on "odd-only" shortcut (you can toggle it on safely if centered)
#
# Outputs:
#   - w(center)
#   - max |sigma11|, max |sigma22|, max |tau12|, max von Mises + locations
#   - contour plots of w and von Mises
# ============================================================

# -----------------------------
# 1) Inputs (A5)
# -----------------------------
P_total = 20e3       # N (total force)
a = 2.8              # m
b = 0.67             # m

# Patch dimensions (u along x1, v along x2)
u = 0.250            # m  (250 mm) along x1
v = 0.200            # m  (200 mm) along x2

# Centered patch location
xi1 = a / 2.0
xi2 = b / 2.0

# Material / thickness
t  = 0.0065          # m
E  = 210e9           # Pa
nu = 0.30
G  = E/(2*(1+nu))    # Pa
z  = t/2.0           # top surface

# Truncation
Mmax = 49
Nmax = 29

# Evaluation grid for plots/max search
nx = 161
ny = 161

# Optional speed-up: for perfectly centered symmetric patch, only odd-odd contribute.
# Keep False unless you are 100% sure everything is centered and consistent.
ONLY_ODD = False

# -----------------------------
# 2) Derived
# -----------------------------
D = E * t**3 / (12.0 * (1.0 - nu**2))

A_patch = u * v
p0 = P_total / A_patch  # N/m^2 uniform pressure on patch

xL1, xL2 = xi1 - u/2.0, xi1 + u/2.0
yL1, yL2 = xi2 - v/2.0, xi2 + v/2.0

print("=== Load sanity check ===")
print(f"A_patch = {A_patch:.6f} m^2")
print(f"p0 = {p0:.3f} N/m^2  (= {p0/1e3:.3f} kN/m^2 = {p0/1e6:.3f} MPa)")
print(f"P_check = p0*A_patch = {p0*A_patch:.3f} N")
print("=========================\n")

# -----------------------------
# 3) Robust Pmn by integral definition
# -----------------------------
def int_sin(m, L, x1, x2):
    """
    I = ∫_{x1}^{x2} sin(m*pi*x/L) dx
      = [-cos(m*pi*x/L) / (m*pi/L)]_{x1}^{x2}
    """
    k = m*np.pi/L
    return (-np.cos(k*x2) + np.cos(k*x1)) / k

def Pmn_patch(m, n):
    """
    Navier load coefficient for SSSS plate:
      p(x1,x2) = Σ Pmn sin(mπx1/a) sin(nπx2/b)
      Pmn = (4/(ab)) ∬_patch p0 sin(mπx1/a) sin(nπx2/b) dx2 dx1

    Since patch load is separable:
      ∬ = (∫ sin(mπx1/a) dx1)(∫ sin(nπx2/b) dx2)
    """
    if ONLY_ODD and ((m % 2 == 0) or (n % 2 == 0)):
        return 0.0

    Ix = int_sin(m, a, xL1, xL2)
    Iy = int_sin(n, b, yL1, yL2)
    return (4.0/(a*b)) * p0 * Ix * Iy

def Wmn(m, n):
    denom = D * np.pi**4 * ((m/a)**2 + (n/b)**2)**2
    return Pmn_patch(m, n) / denom

# -----------------------------
# 4) Compute fields (w and derivatives)
# -----------------------------
def compute_fields(Mmax, Nmax, nx, ny):
    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)

    w    = np.zeros((ny, nx), dtype=float)
    w_xx = np.zeros((ny, nx), dtype=float)
    w_yy = np.zeros((ny, nx), dtype=float)
    w_xy = np.zeros((ny, nx), dtype=float)

    # Decide index lists
    if ONLY_ODD:
        m_list = range(1, Mmax+1, 2)
        n_list = range(1, Nmax+1, 2)
    else:
        m_list = range(1, Mmax+1)
        n_list = range(1, Nmax+1)

    # Precompute trig
    sin_mx = {m: np.sin(m*np.pi*x1/a) for m in m_list}
    cos_mx = {m: np.cos(m*np.pi*x1/a) for m in m_list}
    sin_ny = {n: np.sin(n*np.pi*x2/b) for n in n_list}
    cos_ny = {n: np.cos(n*np.pi*x2/b) for n in n_list}

    for m in m_list:
        for n in n_list:
            Pmn = Pmn_patch(m, n)
            if Pmn == 0.0:
                continue
            W = Wmn(m, n)

            S = np.outer(sin_ny[n], sin_mx[m])  # sin(nπx2/b)*sin(mπx1/a)
            C = np.outer(cos_ny[n], cos_mx[m])  # cos(nπx2/b)*cos(mπx1/a)

            w    += W * S
            w_xx += W * (-(m*np.pi/a)**2) * S
            w_yy += W * (-(n*np.pi/b)**2) * S
            w_xy += W * ((m*np.pi/a)*(n*np.pi/b)) * C

    # Stresses at top surface (directly from w-derivatives)
    sigma11 = -(E*z/(1.0-nu**2)) * (w_xx + nu*w_yy)
    sigma22 = -(E*z/(1.0-nu**2)) * (w_yy + nu*w_xx)
    tau12   = -(2.0*G*z) * w_xy

    sigma_vm = np.sqrt(sigma11**2 - sigma11*sigma22 + sigma22**2 + 3.0*tau12**2)

    return x1, x2, w, sigma11, sigma22, tau12, sigma_vm

x1, x2, w, s11, s22, t12, svm = compute_fields(Mmax, Nmax, nx, ny)

# -----------------------------
# 5) Values in center
# -----------------------------
def bilinear_at(field, x, y, xp, yp):
    ix = np.searchsorted(x, xp) - 1
    iy = np.searchsorted(y, yp) - 1
    ix = np.clip(ix, 0, len(x)-2)
    iy = np.clip(iy, 0, len(y)-2)
    x0, x1_ = x[ix], x[ix+1]
    y0, y1_ = y[iy], y[iy+1]
    fx = (xp-x0)/(x1_-x0)
    fy = (yp-y0)/(y1_-y0)
    f00 = field[iy, ix]
    f10 = field[iy, ix+1]
    f01 = field[iy+1, ix]
    f11 = field[iy+1, ix+1]
    return (1-fx)*(1-fy)*f00 + fx*(1-fy)*f10 + (1-fx)*fy*f01 + fx*fy*f11

w_center = bilinear_at(w, x1, x2, a/2.0, b/2.0)

# -----------------------------
# 6) Maxima + location
# -----------------------------
def max_abs_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    return float(np.abs(field[iy, ix])), float(field[iy, ix]), float(x1[ix]), float(x2[iy])

def max_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(field), field.shape)
    return float(field[iy, ix]), float(x1[ix]), float(x2[iy])

s11_max_abs, s11_val, x_s11, y_s11 = max_abs_with_location(s11, x1, x2)
s22_max_abs, s22_val, x_s22, y_s22 = max_abs_with_location(s22, x1, x2)
t12_max_abs, t12_val, x_t12, y_t12 = max_abs_with_location(t12, x1, x2)
svm_max, x_vm, y_vm = max_with_location(svm, x1, x2)

print("=== A5 (Navier SSSS) PATCH LOAD RESULTS ===")
print(f"Mmax={Mmax}, Nmax={Nmax}, ONLY_ODD={ONLY_ODD}")
print(f"Patch: u={u*1e3:.0f} mm (x1), v={v*1e3:.0f} mm (x2), centered")
print(f"Total force P = {P_total/1e3:.1f} kN  =>  p0 = {p0/1e6:.3f} MPa")
print(f"w(center) = {w_center*1e3:.3f} mm")
print("")
print(f"max |sigma11| = {s11_max_abs/1e6:.3f} MPa (value={s11_val/1e6:.3f} MPa) at (x1,x2)=({x_s11:.3f},{y_s11:.3f}) m")
print(f"max |sigma22| = {s22_max_abs/1e6:.3f} MPa (value={s22_val/1e6:.3f} MPa) at (x1,x2)=({x_s22:.3f},{y_s22:.3f}) m")
print(f"max |tau12|   = {t12_max_abs/1e6:.3f} MPa (value={t12_val/1e6:.3f} MPa) at (x1,x2)=({x_t12:.3f},{y_t12:.3f}) m")
print(f"max von Mises = {svm_max/1e6:.3f} MPa at (x1,x2)=({x_vm:.3f},{y_vm:.3f}) m")
print("==========================================")

# -----------------------------
# 7) Plots: deflection and von Mises
# -----------------------------
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, w*1e3, levels=40)
plt.colorbar(cs, label="w [mm]")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.title("Deflection w (SSSS Navier) – patch load")
plt.scatter([a/2.0], [b/2.0], s=40, marker="x", color="black", label="Center")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, svm/1e6, levels=40)
plt.colorbar(cs, label="von Mises [MPa]")
plt.scatter([x_vm], [y_vm], marker="x", s=90, color="black", label="Max von Mises")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.legend()
plt.title("von Mises (top surface, SSSS Navier) – patch load")
plt.tight_layout()
plt.show()
