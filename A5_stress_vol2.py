import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# A5 – Navier (SSSS) plate with RECTANGULAR PATCH LOAD (centered)
#
# THIS VERSION: CONVERGENCE STUDY FOR STRESSES ONLY
#   - Builds convergence surfaces for:
#       max|sigma11|, max|sigma22|, max|tau12|, max(von Mises)
#     vs (M,N), where stresses come from w-derivatives.
#   - Uses reference at (M_limit, N_limit)
#   - Finds first (M,N) meeting rel_tol for EACH stress metric
#   - Prints convergence points + values
#   - Uses the converged (M,N) for final contour plots of von Mises
#
# Notes:
#   - For patch loads, odd-only shortcut is optional (ONLY_ODD).
#   - Stress convergence generally needs higher (M,N) than deflection.
# ============================================================

# -----------------------------
# 1) Inputs (A5)
# -----------------------------
P_total = 20e3       # N (total force)
a = 2.8              # m
b = 0.67             # m

# Patch dimensions (u along x1, v along x2)
u = 0.250            # m (250 mm) along x1
v = 0.200            # m (200 mm) along x2

# Centered patch location
xi1 = a / 2.0
xi2 = b / 2.0

# Material / thickness
t  = 0.0065          # m
E  = 210e9           # Pa
nu = 0.30
G  = E/(2*(1+nu))    # Pa
z  = t/2.0           # top surface

# Optional speed-up: for perfectly centered symmetric patch, only odd-odd contribute.
ONLY_ODD = False

# -----------------------------
# 2) Convergence settings (STRESSES)
# -----------------------------
M_limit = 50         # scan up to this M
N_limit = 50         # scan up to this N
rel_tol = 1e-4       # relative error vs reference at (M_limit, N_limit)

# Grid used to evaluate maxima during convergence scan (trade speed/accuracy)
nx_conv = 81
ny_conv = 81

# Grid for final plotting (after converged (M,N) found)
nx_plot = 161
ny_plot = 161

# -----------------------------
# 3) Derived
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
# 4) Robust Pmn by integral definition
# -----------------------------
def int_sin(m, L, x1, x2):
    """I = ∫_{x1}^{x2} sin(m*pi*x/L) dx in closed form."""
    k = m*np.pi/L
    return (-np.cos(k*x2) + np.cos(k*x1)) / k

def Pmn_patch(m, n):
    """
    Pmn = (4/(ab)) ∬_patch p0 sin(mπx1/a) sin(nπx2/b) dx2 dx1
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
# 5) Compute stresses for a given (M,N) and return maxima
# -----------------------------
def stress_maxima(Mmax, Nmax, nx, ny):
    """
    Computes stress fields on (ny x nx) grid and returns:
      max|sigma11|, max|sigma22|, max|tau12|, max(von Mises)
    """
    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)

    w_xx = np.zeros((ny, nx), dtype=float)
    w_yy = np.zeros((ny, nx), dtype=float)
    w_xy = np.zeros((ny, nx), dtype=float)

    if ONLY_ODD:
        m_list = range(1, Mmax + 1, 2)
        n_list = range(1, Nmax + 1, 2)
    else:
        m_list = range(1, Mmax + 1)
        n_list = range(1, Nmax + 1)

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

            S = np.outer(sin_ny[n], sin_mx[m])
            C = np.outer(cos_ny[n], cos_mx[m])

            w_xx += W * (-(m*np.pi/a)**2) * S
            w_yy += W * (-(n*np.pi/b)**2) * S
            w_xy += W * ((m*np.pi/a)*(n*np.pi/b)) * C

    sigma11 = -(E*z/(1.0-nu**2)) * (w_xx + nu*w_yy)
    sigma22 = -(E*z/(1.0-nu**2)) * (w_yy + nu*w_xx)
    tau12   = -(2.0*G*z) * w_xy
    svm     = np.sqrt(sigma11**2 - sigma11*sigma22 + sigma22**2 + 3.0*tau12**2)

    s11_max = float(np.max(np.abs(sigma11)))
    s22_max = float(np.max(np.abs(sigma22)))
    t12_max = float(np.max(np.abs(tau12)))
    svm_max = float(np.max(svm))
    return s11_max, s22_max, t12_max, svm_max

# -----------------------------
# 6) Build convergence surfaces for stress maxima
# -----------------------------
print("Computing convergence surfaces for stress maxima...")
S11surf = np.zeros((N_limit, M_limit), dtype=float)
S22surf = np.zeros((N_limit, M_limit), dtype=float)
T12surf = np.zeros((N_limit, M_limit), dtype=float)
SVMsurf = np.zeros((N_limit, M_limit), dtype=float)

for M in range(1, M_limit + 1):
    for N in range(1, N_limit + 1):
        s11m, s22m, t12m, svmm = stress_maxima(M, N, nx_conv, ny_conv)
        S11surf[N-1, M-1] = s11m
        S22surf[N-1, M-1] = s22m
        T12surf[N-1, M-1] = t12m
        SVMsurf[N-1, M-1] = svmm

# Reference values at (M_limit, N_limit)
s11_ref = float(S11surf[N_limit-1, M_limit-1])
s22_ref = float(S22surf[N_limit-1, M_limit-1])
t12_ref = float(T12surf[N_limit-1, M_limit-1])
svm_ref = float(SVMsurf[N_limit-1, M_limit-1])

# Relative error surfaces
def rel_err_surface(S, Sref):
    denom = abs(Sref) if abs(Sref) > 0 else 1.0
    return np.abs(S - Sref) / denom

Err_s11 = rel_err_surface(S11surf, s11_ref)
Err_s22 = rel_err_surface(S22surf, s22_ref)
Err_t12 = rel_err_surface(T12surf, t12_ref)
Err_svm = rel_err_surface(SVMsurf, svm_ref)

# -----------------------------
# 7) Find convergence point (Mconv,Nconv) for each metric
# -----------------------------
def find_convergence_point(Err, M_lim, N_lim, tol):
    Mconv = None
    Nconv = None
    for k in range(1, max(M_lim, N_lim) + 1):
        candidates = []

        # N = k
        for M in range(1, min(M_lim, k) + 1):
            N = k
            if N <= N_lim and Err[N-1, M-1] <= tol:
                candidates.append((M, N))

        # M = k
        for N in range(1, min(N_lim, k) + 1):
            M = k
            if M <= M_lim and Err[N-1, M-1] <= tol:
                candidates.append((M, N))

        if candidates:
            Mconv, Nconv = min(candidates, key=lambda t: (t[0] + t[1], t[0], t[1]))
            break
    return Mconv, Nconv

M_s11, N_s11 = find_convergence_point(Err_s11, M_limit, N_limit, rel_tol)
M_s22, N_s22 = find_convergence_point(Err_s22, M_limit, N_limit, rel_tol)
M_t12, N_t12 = find_convergence_point(Err_t12, M_limit, N_limit, rel_tol)
M_svm, N_svm = find_convergence_point(Err_svm, M_limit, N_limit, rel_tol)

print("\n===== STRESS CONVERGENCE (max over plate) =====")
print(f"Reference at (M_limit,N_limit)=({M_limit},{N_limit}), ONLY_ODD={ONLY_ODD}, rel_tol={rel_tol}\n")

print(f"max|sigma11| ref = {s11_ref/1e6:.3f} MPa")
if M_s11 is None:
    print("  -> NOT converged within limits.")
else:
    print(f"  -> converged at (M,N)=({M_s11},{N_s11}), value={S11surf[N_s11-1, M_s11-1]/1e6:.3f} MPa, relerr={Err_s11[N_s11-1, M_s11-1]:.3e}")

print(f"\nmax|sigma22| ref = {s22_ref/1e6:.3f} MPa")
if M_s22 is None:
    print("  -> NOT converged within limits.")
else:
    print(f"  -> converged at (M,N)=({M_s22},{N_s22}), value={S22surf[N_s22-1, M_s22-1]/1e6:.3f} MPa, relerr={Err_s22[N_s22-1, M_s22-1]:.3e}")

print(f"\nmax|tau12| ref = {t12_ref/1e6:.3f} MPa")
if M_t12 is None:
    print("  -> NOT converged within limits.")
else:
    print(f"  -> converged at (M,N)=({M_t12},{N_t12}), value={T12surf[N_t12-1, M_t12-1]/1e6:.3f} MPa, relerr={Err_t12[N_t12-1, M_t12-1]:.3e}")

print(f"\nmax(von Mises) ref = {svm_ref/1e6:.3f} MPa")
if M_svm is None:
    print("  -> NOT converged within limits.")
else:
    print(f"  -> converged at (M,N)=({M_svm},{N_svm}), value={SVMsurf[N_svm-1, M_svm-1]/1e6:.3f} MPa, relerr={Err_svm[N_svm-1, M_svm-1]:.3e}")

# Choose a final (M,N) to use for plotting:
# safest is to use the strictest requirement (largest "budget" max(M,N) among the converged ones),
# otherwise fall back to (M_limit,N_limit).
def budget(pair):
    if pair[0] is None or pair[1] is None:
        return -1
    return max(pair[0], pair[1])

pairs = [(M_s11, N_s11), (M_s22, N_s22), (M_t12, N_t12), (M_svm, N_svm)]
best_budget = max(budget(p) for p in pairs)

if best_budget < 0:
    M_use, N_use = M_limit, N_limit
    print(f"\nUsing fallback (M,N)=({M_use},{N_use}) (no metric converged within limits).")
else:
    # pick the pair with max budget; if ties, prefer von Mises, then smallest M+N
    # (von Mises is usually the governing one)
    candidates = []
    for p in pairs:
        if budget(p) == best_budget:
            candidates.append(p)

    # tie-break: prefer von Mises point if it is in candidates
    if (M_svm, N_svm) in candidates:
        M_use, N_use = M_svm, N_svm
    else:
        M_use, N_use = min(candidates, key=lambda t: (t[0] + t[1], t[0], t[1]))

    print(f"\nUsing (M,N)=({M_use},{N_use}) for final field plots (governed by strictest converged metric).")

# -----------------------------
# 8) Final fields at (M_use, N_use) for locations + von Mises contour
# -----------------------------
def compute_full_fields(Mmax, Nmax, nx, ny):
    x1 = np.linspace(0.0, a, nx)
    x2 = np.linspace(0.0, b, ny)

    w    = np.zeros((ny, nx), dtype=float)
    w_xx = np.zeros((ny, nx), dtype=float)
    w_yy = np.zeros((ny, nx), dtype=float)
    w_xy = np.zeros((ny, nx), dtype=float)

    if ONLY_ODD:
        m_list = range(1, Mmax + 1, 2)
        n_list = range(1, Nmax + 1, 2)
    else:
        m_list = range(1, Mmax + 1)
        n_list = range(1, Nmax + 1)

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

            S = np.outer(sin_ny[n], sin_mx[m])
            C = np.outer(cos_ny[n], cos_mx[m])

            w    += W * S
            w_xx += W * (-(m*np.pi/a)**2) * S
            w_yy += W * (-(n*np.pi/b)**2) * S
            w_xy += W * ((m*np.pi/a)*(n*np.pi/b)) * C

    sigma11 = -(E*z/(1.0-nu**2)) * (w_xx + nu*w_yy)
    sigma22 = -(E*z/(1.0-nu**2)) * (w_yy + nu*w_xx)
    tau12   = -(2.0*G*z) * w_xy
    svm     = np.sqrt(sigma11**2 - sigma11*sigma22 + sigma22**2 + 3.0*tau12**2)

    return x1, x2, w, sigma11, sigma22, tau12, svm

x1p, x2p, wp, s11p, s22p, t12p, svmp = compute_full_fields(M_use, N_use, nx_plot, ny_plot)

def max_abs_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(np.abs(field)), field.shape)
    return float(np.abs(field[iy, ix])), float(field[iy, ix]), float(x1[ix]), float(x2[iy])

def max_with_location(field, x1, x2):
    iy, ix = np.unravel_index(np.argmax(field), field.shape)
    return float(field[iy, ix]), float(x1[ix]), float(x2[iy])

s11_max_abs, s11_val, x_s11, y_s11 = max_abs_with_location(s11p, x1p, x2p)
s22_max_abs, s22_val, x_s22, y_s22 = max_abs_with_location(s22p, x1p, x2p)
t12_max_abs, t12_val, x_t12, y_t12 = max_abs_with_location(t12p, x1p, x2p)
svm_max, x_vm, y_vm = max_with_location(svmp, x1p, x2p)

print("\n===== MAX STRESSES (using chosen M_use,N_use) =====")
print(f"(M,N)=({M_use},{N_use}) on grid ({nx_plot}x{ny_plot})")
print(f"max |sigma11| = {s11_max_abs/1e6:.3f} MPa (value={s11_val/1e6:.3f} MPa) at (x1,x2)=({x_s11:.3f},{y_s11:.3f}) m")
print(f"max |sigma22| = {s22_max_abs/1e6:.3f} MPa (value={s22_val/1e6:.3f} MPa) at (x1,x2)=({x_s22:.3f},{y_s22:.3f}) m")
print(f"max |tau12|   = {t12_max_abs/1e6:.3f} MPa (value={t12_val/1e6:.3f} MPa) at (x1,x2)=({x_t12:.3f},{y_t12:.3f}) m")
print(f"max von Mises = {svm_max/1e6:.3f} MPa at (x1,x2)=({x_vm:.3f},{y_vm:.3f}) m")

# -----------------------------
# 9) Plot: von Mises contour (top surface)
# -----------------------------
X1, X2 = np.meshgrid(x1p, x2p)

plt.figure(figsize=(8, 5))
cs = plt.contourf(X1, X2, svmp/1e6, levels=40)
plt.colorbar(cs, label="von Mises [MPa]")
plt.scatter([x_vm], [y_vm], marker="x", s=90, color="black", label="Max von Mises")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.legend()
plt.title(f"von Mises (top surface, SSSS Navier) – patch load\n(M,N)=({M_use},{N_use}), ONLY_ODD={ONLY_ODD}")
plt.tight_layout()
plt.show()
