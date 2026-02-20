import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Project A.5 – Energy method (Rayleigh–Ritz) for a "concentrated" load
# Load: Total force P_total = 20 kN uniformly distributed over a patch
#       200 mm x 250 mm at the CENTER of the plate.
#
# Plate: a x b, thickness t, isotropic (E, nu), linear elastic.
#
# Boundary condition cases:
#   1) SSSS  (simply supported on all edges)
#   2) CCSS  (more appropriate: clamped on x1=0,a ; simply supported on x2=0,b)
#
# Outputs:
#   - w(center)
#   - max |sigma1|, max |sigma2|, max |tau12|, max von Mises (top surface)
#   - ONLY plots: deflection and von Mises (for both BC cases)
#
# IMPORTANT: Patch load integration is done on the PATCH interval directly
#            (not by masking global Gauss points) to guarantee total load = 20 kN.
# ============================================================

# -----------------------------
# 1) Inputs
# -----------------------------
a  = 2.8           # m (x1-direction)
b  = 0.67          # m (x2-direction)
t  = 0.0065        # m
E  = 210e9         # Pa
nu = 0.30          # -
G  = E/(2*(1+nu))  # Pa
D  = E*t**3/(12.0*(1.0-nu**2))

# top surface coordinate for stresses
z = t/2.0

# Load (A.5)
P_total = 20e3        # N
patch_x = 0.250       # m (250 mm) along x1
patch_y = 0.200       # m (200 mm) along x2
A_patch = patch_x * patch_y
p0 = P_total / A_patch   # N/m^2  (uniform pressure on patch)

# Patch location (centered)
x1c, x2c = a/2.0, b/2.0
xL1, xL2 = x1c - patch_x/2.0, x1c + patch_x/2.0
yL1, yL2 = x2c - patch_y/2.0, x2c + patch_y/2.0

# Truncation (start moderate; increase if needed)
M = 20
N = 20

# Quadrature for ENERGY integrals over whole plate
# Rule of thumb: >= 2*M and 2*N; for stresses, 3x is safer
nqx = max(60, 3*M)
nqy = max(60, 3*N)

# Quadrature for LOAD integrals over PATCH only
nq_patch_x = max(60, 3*M)
nq_patch_y = max(60, 3*N)

# Plot grid (for contours + maxima search)
nx = 201
ny = 201
xg = np.linspace(0.0, a, nx)
yg = np.linspace(0.0, b, ny)
Xg, Yg = np.meshgrid(xg, yg)


# ============================================================
# 2) Basis functions (Rayleigh–Ritz)
# ============================================================
def basis_SS(k, L, x):
    return np.sin(k*np.pi*x/L)

def basis_SS_d1(k, L, x):
    return (k*np.pi/L) * np.cos(k*np.pi*x/L)

def basis_SS_d2(k, L, x):
    return -(k*np.pi/L)**2 * np.sin(k*np.pi*x/L)

def basis_CC(k, L, x):
    # admissible clamped-clamped: w=0 and dw/dx=0 at x=0,L
    return np.sin(np.pi*x/L) * np.sin(k*np.pi*x/L)

def basis_CC_d1(k, L, x):
    s1 = np.sin(np.pi*x/L)
    c1 = np.cos(np.pi*x/L)
    sk = np.sin(k*np.pi*x/L)
    ck = np.cos(k*np.pi*x/L)
    return (np.pi/L)*c1*sk + (k*np.pi/L)*s1*ck

def basis_CC_d2(k, L, x):
    s1 = np.sin(np.pi*x/L)
    c1 = np.cos(np.pi*x/L)
    sk = np.sin(k*np.pi*x/L)
    ck = np.cos(k*np.pi*x/L)
    term1 = (np.pi/L) * (-(np.pi/L)*s1*sk + c1*(k*np.pi/L)*ck)
    term2 = (k*np.pi/L) * ((np.pi/L)*c1*ck + s1*(-k*np.pi/L)*sk)
    return term1 + term2


# ============================================================
# 3) Quadrature utilities
# ============================================================
def gauss_interval(n, x1, x2):
    """Gauss-Legendre nodes/weights on [x1,x2]."""
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5*(xi + 1.0)*(x2 - x1) + x1
    w = 0.5*(x2 - x1)*wi
    return x, w

def gauss_on_0L(n, L):
    return gauss_interval(n, 0.0, L)


# ============================================================
# 4) Rayleigh–Ritz solver with PATCH LOAD
# ============================================================
def ritz_solve_patch(M, N, bc_x1="SS", bc_x2="SS"):
    """
    Solve for Wmn in:
      w(x1,x2)=sum_m sum_n Wmn X_m1(x1) X_n2(x2)
    using:
      U = 0.5 D ∬ [ w_xx^2 + w_yy^2 + 2ν w_xx w_yy + 2(1-ν) w_xy^2 ] dA
      V = ∬_{patch} p0 w dA

    Returns:
      W (MxN), basis_info
    """

    # Choose basis per direction
    if bc_x1 == "SS":
        f0x, f1x, f2x = basis_SS, basis_SS_d1, basis_SS_d2
    elif bc_x1 == "CC":
        f0x, f1x, f2x = basis_CC, basis_CC_d1, basis_CC_d2
    else:
        raise ValueError("bc_x1 must be 'SS' or 'CC'")

    if bc_x2 == "SS":
        f0y, f1y, f2y = basis_SS, basis_SS_d1, basis_SS_d2
    elif bc_x2 == "CC":
        f0y, f1y, f2y = basis_CC, basis_CC_d1, basis_CC_d2
    else:
        raise ValueError("bc_x2 must be 'SS' or 'CC'")

    # ---- Whole-plate quadrature for bending energy ----
    xq, wx = gauss_on_0L(nqx, a)
    yq, wy = gauss_on_0L(nqy, b)

    # Basis evaluated on whole-plate quadrature
    X0 = np.zeros((M, nqx)); X1 = np.zeros((M, nqx)); X2 = np.zeros((M, nqx))
    for m in range(1, M+1):
        X0[m-1,:] = f0x(m, a, xq)
        X1[m-1,:] = f1x(m, a, xq)
        X2[m-1,:] = f2x(m, a, xq)

    Y0 = np.zeros((N, nqy)); Y1 = np.zeros((N, nqy)); Y2 = np.zeros((N, nqy))
    for n in range(1, N+1):
        Y0[n-1,:] = f0y(n, b, yq)
        Y1[n-1,:] = f1y(n, b, yq)
        Y2[n-1,:] = f2y(n, b, yq)

    def int_mat(A, B, w):
        # A: (K,nq), B: (K,nq), w: (nq,)
        return (A * w) @ B.T

    # 1D integral matrices
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

    # ---- Patch quadrature for load vector (EXACT interval integration) ----
    xqp, wxp = gauss_interval(nq_patch_x, xL1, xL2)
    yqp, wyp = gauss_interval(nq_patch_y, yL1, yL2)

    X0p = np.zeros((M, nq_patch_x))
    for m in range(1, M+1):
        X0p[m-1,:] = f0x(m, a, xqp)

    Y0p = np.zeros((N, nq_patch_y))
    for n in range(1, N+1):
        Y0p[n-1,:] = f0y(n, b, yqp)

    Ix_patch = (X0p * wxp).sum(axis=1)  # ∫_{xL1}^{xL2} X_m dx
    Iy_patch = (Y0p * wyp).sum(axis=1)  # ∫_{yL1}^{yL2} Y_n dy

    # Quick sanity check: total load recovered by quadrature should be ~ P_total
    # (Only works exactly if w=1; here just check patch area)
    A_patch_num = wxp.sum() * wyp.sum()
    P_num = p0 * A_patch_num
    print(f"[Check] Patch area exact={A_patch:.6f} m^2, numeric={A_patch_num:.6f} m^2")
    print(f"[Check] Total load exact={P_total:.2f} N, numeric={P_num:.2f} N")

    # ---- Assemble linear system A W = B ----
    ndof = M*N
    A = np.zeros((ndof, ndof))
    B = np.zeros(ndof)

    def idx(m, n):
        return m*N + n  # m=0..M-1, n=0..N-1

    # Bending bilinear form:
    # A_ij = D ∬ [ φ_xx ψ_xx + φ_yy ψ_yy + ν(φ_xx ψ_yy + φ_yy ψ_xx) + 2(1-ν) φ_xy ψ_xy ]
    for m in range(M):
        for n in range(N):
            I = idx(m, n)
            B[I] = p0 * Ix_patch[m] * Iy_patch[n]

            for mp in range(M):
                for np_ in range(N):
                    J = idx(mp, np_)

                    term_xx_xx = Ix_22[m, mp] * Iy_00[n, np_]
                    term_yy_yy = Ix_00[m, mp] * Iy_22[n, np_]
                    term_xx_yy = Ix_20[m, mp] * Iy_02[n, np_]
                    term_yy_xx = Ix_02[m, mp] * Iy_20[n, np_]
                    term_xy_xy = Ix_11[m, mp] * Iy_11[n, np_]

                    A[I, J] = D * (
                        term_xx_xx
                        + term_yy_yy
                        + nu*(term_xx_yy + term_yy_xx)
                        + 2.0*(1.0-nu)*term_xy_xy
                    )

    Wvec = np.linalg.solve(A, B)
    W = Wvec.reshape((M, N))

    basis_info = {
        "bc_x1": bc_x1, "bc_x2": bc_x2,
        "f0x": f0x, "f1x": f1x, "f2x": f2x,
        "f0y": f0y, "f1y": f1y, "f2y": f2y,
    }
    return W, basis_info


# ============================================================
# 5) Postprocessing on a grid
# ============================================================
def eval_fields_on_grid(W, basis_info, x, y):
    M, N = W.shape
    f0x, f1x, f2x = basis_info["f0x"], basis_info["f1x"], basis_info["f2x"]
    f0y, f1y, f2y = basis_info["f0y"], basis_info["f1y"], basis_info["f2y"]

    nx_ = len(x)
    ny_ = len(y)

    X0g = np.zeros((M, nx_)); X1g = np.zeros((M, nx_)); X2g = np.zeros((M, nx_))
    for m in range(1, M+1):
        X0g[m-1,:] = f0x(m, a, x)
        X1g[m-1,:] = f1x(m, a, x)
        X2g[m-1,:] = f2x(m, a, x)

    Y0g = np.zeros((N, ny_)); Y1g = np.zeros((N, ny_)); Y2g = np.zeros((N, ny_))
    for n in range(1, N+1):
        Y0g[n-1,:] = f0y(n, b, y)
        Y1g[n-1,:] = f1y(n, b, y)
        Y2g[n-1,:] = f2y(n, b, y)

    # w, curvatures, twist
    w    = np.einsum("mx,ny,mn->yx", X0g, Y0g, W)
    w_xx = np.einsum("mx,ny,mn->yx", X2g, Y0g, W)
    w_yy = np.einsum("mx,ny,mn->yx", X0g, Y2g, W)
    w_xy = np.einsum("mx,ny,mn->yx", X1g, Y1g, W)

    # stresses at z=t/2
    sigma1 = -(E*z/(1.0-nu**2)) * (w_xx + nu*w_yy)
    sigma2 = -(E*z/(1.0-nu**2)) * (w_yy + nu*w_xx)
    tau12  = -(2.0*G*z) * w_xy
    svm    = np.sqrt(sigma1**2 - sigma1*sigma2 + sigma2**2 + 3.0*tau12**2)

    return {"w": w, "sigma1": sigma1, "sigma2": sigma2, "tau12": tau12, "svm": svm}


def bilinear_center(field, x, y):
    xc, yc = a/2.0, b/2.0
    ix = np.searchsorted(x, xc) - 1
    iy = np.searchsorted(y, yc) - 1
    ix = np.clip(ix, 0, len(x)-2)
    iy = np.clip(iy, 0, len(y)-2)
    x0, x1 = x[ix], x[ix+1]
    y0, y1 = y[iy], y[iy+1]
    fx = (xc-x0)/(x1-x0)
    fy = (yc-y0)/(y1-y0)
    f00 = field[iy, ix]
    f10 = field[iy, ix+1]
    f01 = field[iy+1, ix]
    f11 = field[iy+1, ix+1]
    return (1-fx)*(1-fy)*f00 + fx*(1-fy)*f10 + (1-fx)*fy*f01 + fx*fy*f11

def max_abs_with_location(A, x, y):
    idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    return float(np.abs(A[idx])), float(A[idx]), (float(x[idx[1]]), float(y[idx[0]]))

def max_with_location(A, x, y):
    idx = np.unravel_index(np.argmax(A), A.shape)
    return float(A[idx]), (float(x[idx[1]]), float(y[idx[0]]))

def print_results(tag, out):
    wc = bilinear_center(out["w"], xg, yg)

    s1max, s1val, s1loc = max_abs_with_location(out["sigma1"], xg, yg)
    s2max, s2val, s2loc = max_abs_with_location(out["sigma2"], xg, yg)
    t12max, t12val, t12loc = max_abs_with_location(out["tau12"], xg, yg)
    vmmax, vmloc = max_with_location(out["svm"], xg, yg)

    print(f"\n===== {tag} =====")
    print(f"Patch: {patch_y*1e3:.0f}mm x {patch_x*1e3:.0f}mm at center, total P = {P_total/1e3:.1f} kN")
    print(f"Uniform patch pressure p0 = {p0/1e6:.3f} MPa")
    print(f"w(center) = {wc*1e3:.3f} mm")

    print(f"max |sigma1| = {s1max/1e6:.3f} MPa (value={s1val/1e6:.3f} MPa) at (x1,x2)=({s1loc[0]:.4f},{s1loc[1]:.4f}) m")
    print(f"max |sigma2| = {s2max/1e6:.3f} MPa (value={s2val/1e6:.3f} MPa) at (x1,x2)=({s2loc[0]:.4f},{s2loc[1]:.4f}) m")
    print(f"max |tau12|  = {t12max/1e6:.3f} MPa (value={t12val/1e6:.3f} MPa) at (x1,x2)=({t12loc[0]:.4f},{t12loc[1]:.4f}) m")
    print(f"max von Mises = {vmmax/1e6:.3f} MPa at (x1,x2)=({vmloc[0]:.4f},{vmloc[1]:.4f}) m")


# ============================================================
# 6) Solve cases and plot ONLY required fields
# ============================================================
print("Solving A.5 with Rayleigh–Ritz (energy method) ...")
print(f"Using M=N={M}, nqx={nqx}, nqy={nqy}, plot grid={nx}x{ny}")

# Case 1: SSSS
W_ssss, info_ssss = ritz_solve_patch(M, N, bc_x1="SS", bc_x2="SS")
out_ssss = eval_fields_on_grid(W_ssss, info_ssss, xg, yg)

# Case 2: CCSS (more appropriate)
W_ccss, info_ccss = ritz_solve_patch(M, N, bc_x1="CC", bc_x2="SS")
out_ccss = eval_fields_on_grid(W_ccss, info_ccss, xg, yg)

print_results("Rayleigh–Ritz (SSSS)", out_ssss)
print_results("Rayleigh–Ritz (CCSS) [clamped x1, SS x2]", out_ccss)


# ONLY required plots
def plot_field(field, title, cbar_label, scale=1.0):
    plt.figure(figsize=(8, 5))
    cs = plt.contourf(Xg, Yg, field*scale, levels=40)
    plt.colorbar(cs, label=cbar_label)
    plt.xlabel(r"$x_1$ [m]")
    plt.ylabel(r"$x_2$ [m]")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Deflection (mm)
plot_field(out_ssss["w"], f"Deflection w – Rayleigh–Ritz (SSSS), M=N={M}", "w [mm]", scale=1e3)
plot_field(out_ccss["w"], f"Deflection w – Rayleigh–Ritz (CCSS), M=N={M}", "w [mm]", scale=1e3)

# von Mises (MPa)
plot_field(out_ssss["svm"], r"$\sigma_{vM}$ [MPa]", scale=1e-6)
plot_field(out_ccss["svm"], r"$\sigma_{vM}$ [MPa]", scale=1e-6)
