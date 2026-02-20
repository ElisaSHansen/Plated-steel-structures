import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Project A.4 – Plate bending under uniform pressure p0
# Methods:
#   1) Navier (SSSS) series (reference for simply supported)
#   2) Rayleigh–Ritz (energy method) with:
#        a) SSSS (should match Navier if basis is sine-sine)
#        b) CCSS (more "appropriate" BC: clamped on x1=0,a; SS on x2=0,b)
#
# Outputs (as asked):
#   - deflection in center w(a/2,b/2)
#   - maximum stresses (direct sigma1/sigma2, shear tau12, von Mises)
#   - ONLY plots of deflections and von Mises (no extra plots)
# ============================================================

# -----------------------------
# 1) Inputs
# -----------------------------
p0 = 10e3          # N/m^2
a  = 2.8           # m  (x1-direction)
b  = 0.67          # m  (x2-direction)
t  = 0.0065        # m
E  = 210e9         # Pa
nu = 0.30          # -
G  = E/(2*(1+nu))  # Pa
D  = E * t**3 / (12.0*(1.0-nu**2))

# stresses at top surface
z = t/2

# Truncation (choose reasonable; you can increase if needed)
M = 20
N = 20

# Plot grid
nx = 201
ny = 201
xg = np.linspace(0.0, a, nx)
yg = np.linspace(0.0, b, ny)


# ============================================================
# 2) Basis functions for Rayleigh–Ritz
# ============================================================
def basis_SS(k, L, x):
    """Simply supported shape: sin(k*pi*x/L). k=1.."""
    return np.sin(k*np.pi*x/L)

def basis_SS_d1(k, L, x):
    return (k*np.pi/L) * np.cos(k*np.pi*x/L)

def basis_SS_d2(k, L, x):
    return - (k*np.pi/L)**2 * np.sin(k*np.pi*x/L)

def basis_CC(k, L, x):
    """
    Clamped-clamped (geometric BC): w=0 and dw/dx=0 at x=0,L
    A simple robust choice:
        X_k(x) = sin(pi x/L) * sin(k pi x/L)
    k=1.. gives a set of admissible functions.
    """
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

    # derivative of: (π/L)c1*sk + (kπ/L)s1*ck
    term1 = (np.pi/L) * (-(np.pi/L)*s1*sk + c1*(k*np.pi/L)*ck)
    term2 = (k*np.pi/L) * ((np.pi/L)*c1*ck + s1*(-k*np.pi/L)*sk)
    return term1 + term2


def gauss_legendre(n, L):
    """Nodes/weights on [0,L]"""
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5*(xi+1.0)*L
    w = 0.5*L*wi
    return x, w


# ============================================================
# 3) Rayleigh–Ritz solver (energy method)
# ============================================================
def ritz_solve(M, N, bc_x1="SS", bc_x2="SS", nqx=3*M, nqy=3*N):
    """
    Solve for Wmn in w(x1,x2)=sum_m sum_n Wmn X_m1(x1) X_n2(x2)
    using total potential energy (bending energy + load potential).

    Returns W (MxN) and a dict with basis evaluators for postprocessing.
    """

    # choose basis sets for x1 and x2
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

    # quadrature points
    xq, wx = gauss_legendre(nqx, a)
    yq, wy = gauss_legendre(nqy, b)

    # precompute basis values at quadrature points
    X0 = np.zeros((M, nqx))
    X1 = np.zeros((M, nqx))
    X2 = np.zeros((M, nqx))
    for m in range(1, M+1):
        X0[m-1,:] = f0x(m, a, xq)
        X1[m-1,:] = f1x(m, a, xq)
        X2[m-1,:] = f2x(m, a, xq)

    Y0 = np.zeros((N, nqy))
    Y1 = np.zeros((N, nqy))
    Y2 = np.zeros((N, nqy))
    for n in range(1, N+1):
        Y0[n-1,:] = f0y(n, b, yq)
        Y1[n-1,:] = f1y(n, b, yq)
        Y2[n-1,:] = f2y(n, b, yq)

    # build 1D integral matrices (weighted inner products)
    # Ix_00[m,mp] = ∫ X0_m X0_mp dx etc.
    def int_mat(A, B, w):
        # A: (K,nq), B: (K,nq), w: (nq,)
        return (A * w) @ B.T

    Ix_00 = int_mat(X0, X0, wx)
    Ix_11 = int_mat(X1, X1, wx)
    Ix_20 = int_mat(X2, X0, wx)
    Ix_02 = Ix_20.T
    Ix_22 = int_mat(X2, X2, wx)

    Iy_00 = int_mat(Y0, Y0, wy)
    Iy_11 = int_mat(Y1, Y1, wy)
    Iy_20 = int_mat(Y2, Y0, wy)  # note: using Y2 as "second deriv"
    Iy_02 = Iy_20.T
    Iy_22 = int_mat(Y2, Y2, wy)

    # Load vector (uniform p): B(m,n) = ∬ p X0_m Y0_n = p (∫X0_m dx)(∫Y0_n dy)
    Ix_0 = (X0 * wx).sum(axis=1)  # (M,)
    Iy_0 = (Y0 * wy).sum(axis=1)  # (N,)

    # Assemble full A and B
    ndof = M*N
    A = np.zeros((ndof, ndof))
    Bvec = np.zeros(ndof)

    def idx(m, n):
        # m=0..M-1, n=0..N-1
        return m*N + n

    # Bilinear form for bending energy:
    # U = 0.5 D ∬ [ w_xx^2 + w_yy^2 + 2ν w_xx w_yy + 2(1-ν) w_xy^2 ] dA
    # => A_ij = D ∬ [ φ_xx ψ_xx + φ_yy ψ_yy + ν(φ_xx ψ_yy + φ_yy ψ_xx) + 2(1-ν) φ_xy ψ_xy ]
    for m in range(M):
        for n in range(N):
            I = idx(m, n)

            # load
            Bvec[I] = p0 * Ix_0[m] * Iy_0[n]

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

    # Solve A W = B
    Wvec = np.linalg.solve(A, Bvec)
    W = Wvec.reshape((M, N))

    basis_info = {
        "bc_x1": bc_x1, "bc_x2": bc_x2,
        "f0x": f0x, "f1x": f1x, "f2x": f2x,
        "f0y": f0y, "f1y": f1y, "f2y": f2y,
    }
    return W, basis_info


# ============================================================
# 4) Postprocessing: w-field and stresses from w-derivatives
# ============================================================
def eval_fields_on_grid(W, basis_info, x, y):
    """
    Evaluate w, w_xx, w_yy, w_xy on grid y,x then stresses and von Mises.
    Returns dict with arrays (ny,nx).
    """
    M, N = W.shape
    f0x, f1x, f2x = basis_info["f0x"], basis_info["f1x"], basis_info["f2x"]
    f0y, f1y, f2y = basis_info["f0y"], basis_info["f1y"], basis_info["f2y"]

    nx_ = len(x)
    ny_ = len(y)

    X0g = np.zeros((M, nx_))
    X1g = np.zeros((M, nx_))
    X2g = np.zeros((M, nx_))
    for m in range(1, M+1):
        X0g[m-1,:] = f0x(m, a, x)
        X1g[m-1,:] = f1x(m, a, x)
        X2g[m-1,:] = f2x(m, a, x)

    Y0g = np.zeros((N, ny_))
    Y1g = np.zeros((N, ny_))
    Y2g = np.zeros((N, ny_))
    for n in range(1, N+1):
        Y0g[n-1,:] = f0y(n, b, y)
        Y1g[n-1,:] = f1y(n, b, y)
        Y2g[n-1,:] = f2y(n, b, y)

    # w(y,x) = Σ_mn Wmn X0_m(x) Y0_n(y)
    w    = np.einsum("mx,ny,mn->yx", X0g, Y0g, W)
    w_xx = np.einsum("mx,ny,mn->yx", X2g, Y0g, W)
    w_yy = np.einsum("mx,ny,mn->yx", X0g, Y2g, W)
    w_xy = np.einsum("mx,ny,mn->yx", X1g, Y1g, W)

    # stresses (top surface)
    sigma1 = -(E*z/(1.0-nu**2)) * (w_xx + nu*w_yy)
    sigma2 = -(E*z/(1.0-nu**2)) * (w_yy + nu*w_xx)
    tau12  = -(2.0*G*z) * w_xy
    svm    = np.sqrt(sigma1**2 - sigma1*sigma2 + sigma2**2 + 3.0*tau12**2)

    return {
        "w": w, "w_xx": w_xx, "w_yy": w_yy, "w_xy": w_xy,
        "sigma1": sigma1, "sigma2": sigma2, "tau12": tau12, "svm": svm
    }

def center_value(field, x, y):
    """Bilinear interp at (a/2,b/2) from a y,x array."""
    xc = a/2
    yc = b/2
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


# ============================================================
# 5) Navier (SSSS) closed-form coefficients for uniform load
# ============================================================
def navier_fields(M, N, x, y):
    """
    Navier series for SSSS under uniform pressure p0.
    Uses odd-odd terms only.
    """
    nx_ = len(x)
    ny_ = len(y)
    w_xx = np.zeros((ny_, nx_))
    w_yy = np.zeros((ny_, nx_))
    w_xy = np.zeros((ny_, nx_))
    w    = np.zeros((ny_, nx_))

    sin_mx = {m: np.sin(m*np.pi*x/a) for m in range(1, M+1)}
    cos_mx = {m: np.cos(m*np.pi*x/a) for m in range(1, M+1)}
    sin_ny = {n: np.sin(n*np.pi*y/b) for n in range(1, N+1)}
    cos_ny = {n: np.cos(n*np.pi*y/b) for n in range(1, N+1)}

    for m in range(1, M+1):
        for n in range(1, N+1):
            if (m % 2 == 0) or (n % 2 == 0):
                continue
            Pmn = 16.0*p0/(np.pi**2*m*n)
            Wmn = Pmn / (D*np.pi**4 * ((m/a)**2 + (n/b)**2)**2)

            S = np.outer(sin_ny[n], sin_mx[m])
            C = np.outer(cos_ny[n], cos_mx[m])

            w    += Wmn * S
            w_xx += Wmn * (-(m*np.pi/a)**2) * S
            w_yy += Wmn * (-(n*np.pi/b)**2) * S
            w_xy += Wmn * ((m*np.pi/a)*(n*np.pi/b)) * C

    sigma1 = -(E*z/(1.0-nu**2)) * (w_xx + nu*w_yy)
    sigma2 = -(E*z/(1.0-nu**2)) * (w_yy + nu*w_xx)
    tau12  = -(2.0*G*z) * w_xy
    svm    = np.sqrt(sigma1**2 - sigma1*sigma2 + sigma2**2 + 3.0*tau12**2)

    return {"w": w, "sigma1": sigma1, "sigma2": sigma2, "tau12": tau12, "svm": svm}


# ============================================================
# 6) Run: Navier + Ritz(SSSS) + Ritz(CCSS)
# ============================================================
print("Solving...")

# Navier (SSSS)
nav = navier_fields(M, N, xg, yg)

# Ritz SSSS (energy) – same BC as Navier (should be close)
W_ssss, info_ssss = ritz_solve(M, N, bc_x1="SS", bc_x2="SS")
ritz_ssss = eval_fields_on_grid(W_ssss, info_ssss, xg, yg)

# Ritz CCSS (more appropriate BC): clamped on x1=0,a and SS on x2=0,b
W_ccss, info_ccss = ritz_solve(M, N, bc_x1="CC", bc_x2="SS")
ritz_ccss = eval_fields_on_grid(W_ccss, info_ccss, xg, yg)

# ============================================================
# 7) Print results (center deflection + max stresses)
# ============================================================
def print_results(tag, out):
    wc = center_value(out["w"], xg, yg)

    s1max, s1val, s1loc = max_abs_with_location(out["sigma1"], xg, yg)
    s2max, s2val, s2loc = max_abs_with_location(out["sigma2"], xg, yg)
    t12max, t12val, t12loc = max_abs_with_location(out["tau12"], xg, yg)
    vmmax, vmloc = max_with_location(out["svm"], xg, yg)

    print(f"\n===== {tag} =====")
    print(f"w(center) = w(a/2,b/2) = {wc*1e3:.4f} mm")

    print(f"max |sigma1| = {s1max/1e6:.3f} MPa (value={s1val/1e6:.3f} MPa) at (x1,x2)=({s1loc[0]:.4f},{s1loc[1]:.4f}) m")
    print(f"max |sigma2| = {s2max/1e6:.3f} MPa (value={s2val/1e6:.3f} MPa) at (x1,x2)=({s2loc[0]:.4f},{s2loc[1]:.4f}) m")
    print(f"max |tau12|  = {t12max/1e6:.3f} MPa (value={t12val/1e6:.3f} MPa) at (x1,x2)=({t12loc[0]:.4f},{t12loc[1]:.4f}) m")
    print(f"max von Mises = {vmmax/1e6:.3f} MPa at (x1,x2)=({vmloc[0]:.4f},{vmloc[1]:.4f}) m")

print_results("Navier (SSSS)", nav)
print_results("Rayleigh–Ritz (SSSS)", ritz_ssss)
print_results("Rayleigh–Ritz (CCSS) [clamped x1, SS x2]", ritz_ccss)


# ============================================================
# 8) ONLY required plots: deflections + von Mises
#    (Navier SSSS and Ritz CCSS shown)
# ============================================================
X, Y = np.meshgrid(xg, yg)

# Deflection plots
plt.figure(figsize=(8, 5))
cs = plt.contourf(X, Y, nav["w"]*1e3, levels=40)
plt.colorbar(cs, label="w [mm]")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.title(f"Deflection w – Navier (SSSS), M=N={M}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
cs = plt.contourf(X, Y, ritz_ccss["w"]*1e3, levels=40)
plt.colorbar(cs, label="w [mm]")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.title(f"Deflection w – Rayleigh–Ritz (CCSS), M=N={M}")
plt.tight_layout()
plt.show()

# von Mises plots
plt.figure(figsize=(8, 5))
cs = plt.contourf(X, Y, nav["svm"]/1e6, levels=40)
plt.colorbar(cs, label=r"$\sigma_{vM}$ [MPa]")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.title(f"von Mises – Navier (SSSS), M=N={M}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
cs = plt.contourf(X, Y, ritz_ccss["svm"]/1e6, levels=40)
plt.colorbar(cs, label=r"$\sigma_{vM}$ [MPa]")
plt.xlabel(r"$x_1$ [m]")
plt.ylabel(r"$x_2$ [m]")
plt.title(f"von Mises – Rayleigh–Ritz (CCSS), M=N={M}")
plt.tight_layout()
plt.show()

