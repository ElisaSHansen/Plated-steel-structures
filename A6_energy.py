import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------- Gauss–Legendre quadrature grid ----------------
def gauss_grid(nq, a, b):
    xi, wi = np.polynomial.legendre.leggauss(nq)
    x = 0.5 * (xi + 1.0) * a
    wx = 0.5 * a * wi
    y = 0.5 * (xi + 1.0) * b
    wy = 0.5 * b * wi
    return x, wx, y, wy

# ---------------- Basis functions ----------------
def basis_funcs(case_name: str):
    c = case_name.upper()

    if c == "SSSS":
        # phi = sin(m*pi*x/a)*sin(n*pi*y/b)
        def phix(x, y, a, b, m, n):
            A = m * math.pi / a
            B = n * math.pi / b
            return A * math.cos(A * x) * math.sin(B * y)

        def phixx(x, y, a, b, m, n):
            A = m * math.pi / a
            B = n * math.pi / b
            return -(A * A) * math.sin(A * x) * math.sin(B * y)

        def phiyy(x, y, a, b, m, n):
            A = m * math.pi / a
            B = n * math.pi / b
            return -(B * B) * math.sin(A * x) * math.sin(B * y)

        def phixy(x, y, a, b, m, n):
            A = m * math.pi / a
            B = n * math.pi / b
            return A * B * math.cos(A * x) * math.cos(B * y)

        def phi_np(X, Y, a, b, m, n):
            return np.sin(m * math.pi * X / a) * np.sin(n * math.pi * Y / b)

        return phix, phixx, phiyy, phixy, phi_np

    if c == "CSCS":
        # clamped on x=0,a (short sides), simply supported on y=0,b (long sides)
        # phi = (1 - cos(2m*pi*x/a))*sin(n*pi*y/b)
        def phix(x, y, a, b, m, n):
            A = 2.0 * m * math.pi / a
            B = n * math.pi / b
            return A * math.sin(A * x) * math.sin(B * y)

        def phixx(x, y, a, b, m, n):
            A = 2.0 * m * math.pi / a
            B = n * math.pi / b
            return (A * A) * math.cos(A * x) * math.sin(B * y)

        def phiyy(x, y, a, b, m, n):
            A = 2.0 * m * math.pi / a
            B = n * math.pi / b
            return -(B * B) * (1.0 - math.cos(A * x)) * math.sin(B * y)

        def phixy(x, y, a, b, m, n):
            A = 2.0 * m * math.pi / a
            B = n * math.pi / b
            return A * math.sin(A * x) * B * math.cos(B * y)

        def phi_np(X, Y, a, b, m, n):
            return (1.0 - np.cos(2.0 * m * math.pi * X / a)) * np.sin(n * math.pi * Y / b)

        return phix, phixx, phiyy, phixy, phi_np

    raise ValueError("Ukjent case_name. Bruk 'SSSS' eller 'CSCS'.")

# ---------------- Build Ritz matrices: K a = Nx G a ----------------
def build_matrices(a, b, h, E, nu, case_name="SSSS", M=5, N=5, nq=30):
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    phix, phixx, phiyy, phixy, _phi_np = basis_funcs(case_name)

    modes = [(m, n) for m in range(1, M + 1) for n in range(1, N + 1)]
    ndof = len(modes)

    K = np.zeros((ndof, ndof), dtype=float)
    G = np.zeros((ndof, ndof), dtype=float)

    x, wx, y, wy = gauss_grid(nq, a, b)

    for i, (mi, ni) in enumerate(modes):
        for j, (mj, nj) in enumerate(modes):
            kij = 0.0
            gij = 0.0
            for ix, xv in enumerate(x):
                for iy, yv in enumerate(y):
                    wgt = wx[ix] * wy[iy]

                    xx_i = phixx(xv, yv, a, b, mi, ni)
                    yy_i = phiyy(xv, yv, a, b, mi, ni)
                    xy_i = phixy(xv, yv, a, b, mi, ni)

                    xx_j = phixx(xv, yv, a, b, mj, nj)
                    yy_j = phiyy(xv, yv, a, b, mj, nj)
                    xy_j = phixy(xv, yv, a, b, mj, nj)

                    # Full isotrop platebøyenergi (bilinearform)
                    kij += wgt * (
                        xx_i * xx_j
                        + yy_i * yy_j
                        + nu * (xx_i * yy_j + yy_i * xx_j)
                        + 2.0 * (1.0 - nu) * (xy_i * xy_j)
                    )

                    # Geometrisk stivhet for uniaxial Nx i x-retning: ∬ phi_xi * phi_xj dA
                    x_i = phix(xv, yv, a, b, mi, ni)
                    x_j = phix(xv, yv, a, b, mj, nj)
                    gij += wgt * (x_i * x_j)

            K[i, j] = D * kij
            G[i, j] = gij

    K = 0.5 * (K + K.T)
    G = 0.5 * (G + G.T)
    return K, G, modes

def solve_generalized_evp(K, G):
    A = np.linalg.solve(G, K)  # inv(G)K
    evals, evecs = np.linalg.eig(A)
    evals = np.real(evals)
    evecs = np.real(evecs)
    idx = np.argsort(evals)
    return evals[idx], evecs[:, idx]

def reconstruct_w(a, b, modes, coeffs, case_name="SSSS", nx=241, ny=241):
    *_, phi_np = basis_funcs(case_name)

    x = np.linspace(0.0, a, nx)
    y = np.linspace(0.0, b, ny)
    X, Y = np.meshgrid(x, y)
    W = np.zeros_like(X, dtype=float)

    for (m, n), c in zip(modes, coeffs):
        W += c * phi_np(X, Y, a, b, m, n)

    W /= np.max(np.abs(W))
    return X, Y, W

def plot_mode_shape_3d(X, Y, W, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, W, rstride=3, cstride=3, linewidth=0, antialiased=True)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("Normalized w")
    ax.set_title(title)
    plt.tight_layout()

def run_case(case_name, a, b, h, E, nu, M=6, N=6, nq=30, nx_plot=241, ny_plot=241):
    K, G, modes = build_matrices(a, b, h, E, nu, case_name=case_name, M=M, N=N, nq=nq)
    evals, evecs = solve_generalized_evp(K, G)

    pos_idx = np.where(evals > 0.0)[0]
    if len(pos_idx) == 0:
        raise RuntimeError("Ingen positive egenverdier. Øk nq og/eller M,N.")

    i_cr = pos_idx[0]
    Nx_cr = evals[i_cr]         # N/m
    sigma_cr = Nx_cr / h        # Pa

    a_cr = evecs[:, i_cr]
    kmax = int(np.argmax(np.abs(a_cr)))
    m_dom, n_dom = modes[kmax]

    print(f"--- {case_name} ---")
    print(f"Dominerende basis: m={m_dom}, n={n_dom}")
    print(f"Nx_cr = {Nx_cr/1e3:.3f} kN/m")
    print(f"sigma_cr = {sigma_cr/1e6:.3f} MPa")
    print()

    X, Y, W = reconstruct_w(a, b, modes, a_cr, case_name=case_name, nx=nx_plot, ny=ny_plot)
    plot_mode_shape_3d(X, Y, W, f"Kritisk buckling mode shape (3D) – {case_name}")

if __name__ == "__main__":
    # Dine verdier
    a = 2800e-3  # m
    b = 670e-3   # m
    h = 6.5e-3   # m
    E = 2.1e11   # Pa
    nu = 0.30

    # Ritz-innstillinger
    M = 6
    N = 6
    nq = 30

    run_case("SSSS", a, b, h, E, nu, M=M, N=N, nq=nq)
    run_case("CSCS", a, b, h, E, nu, M=M, N=N, nq=nq)

    plt.show()