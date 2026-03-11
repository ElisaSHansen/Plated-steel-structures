import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# B1 – Book-based method for yield pressure of longitudinal
# ------------------------------------------------------------
# Method follows the textbook pages you sent:
#
# 1) Orthotropic plate with longitudinals smeared out
# 2) Use textbook expression for D1:
#       D1 = D * [1 + b1 + 12(1-nu^2)*a1*c1^2/(1+a1)]
#    with D2 = D and D12 = D
#
# 3) Navier solution for orthotropic simply supported plate
#
# 4) Find M11,max numerically from:
#       M11 = D1 * (w_x1x1 + nu*w_x2x2)
#
# 5) Compute effective breadth be with Schade formula
#
# 6) Compute elastic section modulus We for one longitudinal
#    with attached plating
#
# 7) Convergence is checked on sigma_max/p
#    Then yield pressure is computed from:
#       p_y,max = sigma_y / (sigma_max / p)
# ============================================================


# -----------------------------
# USER SETTINGS
# -----------------------------
Mref = 59
Nref = 119
Mmax_sweep =59
Nmax_sweep = 119
TOL = 1e-3

nx, ny = 401, 201   # grid for max search


# -----------------------------
# 1) INPUT DATA
# -----------------------------
E = 210e9           # Pa
nu = 0.30           # -
sigma_y = 250e6     # Pa

# Plate panel
a = 2.8             # m  (between transverse beams)
b = 9.95            # m  (panel width)
t = 6.5e-3          # m  plate thickness

# Longitudinal flat bar
h_s = 160e-3        # m  height
t_s = 12e-3         # m  thickness
d1  = 670e-3        # m  spacing


# -----------------------------
# 2) GRID
# -----------------------------
x1 = np.linspace(0.0, a, nx)
x2 = np.linspace(0.0, b, ny)
X1, X2 = np.meshgrid(x1, x2, indexing="xy")


# -----------------------------
# 3) BOOK-BASED ORTHOTROPIC RIGIDITIES
# -----------------------------
def plate_rigidity(E, nu, t):
    return E * t**3 / (12.0 * (1.0 - nu**2))

def D1_book_flatbar(E, nu, t, d1, h_s, t_s):
    """
    Book-based D1 for a plate with longitudinals smeared out.

    Uses:
      D1 = D * [1 + b1 + 12(1-nu^2)*a1*c1^2/(1+a1)]

    where for one flat-bar stiffener per spacing d1:
      A1 = h_s * t_s
      I1 = t_s * h_s^3 / 12
      e1 = h_s / 2   (distance from plate surface on stiffener side to stiffener NA)
      a1 = A1 / (d1 * t)
      b1 = E*I1 / (D*d1)
      c1 = e1/t + 0.5
    """
    D = plate_rigidity(E, nu, t)

    A1 = h_s * t_s
    I1 = t_s * h_s**3 / 12.0
    e1 = h_s / 2.0

    a1 = A1 / (d1 * t)
    b1 = E * I1 / (D * d1)
    c1 = e1 / t + 0.5

    D1 = D * (1.0 + b1 + (12.0 * (1.0 - nu**2) * a1 * c1**2) / (1.0 + a1))
    return D1, D, a1, b1, c1

D1, D, a1, b1, c1 = D1_book_flatbar(E, nu, t, d1, h_s, t_s)
D2 = D
D12 = D


# -----------------------------
# 4) EFFECTIVE BREADTH (SCHADE)
# -----------------------------
def schade_effective_breadth(b_real, CL):
    """
    be/b = min[ 1.1 / (1 + 2(b/CL)^2), 1 ]
    """
    ratio = 1.1 / (1.0 + 2.0 * (b_real / CL)**2)
    ratio = min(ratio, 1.0)
    return ratio * b_real, ratio

CL = a
be, be_ratio = schade_effective_breadth(d1, CL)


# -----------------------------
# 5) NAVIER COEFFICIENTS
# -----------------------------
def Wmn_uniform(p, m, n, a, b, D1, D2, D12):
    """
    Book-style Navier denominator:
      Wmn = 16 p / [m n pi^6 * ( D1 (m/a)^4 + 2 D12 (mn/ab)^2 + D2 (n/b)^4 )]
    """
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0

    denom = (
        m * n * np.pi**6 *
        (
            D1 * (m / a)**4
            + 2.0 * D12 * (m * n / (a * b))**2
            + D2 * (n / b)**4
        )
    )
    return 16.0 * p / denom


# -----------------------------
# 6) DEFLECTION AND MOMENT FIELD
# -----------------------------
def compute_fields(p, Mmax, Nmax):
    """
    Computes:
      w
      w_x1x1
      w_x2x2
      M11 = D1 * (w_x1x1 + nu*w_x2x2)
    """
    w = np.zeros_like(X1)
    w_xx = np.zeros_like(X1)
    w_yy = np.zeros_like(X1)

    for m in range(1, Mmax + 1, 2):
        alpha = m * np.pi / a
        sin_m = np.sin(alpha * X1)

        for n in range(1, Nmax + 1, 2):
            beta = n * np.pi / b
            sin_n = np.sin(beta * X2)

            W = Wmn_uniform(p, m, n, a, b, D1, D2, D12)
            if W == 0.0:
                continue

            S = sin_m * sin_n
            w    += W * S
            w_xx += W * (-(alpha**2)) * S
            w_yy += W * (-(beta**2))  * S

    M11 = D1 * (w_xx + nu * w_yy)
    return w, M11


# -----------------------------
# 7) SECTION MODULUS (BOOK STYLE)
# -----------------------------
def section_modulus_longitudinal_with_plate(be, t, h_s, t_s):
    """
    Computes section modulus We for:
      - effective plate flange: width be, thickness t
      - one flat-bar stiffener: height h_s, thickness t_s

    Coordinate z=0 at plate mid-surface.
    Stiffener assumed on one side of plate.

    Returns:
      A, z_na, I, y_extreme, We
    """
    A_plate = be * t
    A_stiff = h_s * t_s
    A_total = A_plate + A_stiff

    z_plate = 0.0
    z_stiff = t / 2.0 + h_s / 2.0

    z_na = (A_plate * z_plate + A_stiff * z_stiff) / A_total

    I_plate_cent = be * t**3 / 12.0
    I_stiff_cent = t_s * h_s**3 / 12.0

    I_plate = I_plate_cent + A_plate * (z_plate - z_na)**2
    I_stiff = I_stiff_cent + A_stiff * (z_stiff - z_na)**2
    I_total = I_plate + I_stiff

    z_extreme = t / 2.0 + h_s
    y_extreme = z_extreme - z_na

    We = I_total / y_extreme
    return A_total, z_na, I_total, y_extreme, We

A_sec, z_na, I_sec, y_ext, We = section_modulus_longitudinal_with_plate(be, t, h_s, t_s)


# -----------------------------
# 8) SIGMA_MAX/P AND p_y FOR GIVEN (M,N)
# -----------------------------
def response_for_MN(Mmax, Nmax):
    """
    For p = 1 Pa:
      1) compute M11
      2) find max absolute M11
      3) sigma_per_Pa = M11_max * d1 / We
      4) p_y = sigma_y / sigma_per_Pa
    """
    p_test = 1.0
    w, M11 = compute_fields(p_test, Mmax, Nmax)

    idx = np.unravel_index(np.argmax(np.abs(M11)), M11.shape)
    M11_max = float(np.abs(M11[idx]))
    x_at = float(x1[idx[1]])
    y_at = float(x2[idx[0]])

    sigma_per_Pa = M11_max * d1 / We
    p_y = sigma_y / sigma_per_Pa

    return sigma_per_Pa, p_y, M11_max, x_at, y_at


# -----------------------------
# 9) CONVERGENCE
# -----------------------------
def ensure_odd(k):
    return k if (k % 2 == 1) else (k - 1)

def first_converged(trunc_list, values, ref_value, tol):
    rel_err = np.abs(values - ref_value) / (np.abs(ref_value) + 1e-30)
    for i, e in enumerate(rel_err):
        if e <= tol:
            return trunc_list[i], rel_err
    return trunc_list[-1], rel_err


# -----------------------------
# 10) RUN
# -----------------------------
if __name__ == "__main__":
    Mref = ensure_odd(Mref)
    Nref = ensure_odd(Nref)
    Mmax_sweep = ensure_odd(Mmax_sweep)
    Nmax_sweep = ensure_odd(Nmax_sweep)

    # Reference based on sigma_max/p
    sigma_ref, py_ref, _, x_ref, y_ref = response_for_MN(Mref, Nref)

    # Sweep M with N fixed, checking convergence of sigma_max/p
    M_list = np.arange(1, min(Mmax_sweep, Mref) + 1, 2)
    sigma_M = np.array([response_for_MN(M, Nref)[0] for M in M_list])
    Mconv, relM = first_converged(M_list, sigma_M, sigma_ref, TOL)

    # Sweep N with M fixed, checking convergence of sigma_max/p
    N_list = np.arange(1, min(Nmax_sweep, Nref) + 1, 2)
    sigma_N = np.array([response_for_MN(Mref, N)[0] for N in N_list])
    Nconv, relN = first_converged(N_list, sigma_N, sigma_ref, TOL)

    # Final values using converged M and N
    sigma_per_Pa, py_final, M11max_1Pa, x_at, y_at = response_for_MN(Mconv, Nconv)

    # Deflection and moment field at py_final
    w_final, M11_final = compute_fields(py_final, Mconv, Nconv)
    w_max = float(np.max(np.abs(w_final)))

    ix_c = int(np.argmin(np.abs(x1 - a/2)))
    iy_c = int(np.argmin(np.abs(x2 - b/2)))
    w_center = float(w_final[iy_c, ix_c])

    # Stress field at py_final
    sigma_field = np.abs(M11_final) * d1 / We / 1e6   # MPa
    sigma_max_final = float(np.max(sigma_field))

    # -------------------------
    # OUTPUT
    # -------------------------
    print("\n" + "=" * 78)
    print("B1 – Yield pressure using textbook method")
    print("=" * 78)

    print("\nGeometry")
    print(f"  Panel: a = {a:.3f} m, b = {b:.3f} m")
    print(f"  Plate thickness t = {t*1e3:.2f} mm")
    print(f"  Longitudinal flat bar = {h_s*1e3:.0f} x {t_s*1e3:.0f} mm")
    print(f"  Stiffener spacing d1 = {d1*1e3:.0f} mm")

    print("\nMaterial")
    print(f"  E = {E/1e9:.1f} GPa")
    print(f"  nu = {nu:.2f}")
    print(f"  sigma_y = {sigma_y/1e6:.1f} MPa")

    print("\nOrthotropic rigidities (book-based)")
    print(f"  D  = {D:.6e} N·m")
    print(f"  a1 = {a1:.6f}")
    print(f"  b1 = {b1:.6f}")
    print(f"  c1 = {c1:.6f}")
    print(f"  D1 = {D1:.6e} N·m")
    print(f"  D2 = {D2:.6e} N·m")
    print(f"  D12 = {D12:.6e} N·m")
    print(f"  D1/D2 = {D1/D2:.3f}")

    print("\nEffective breadth (Schade)")
    print(f"  CL = {CL:.3f} m")
    print(f"  be/b = {be_ratio:.6f}")
    print(f"  be = {be:.3f} m ({be*1e3:.1f} mm)")

    print("\nSection properties (longitudinal + attached plate)")
    print(f"  Area A = {A_sec*1e6:.3f} mm^2")
    print(f"  Neutral axis from plate mid-surface z_na = {z_na*1e3:.3f} mm")
    print(f"  Moment of inertia I = {I_sec*1e12:.6e} mm^4")
    print(f"  Extreme fiber distance y = {y_ext*1e3:.3f} mm")
    print(f"  Elastic section modulus We = {We*1e9:.6e} mm^3")

    print("\nConvergence on sigma_max / p")
    print(f"  Reference: (Mref,Nref)=({Mref},{Nref})")
    print(f"  Tolerance = {TOL:g}")
    print(f"  sigma_ref/p = {sigma_ref:.6e} Pa/Pa")
    print(f"  Mconv = {Mconv}")
    print(f"  Nconv = {Nconv}")
    print(f"  sigma_final/p = {sigma_per_Pa:.6e} Pa/Pa")
    print(f"  relative error = {abs(sigma_per_Pa - sigma_ref)/abs(sigma_ref):.3e}")

    print("\nMaximum longitudinal bending moment")
    print(f"  M11,max / p = {M11max_1Pa:.6e} N")
    print(f"  occurs at x1 = {x_at:.4f} m = {x_at/a:.3f} a")
    print(f"  occurs at x2 = {y_at:.4f} m = {y_at/b:.3f} b")

    print("\nYield pressure")
    print(f"  sigma_max / p = {sigma_per_Pa:.6e} Pa/Pa")
    print(f"  py_ref   = {py_ref:.6e} Pa")
    print(f"  py_final = {py_final:.6e} Pa")
    print(f"           = {py_final/1e3:.6f} kN/m^2")
    print(f"           = {py_final/1e6:.6f} MPa")

    print("\nDeflection at py,max")
    print(f"  w_max    = {w_max*1e3:.6f} mm")
    print(f"  w_center = {w_center*1e3:.6f} mm")

    print("\nStress at py,max")
    print(f"  sigma_max = {sigma_max_final:.6f} MPa")

    print("=" * 78)

    # -------------------------
    # PLOTS
    # -------------------------
    plt.figure()
    plt.plot(M_list, sigma_M / 1e6, marker="o")
    plt.axhline(sigma_ref / 1e6, linestyle="--", label="Reference")
    plt.xlabel("Mmax (odd), with N = Nref")
    plt.ylabel(r"$\sigma_{\max}/p$ [MPa/Pa]")
    plt.title(f"Convergence of sigma_max/p in M (Nref = {Nref})")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(N_list, sigma_N / 1e6, marker="o")
    plt.axhline(sigma_ref / 1e6, linestyle="--", label="Reference")
    plt.xlabel("Nmax (odd), with M = Mref")
    plt.ylabel(r"$\sigma_{\max}/p$ [MPa/Pa]")
    plt.title(f"Convergence of sigma_max/p in N (Mref = {Mref})")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.contourf(X1, X2, M11_final / 1e3, levels=40)
    plt.colorbar(label="M11 [kN·m/m]")
    plt.xlabel("x1 [m]")
    plt.ylabel("x2 [m]")
    plt.title("Longitudinal bending moment M11 at p_y,max")

    plt.figure()
    plt.contourf(X1, X2, w_final * 1e3, levels=40)
    plt.colorbar(label="Deflection [mm]")
    plt.xlabel("x1 [m]")
    plt.ylabel("x2 [m]")
    plt.title("Deflection at p_y,max")

    plt.show()

    # ============================================================
    # ADVANCED PLOT 4: 3D surface plot of M11
    # ============================================================
    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X1, X2, M11_final / 1e3,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_xlabel(r"$x_1$ [m]")
    ax.set_ylabel(r"$x_2$ [m]")
    ax.set_zlabel(r"$M_{11}$ [kN·m/m]")
    ax.set_title(r"3D surface of $M_{11}$ at $p_{y,\max}$")

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # ADVANCED PLOT 6: Stress field
    # ============================================================
    plt.figure(figsize=(8, 4.8))
    cf = plt.contourf(X1, X2, sigma_field, levels=40, cmap="inferno")
    plt.colorbar(cf, label=r"$|\sigma|$ [MPa]")
    plt.plot(x_at, y_at, "co", markeredgecolor="k", label="Maximum stress")
    plt.xlabel(r"$x_1$ [m]")
    plt.ylabel(r"$x_2$ [m]")
    plt.title(r"Stress field at $p_{y,\max}$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ============================================================
    # DIAGNOSTIC: Check location of M11 maximum along x1 = a/2
    # ============================================================
    ix_mid = np.argmin(np.abs(x1 - a / 2))

    M_line = M11_final[:, ix_mid]

    idx_line = np.argmax(np.abs(M_line))
    x2_line_max = x2[idx_line]
    M_line_max = M_line[idx_line]

    print("\n--- Diagnostic: M11 along x1 = a/2 ---")
    print(f"Maximum along line occurs at x2 = {x2_line_max:.4f} m = {x2_line_max/b:.4f} b")
    print(f"M11_max_line = {M_line_max/1e3:.3f} kN·m/m")

    iy_center = np.argmin(np.abs(x2 - b / 2))
    M_center = M_line[iy_center]

    iy_edge = np.argmin(np.abs(x2 - 0.02 * b))
    M_edge = M_line[iy_edge]

    print(f"M11 at center (x2 = 0.5b) = {M_center/1e3:.3f} kN·m/m")
    print(f"M11 at x2 = 0.02b        = {M_edge/1e3:.3f} kN·m/m")

    plt.figure(figsize=(7, 4))
    plt.plot(x2 / b, M_line / 1e3, label=r"$M_{11}(x_1=a/2,x_2)$")
    plt.axvline(x2_line_max / b, color="r", linestyle="--", label="Maximum")
    plt.axvline(0.5, color="k", linestyle=":", label="Center (0.5b)")
    plt.xlabel(r"$x_2/b$")
    plt.ylabel(r"$M_{11}$ [kN·m/m]")
    plt.title(r"Distribution of $M_{11}$ along $x_1 = a/2$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# PLOT: |M11|/p through maximum point (book-style)
# ============================================================

M11_over_p = np.abs(M11_final / py_final)   # [m^2]

# Finn indeks til maksimumspunktet
ix_max = np.argmin(np.abs(x1 - x_at))
iy_max = np.argmin(np.abs(x2 - y_at))

# Kurve 1: |M11|(x1_max, x2/b)
curve_x2 = M11_over_p[:, ix_max]
x2_norm = x2 / b

# Kurve 2: |M11|(x1/a, x2_max)
curve_x1 = M11_over_p[iy_max, :]
x1_norm = x1 / a

plt.figure(figsize=(7.2, 4.8))

plt.plot(x2_norm, curve_x2,
         color="black",
         linewidth=1.8,
         label=rf"$M_{{11}}({x_at/a:.2f},\,x_2/b)$")

plt.plot(x1_norm, curve_x1,
         color="black",
         linewidth=0.9,
         label=rf"$M_{{11}}(x_1/a,\,{y_at/b:.2f})$")

plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.05 * max(np.max(curve_x1), np.max(curve_x2)))

plt.xlabel(r"$x_1/a,\;x_2/b$")
plt.ylabel(r"$M_{11}/p\;[m^2]$")

plt.legend(frameon=False, loc="lower center")
plt.tight_layout()
plt.show()