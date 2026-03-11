import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Project A – Part B1
# Ortotrop Navier + konvergens på p_yield (med separate M og N)
# ------------------------------------------------------------
# Viktig endring ift. "min" gamle kode:
#   D1 regnes nå slik BOKA gjør (smear-out / principal rigidities):
#
#     D  = E*t^3 / (12(1-nu^2))
#     a1 = A1/(d1*t)
#     b1 = E*I1/(D*d1)
#     c1 = e1/t + 0.5
#     D1 = D * [ 1 + b1 + 12(1-nu^2)*a1*c1^2/(1+a1) ]
#
# For longitudinals-only settes (som i boka-eksempel):
#     D2  = D
#     D12 = D   (torsjons-/koblingsstivhet tas som plate-dominert)
#
# p_yield finnes ved:
#   - Løs M11-felt for p=1 Pa
#   - Finn |M11|max
#   - M_strip = |M11|max * be
#   - sigma = M_strip * y / I
#   - p_yield = sigma_y / (sigma per 1 Pa)
#
# Konvergens på p_yield:
#   - Sweep M (N=Nref) -> finn Mconv
#   - Sweep N (M=Mref) -> finn Nconv
# Plottene viser p_yield på y-aksen.
# ============================================================

# -----------------------------
# 0) Brukerinnstillinger
# -----------------------------
Mref = 79            # referanse M (odd)
Nref = 79            # referanse N (odd)
TOL  = 1e-3          # relativ toleranse på p_yield

Mmax_sweep = 79      # sweep opp til (odd)
Nmax_sweep = 79

# Grid for å finne maks |M11| (øker du disse -> mer nøyaktig, men tregere)
nx, ny = 201, 101

# -----------------------------
# 1) Inndata (Project A)
# -----------------------------
E = 210e9            # Pa
nu = 0.30            # -
sigma_y = 250e6      # Pa

# Bay mellom transverse beams
a = 2.8              # m
b = 9.95             # m (panelbredde)

# Plate
t = 6.5e-3           # m

# Longitudinals: flat bar 160x12, spacing 670 mm
h_s = 160e-3         # m (stiverhøyde)
t_s = 12e-3          # m (stivertykkelse)
d1  = 670e-3         # m (avstand mellom longitudinals)

# -----------------------------
# 2) Grid
# -----------------------------
x1 = np.linspace(0, a, nx)
x2 = np.linspace(0, b, ny)
X1, X2 = np.meshgrid(x1, x2, indexing="xy")

# -----------------------------
# 3) Effective breadth (Schade)
# -----------------------------
def schade_effective_breadth(b_real, CL):
    """
    Schade (1951):
      be/b = min( 1.1 / (1 + 2*(b/CL)^2 ), 1 )
    """
    ratio = 1.1 / (1.0 + 2.0 * (b_real / CL)**2)
    ratio = min(ratio, 1.0)
    return ratio * b_real, ratio

CL = a  # simply supported span -> zero moment at supports -> CL ≈ a
be, be_ratio = schade_effective_breadth(d1, CL)

# -----------------------------
# 4) Ortotrope rigiditeter (BOKA: longitudinals-only smear-out)
# -----------------------------
def orthotropic_rigidities_book_longitudinals(E, nu, t, d1, h_s, t_s):
    """
    Boka-formulering (principal rigidities) for panel med kun longitudinals.
    Antakelser:
      - Longitudinals i x1-retning
      - Ingen transverse stiffeners i denne delen -> a2=b2=c2=0
      - D2 = D, D12 = D (plate-dominert) slik i bokeksempelet
    """
    D = E * t**3 / (12.0 * (1.0 - nu**2))

    # Stiffener properties (flat bar):
    A1 = t_s * h_s
    I1 = (t_s * h_s**3) / 12.0

    # e1: distance from plate SURFACE on stiffener side to stiffener NA
    # For flat bar standing on plate: NA is at h_s/2 from plate surface.
    e1 = 0.5 * h_s

    a1 = A1 / (d1 * t)
    b1 = (E * I1) / (D * d1)
    c1 = e1 / t + 0.5

    D1 = D * (1.0 + b1 + (12.0 * (1.0 - nu**2) * a1 * c1**2) / (1.0 + a1))
    D2 = D
    D12 = D

    return D1, D2, D12, D, a1, b1, c1

D1, D2, D12, D_plate, a1, b1, c1 = orthotropic_rigidities_book_longitudinals(E, nu, t, d1, h_s, t_s)

# -----------------------------
# 5) Navier-koeffisienter (uniform last)
# -----------------------------
def Pmn_uniform(p, m, n):
    # For simply supported Navier series:
    # Pmn = 16 p / (m n pi^2) for odd m,n
    if (m % 2 == 1) and (n % 2 == 1):
        return 16.0 * p / (m * n * np.pi**2)
    return 0.0

def Wmn_uniform(p, m, n):
    P = Pmn_uniform(p, m, n)
    if P == 0.0:
        return 0.0
    alpha = m * np.pi / a
    beta  = n * np.pi / b
    denom = D1 * alpha**4 + 2.0 * D12 * alpha**2 * beta**2 + D2 * beta**4
    return P / denom

# -----------------------------
# 6) Momentfelt: M11 = -D1 (w_xx + nu w_yy)
# -----------------------------
def compute_M11_field(p, Mmax, Nmax):
    w_xx = np.zeros_like(X1)
    w_yy = np.zeros_like(X1)

    for m in range(1, Mmax + 1, 2):
        alpha = m * np.pi / a
        sin_m = np.sin(alpha * X1)
        for n in range(1, Nmax + 1, 2):
            beta = n * np.pi / b
            W = Wmn_uniform(p, m, n)
            if W == 0.0:
                continue
            sin_n = np.sin(beta * X2)
            S = sin_m * sin_n
            w_xx += W * (-(alpha**2)) * S
            w_yy += W * (-(beta**2))  * S

    M11 = -D1 * (w_xx + nu * w_yy)
    return M11

# -----------------------------
# 7) Tverrsnitt for spenningskobling: plate strip (be) + stiffener
# -----------------------------
def composite_strip_I_and_yextreme(d_eff):
    """
    Composite strip of width d_eff:
      - Plate strip width d_eff, thickness t, centered at z=0 (plate mid-surface)
      - Stiffener on one side (doesn't matter for |sigma|): height h_s, thickness t_s

    Return:
      I_total about composite NA, and y_extreme = max distance to stiffener extreme fiber.
    """
    # Plate strip
    A_p = d_eff * t
    z_p = 0.0
    I_p_cent = d_eff * t**3 / 12.0

    # Stiffener centroid above mid-surface (magnitude only)
    A_s = t_s * h_s
    z_s = (t / 2.0) + (h_s / 2.0)
    I_s_cent = t_s * h_s**3 / 12.0

    # Neutral axis
    z_na = (A_p * z_p + A_s * z_s) / (A_p + A_s)

    # Parallel axis
    I_p = I_p_cent + A_p * (z_p - z_na)**2
    I_s = I_s_cent + A_s * (z_s - z_na)**2
    I_tot = I_p + I_s

    # Extreme stiffener fiber on that side:
    z_tip = (t / 2.0) + h_s
    y_tip = z_tip - z_na

    return I_tot, abs(y_tip)

I_tot, y_ext = composite_strip_I_and_yextreme(be)

# -----------------------------
# 8) p_yield(M,N) via linear scaling from p=1 Pa
# -----------------------------
def p_yield_for_MN(Mmax, Nmax, d_eff=be):
    p_test = 1.0
    M11 = compute_M11_field(p_test, Mmax, Nmax)

    idx = np.unravel_index(np.argmax(np.abs(M11)), M11.shape)
    M11_max = float(np.abs(M11[idx]))
    x_at = float(x1[idx[1]])
    y_at = float(x2[idx[0]])

    # strip moment carried by one stiffener strip (width d_eff)
    M_strip = M11_max * d_eff

    sigma_per_Pa = M_strip * y_ext / I_tot
    p_yield = sigma_y / sigma_per_Pa

    return p_yield, M11_max, x_at, y_at, sigma_per_Pa

# -----------------------------
# 9) Konvergenshjelpere
# -----------------------------
def ensure_odd(k):
    return k if (k % 2 == 1) else (k - 1)

def first_converged(trunc_list, p_list, p_ref, tol):
    rel_err = np.abs(p_list - p_ref) / (np.abs(p_ref) + 1e-30)
    for i, e in enumerate(rel_err):
        if e <= tol:
            return trunc_list[i], rel_err
    return trunc_list[-1], rel_err

# -----------------------------
# 10) Deflection (valgfritt)
# -----------------------------
def deflection_field(p, Mmax, Nmax):
    w = np.zeros_like(X1)
    for m in range(1, Mmax + 1, 2):
        alpha = m * np.pi / a
        sin_m = np.sin(alpha * X1)
        for n in range(1, Nmax + 1, 2):
            beta = n * np.pi / b
            W = Wmn_uniform(p, m, n)
            if W == 0.0:
                continue
            w += W * sin_m * np.sin(beta * X2)
    return w

# -----------------------------
# 11) Kjør
# -----------------------------
if __name__ == "__main__":
    # make odds
    Mref = ensure_odd(Mref)
    Nref = ensure_odd(Nref)
    Mmax_sweep = ensure_odd(Mmax_sweep)
    Nmax_sweep = ensure_odd(Nmax_sweep)

    # Reference p
    p_ref, _, x_ref, y_ref, _ = p_yield_for_MN(Mref, Nref, be)

    # Sweep M (N fixed = Nref)
    M_list = np.arange(1, min(Mmax_sweep, Mref) + 1, 2)
    pM = np.array([p_yield_for_MN(M, Nref, be)[0] for M in M_list])
    Mconv, relM = first_converged(M_list, pM, p_ref, TOL)

    # Sweep N (M fixed = Mref)
    N_list = np.arange(1, min(Nmax_sweep, Nref) + 1, 2)
    pN = np.array([p_yield_for_MN(Mref, N, be)[0] for N in N_list])
    Nconv, relN = first_converged(N_list, pN, p_ref, TOL)

    # Final (Mconv, Nconv)
    p_final, M11max_1Pa, x_at, y_at, sigma_per_Pa = p_yield_for_MN(Mconv, Nconv, be)

    # -------------------------
    # Plots (p_yield on y-axis)
    # -------------------------
    plt.figure()
    plt.plot(M_list, pM / 1e3, marker="o")
    plt.axhline(p_ref / 1e3, linestyle="--", label=f"Reference (Mref={Mref},Nref={Nref})")
    plt.xlabel("Mmax (odd), with N = Nref")
    plt.ylabel("p_yield [kN/m²]")
    plt.title(f"Convergence of p_yield in M (Nref = {Nref})")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(N_list, pN / 1e3, marker="o")
    plt.axhline(p_ref / 1e3, linestyle="--", label=f"Reference (Mref={Mref},Nref={Nref})")
    plt.xlabel("Nmax (odd), with M = Mref")
    plt.ylabel("p_yield [kN/m²]")
    plt.title(f"Convergence of p_yield in N (Mref = {Mref})")
    plt.grid(True)
    plt.legend()

    # -------------------------
    # Clean output
    # -------------------------
    print("\n" + "=" * 78)
    print("B1 – p_yield (konvergens på p) med D1 fra boka (smear-out)")
    print("=" * 78)

    print("\nGeometri / materiale")
    print(f"  Bay: a = {a:.3f} m, b = {b:.3f} m (simply supported)")
    print(f"  Plate: t = {t*1e3:.2f} mm")
    print(f"  Longitudinal: flat bar {h_s*1e3:.0f}x{t_s*1e3:.0f} mm, spacing d1 = {d1*1e3:.0f} mm")
    print(f"  Material: E = {E/1e9:.0f} GPa, nu = {nu:.2f}, sigma_y = {sigma_y/1e6:.0f} MPa")

    print("\nSchade effective breadth")
    print(f"  CL = {CL:.3f} m")
    print(f"  be/b = {be_ratio:.3f} -> be = {be:.3f} m ({be*1e3:.1f} mm)")

    print("\nBoka-koeffisienter (longitudinals-only)")
    print(f"  D_plate = {D_plate:.3e} N·m")
    print(f"  a1 = {a1:.6f}")
    print(f"  b1 = {b1:.6f}")
    print(f"  c1 = {c1:.6f}")

    print("\nOrtotrope rigiditeter")
    print(f"  D1  = {D1:.6e} N·m")
    print(f"  D2  = {D2:.6e} N·m")
    print(f"  D12 = {D12:.6e} N·m")
    print(f"  D1/D2 = {D1/D2:.1f}")

    print("\nKonvergens på p_yield")
    print(f"  Reference: (Mref,Nref)=({Mref},{Nref})")
    print(f"  Toleranse: {TOL:g}")
    print(f"  --> Mconv = {Mconv} (N fixed = {Nref})")
    print(f"  --> Nconv = {Nconv} (M fixed = {Mref})")
    print(f"  p_ref   = {p_ref/1e3:.6f} kN/m^2")
    print(f"  p_final = {p_final/1e3:.6f} kN/m^2")
    print(f"  rel.err = {np.abs(p_final-p_ref)/np.abs(p_ref):.3e}")

    print("\nInfo ved (Mconv,Nconv) for p = 1 Pa")
    print(f"  |M11|max = {M11max_1Pa:.6g} N·m per m")
    print(f"  location = x1 {x_at:.3f} m ({x_at/a:.3f} a), x2 {y_at:.3f} m ({y_at/b:.3f} b)")
    print(f"  sigma per 1 Pa = {sigma_per_Pa:.6g} Pa")

    print("\nRESULTAT")
    print(f"  p_yield = {p_final:.6g} Pa = {p_final/1e3:.6f} kN/m^2 (kPa)")
    print("=" * 78)

    # -------------------------
    # Deflection at p_yield (optional)
    # -------------------------
    w = deflection_field(p_final, Mconv, Nconv)
    w_max = float(np.max(np.abs(w)))
    ix_c = int(np.argmin(np.abs(x1 - a/2)))
    iy_c = int(np.argmin(np.abs(x2 - b/2)))
    w_c = float(w[iy_c, ix_c])

    print("\nDeflection at p_yield (using Mconv,Nconv)")
    print(f"  w_max    = {w_max*1e3:.4f} mm")
    print(f"  w_center = {w_c*1e3:.4f} mm")

    plt.show()