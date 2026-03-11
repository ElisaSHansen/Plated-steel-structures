print("SCRIPT STARTED")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Project B.1 — BAY between transverse beams
# Orthotropic Navier (SSSS) on ONE bay:
#   x1 in [0,a]=[0,d2]  (between transverse beams)
#   x2 in [0,b]         (deck width)
# Smearing ONLY longitudinals (flat bar 160x12 @ 670 mm)
# Plots:
#   - axes swapped to match your figure: x2 horizontal, x1 vertical
#   - 0 at TOP of x1-axis
#   - controlled z-axis ticks (left side) with fewer decimals
# ============================================================


# -----------------------------
# INPUT DATA
# -----------------------------
b = 9.95        # [m] x2-direction (deck width)
d2 = 2.800      # [m] spacing between transverse beams
a = d2          # [m] analyze ONE bay in x1-direction

E = 2.1e11
nu = 0.30
sigma_y = 250e6
h = 0.0065

# Longitudinals: flat bar 160x12 mm, spacing 670 mm
d1 = 0.670
hL = 0.160
tL = 0.012

# Navier truncation (odd m,n)
M_terms = 51
N_terms = 51

# Grid for fields/plots
nx = 121
ny = 81

# Plot toggles
PLOT_M11 = True


# -----------------------------
# BASIC CONSTANTS
# -----------------------------
def plate_D(E, h, nu):
    return E * h**3 / (12.0 * (1.0 - nu**2))

def shear_modulus(E, nu):
    return E / (2.0 * (1.0 + nu))


# -----------------------------
# EFFECTIVE WIDTH (example-style)
# -----------------------------
def effective_width_example_style(s, a_span):
    b_eff = 1.1 * s / (1.0 + 2.0 * (s / a_span) ** 2)
    return min(b_eff, s)


# -----------------------------
# SECTION MODULUS: plate strip + vertical flat bar
# -----------------------------
def section_modulus_flatbar_with_plate(b_eff, h_plate, h_st, t_st):
    A_plate = b_eff * h_plate
    A_st = t_st * h_st

    z_plate = 0.0
    z_st = (h_plate / 2.0) + (h_st / 2.0)

    z_bar = (A_plate * z_plate + A_st * z_st) / (A_plate + A_st)

    I_plate_c = (b_eff * h_plate**3) / 12.0
    I_st_c = (t_st * h_st**3) / 12.0

    I_total = (
        I_plate_c + A_plate * (z_plate - z_bar) ** 2
        + I_st_c + A_st * (z_st - z_bar) ** 2
    )

    z_top = (h_plate / 2.0) + h_st
    c_top = z_top - z_bar

    return I_total / c_top


# -----------------------------
# TORSION CONSTANT (OPEN thin-walled approx.)
# Flat bar: J ≈ (1/3) h t^3
# -----------------------------
def torsion_constant_flatbar_open(h_bar, t_bar):
    return (1.0 / 3.0) * h_bar * (t_bar ** 3)


# -----------------------------
# SMEARING (ONLY longitudinals)
# Simple lecture-consistent form used previously:
#   D1 increased by b1 and eccentricity term; D2 = D0; D12 includes K1/d1
# -----------------------------
def orthotropic_rigidities_longitudinals_only(D0, E, nu, h_plate,
                                             d1, A1, I1, z1, K1):
    G = shear_modulus(E, nu)

    a1 = A1 / (d1 * h_plate)
    b1 = (E * I1) / (D0 * d1)
    c1 = z1 / h_plate

    denom = (1.0 + a1)

    D1 = D0 * (1.0 + b1 + 12.0 * (1.0 - nu**2) * a1 * c1**2 / denom)
    D2 = D0
    D12 = D0 + (G / 2.0) * (K1 / d1)

    return D1, D2, D12, (a1, b1, c1), G


# -----------------------------
# NAVIER fields for uniform pressure:
# Returns w/p and M11/p on grid
# -----------------------------
def navier_w_M11_over_p(a, b, nu, D1, D2, D12, M_terms, N_terms, nx, ny):
    pi = np.pi
    xs = np.linspace(0.0, a, nx)
    ys = np.linspace(0.0, b, ny)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    w_over_p = np.zeros_like(X, dtype=float)
    M11_over_p = np.zeros_like(X, dtype=float)

    m_list = np.arange(1, M_terms + 1, 2)
    n_list = np.arange(1, N_terms + 1, 2)

    for m in m_list:
        sin_mx = np.sin(m * pi * X / a)
        m_over_a = m / a
        for n in n_list:
            sin_ny = np.sin(n * pi * Y / b)
            n_over_b = n / b

            denom = (D1 * (m_over_a**4)
                     + 2.0 * D12 * ((m * n) / (a * b))**2
                     + D2 * (n_over_b**4))

            # Uniform pressure on full plate (SSSS): odd terms only
            Wmn_over_p = 16.0 / (m * n * (pi**6) * denom)

            shape = sin_mx * sin_ny
            w_over_p += Wmn_over_p * shape

            # M11 = -D1( w_xx + nu w_yy )
            factor = -((m * pi / a) ** 2 + nu * (n * pi / b) ** 2)
            M11_over_p += (-D1) * (factor * Wmn_over_p) * shape

    return xs, ys, X, Y, w_over_p, M11_over_p


# ============================================================
# BUILD STIFFENER PROPERTIES (LONGITUDINALS)
# ============================================================
D0 = plate_D(E, h, nu)

A_L = tL * hL
I_L = (tL * hL**3) / 12.0
zL = (h / 2.0) + (hL / 2.0)          # centroid from plate mid-surface
K1 = torsion_constant_flatbar_open(hL, tL)

D1, D2, D12, (a1_, b1_, c1_), G = orthotropic_rigidities_longitudinals_only(
    D0, E, nu, h,
    d1, A_L, I_L, zL, K1
)

print("\n--- BAY MODEL ---")
print(f"a = {a:.3f} m  (between transverse beams)")
print(f"b = {b:.3f} m  (deck width)")
print("BC: SSSS on all four edges of the BAY")

print("\n--- Orthotropic rigidities ---")
print(f"D0  = {D0:,.2f} N·m")
print(f"D1  = {D1:,.2f} N·m")
print(f"D2  = {D2:,.2f} N·m")
print(f"D12 = {D12:,.2f} N·m")
print(f"G   = {G:,.2e} Pa")
print(f"Smear params: a1={a1_:.3e}, b1={b1_:.3e}, c1={c1_:.3e}")

# ============================================================
# FIELD COMPUTATION
# ============================================================
xs, ys, X, Y, w_over_p, M11_over_p = navier_w_M11_over_p(
    a, b, nu, D1, D2, D12, M_terms, N_terms, nx, ny
)

# Section modulus & stress extraction for one longitudinal line
b_eff = effective_width_example_style(d1, a)

# Samlet treghetsmoment I for plate-stripe + flatbar om nøytralaksen
A_plate = b_eff * h
A_st = tL * hL

z_plate = 0.0
z_st = (h / 2.0) + (hL / 2.0)

z_bar = (A_plate * z_plate + A_st * z_st) / (A_plate + A_st)

I_plate_c = (b_eff * h**3) / 12.0
I_st_c = (tL * hL**3) / 12.0

I_total = (
    I_plate_c + A_plate * (z_plate - z_bar)**2
    + I_st_c + A_st * (z_st - z_bar)**2
)

print(f"I_total = {I_total:.3e} m^4")
W_top = section_modulus_flatbar_with_plate(b_eff, h, hL, tL)

sigma_over_p = (M11_over_p * d1) / W_top
abs_sigma_over_p = np.abs(sigma_over_p)
idx_sig = np.unravel_index(np.argmax(abs_sigma_over_p), abs_sigma_over_p.shape)

sigma_over_p_max = abs_sigma_over_p[idx_sig]
p_max = sigma_y / sigma_over_p_max

# Fields at p_max
w = w_over_p * p_max
M11 = M11_over_p * p_max
sigma = sigma_over_p * p_max

# Max values
w_max = float(np.max(np.abs(w)))
M11_max = float(np.max(np.abs(M11)))
sigma_max = float(np.max(np.abs(sigma)))

x_sig = float(xs[idx_sig[1]])
y_sig = float(ys[idx_sig[0]])

print("\n--- Section modulus / effective width ---")
print(f"b_eff = {b_eff:.3f} m")
print(f"W_top = {W_top:.3e} m^3")

print("\n--- Results at yielding ---")
print(f"p_max              = {p_max/1e6:.4f} MPa")
print(f"sigma_max          = {sigma_max/1e6:.1f} MPa (target {sigma_y/1e6:.0f} MPa)")
print(f"Location sigma_max = x1={x_sig:.2f} m, x2={y_sig:.2f} m")
print(f"w_max              = {w_max*1e3:.2f} mm")
print(f"|M11|max           = {M11_max/1e3:.2f} kN·m/m")

rho = 1000.0
g = 9.81
h_water = p_max / (rho * g)
print(f"Equivalent water head ≈ {h_water:.2f} m")


# ============================================================
# PLOTTING HELPERS
# - swap axes: plot (x2, x1) to match your figure
# - 0 at top of x1-axis
# - FIX z-axis tick labels (left side) with fewer decimals
# ============================================================

def nice_step(value_range):
    """Pick a simple step size for ticks."""
    if value_range <= 0:
        return 1.0
    # crude "nice number" selection
    p10 = 10 ** np.floor(np.log10(value_range))
    candidates = np.array([1, 2, 2.5, 5, 10]) * p10
    # choose about 8 intervals
    target = value_range / 8.0
    return float(candidates[np.argmin(np.abs(candidates - target))])

def plot_surface_swapped(X, Y, Z, title, zlabel,
                         z_unit="auto",
                         force_zlim=None,
                         force_zticks=None,
                         zfmt="%.2f",
                         vmin=None, vmax=None):
    """
    Plot surface with axes swapped:
      - horizontal: x2 (=Y)
      - vertical:   x1 (=X)
    and with x1 shown with 0 at top.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Y, X, Z,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("x2 [m]")
    ax.set_ylabel("x1 [m]")
    ax.set_zlabel(zlabel)

    # 0 at top of x1-axis (the vertical-looking axis in your view)
    ax.set_ylim(a, 0)

    # Z-axis limits and ticks (this is the "left side numbers" you pointed at)
    if force_zlim is not None:
        ax.set_zlim(force_zlim[0], force_zlim[1])

    if force_zticks is not None:
        ax.set_zticks(force_zticks)
    else:
        zmin = float(np.min(Z))
        zmax = float(np.max(Z))
        step = nice_step(abs(zmax - zmin))
        # build ticks in a stable way
        t0 = step * np.floor(zmin / step)
        t1 = step * np.ceil(zmax / step)
        ticks = np.arange(t0, t1 + 0.5 * step, step)
        ax.set_zticks(ticks)

    ax.zaxis.set_major_formatter(mticker.FormatStrFormatter(zfmt))

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()


# -----------------------------
# PLOT 1: Deflection in mm
# Fixed 0.5 mm intervals
# -----------------------------
w_mm = 1e3 * w
w_abs_max = np.max(np.abs(w_mm))

# Rund opp til nærmeste 0.5 mm
upper = 0.5 * np.ceil(w_abs_max / 0.5)

ticks_w = np.arange(0, upper + 0.5, 0.5)

plot_surface_swapped(
    X, Y, w_mm,
    title=f"Deflection w(x1,x2) at p_max = {p_max/1e6:.3f} MPa",
    zlabel="w [mm]",
    force_zlim=(upper, 0),      # 0 på toppen
    force_zticks=ticks_w,
    zfmt="%.1f",
    vmin=0,
    vmax=upper
)
# -----------------------------
# PLOT 2: Stress in MPa
# FIX left-side numbers: 0..250 MPa with 25 MPa step, 0 on top
# -----------------------------
sigma_MPa = sigma / 1e6
sigma_top = 0.0
sigma_bot = sigma_y / 1e6  # 250 MPa

ticks_sigma = np.arange(0.0, sigma_bot + 1e-9, 25.0)

plot_surface_swapped(
    X, Y, sigma_MPa,
    title="Longitudinal stress σ(x1,x2) at p_max",
    zlabel="σ [MPa]",
    force_zlim=(sigma_bot, sigma_top),     # IMPORTANT: 0 at TOP
    force_zticks=ticks_sigma,
    zfmt="%.0f",
    vmin=0.0, vmax=sigma_bot               # colorbar matches axis
)

# -----------------------------
# PLOT 3: M11 in kN·m/m
# Fixed 5 kN·m/m intervals
# -----------------------------
if PLOT_M11:
    M11_kNm = M11 / 1e3

    Mmax = np.max(np.abs(M11_kNm))

    # Rund opp til nærmeste 5
    upper = 5 * np.ceil(Mmax / 5)

    ticks_M11 = np.arange(0, upper + 5, 5)

    plot_surface_swapped(
        X, Y, M11_kNm,
        title="Bending moment resultant M11(x1,x2) at p_max",
        zlabel="M11 [kN·m/m]",
        force_zlim=(upper, 0),     # 0 på toppen
        force_zticks=ticks_M11,
        zfmt="%.0f",
        vmin=0,
        vmax=upper
    )


# ============================================================
# CONVERGENCE STUDY (p_max vs terms)
# ============================================================
print("\n--- Convergence study ---")
term_list = [5, 9, 13, 17, 21, 31, 41, 51]
p_list = []

for MT in term_list:
    xs_c, ys_c, X_c, Y_c, wop_c, Mop_c = navier_w_M11_over_p(
        a, b, nu, D1, D2, D12,
        MT, MT,
        61, 41
    )

    sigma_over_p_c = (Mop_c * d1) / W_top
    max_sigma_over_p_c = np.max(np.abs(sigma_over_p_c))
    p_conv = sigma_y / max_sigma_over_p_c
    p_list.append(p_conv)

    print(f"Terms={MT:2d} -> p_max = {p_conv/1e6:.4f} MPa")

plt.figure()
plt.plot(term_list, np.array(p_list) / 1e6, marker="o")
plt.xlabel("Number of Navier terms (odd m,n up to)")
plt.ylabel("Yield pressure p_max [MPa]")
plt.title("Convergence of yield pressure with Navier terms (BAY model)")
plt.grid(True)
plt.tight_layout()
plt.show()