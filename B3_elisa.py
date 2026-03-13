import math

# ============================================================
# B3 - Redesign with fewer transverse T-bar stiffeners
# Combined buckling script
# ============================================================

# -----------------------------
# MATERIAL
# -----------------------------
E = 210000.0          # MPa = N/mm^2
nu = 0.30
sigma_y = 250.0       # MPa
G = E / (2.0 * (1.0 + nu))

# -----------------------------
# GLOBAL PANEL GEOMETRY
# -----------------------------
a_pp = 55800.0        # mm, full panel length
b_pp = 9950.0         # mm, full panel width

# -----------------------------
# PLATE
# -----------------------------
t_plate = 6.5         # mm

# -----------------------------
# LONGITUDINAL STIFFENERS (unchanged)
# -----------------------------
b_p = 670.0           # spacing between longitudinals = plate field width
h_long = 160.0        # mm
t_long = 12.0         # mm

# In B3, half the transverse T-bars are removed
# Original spacing = 2800 mm -> new spacing = 5600 mm
a_p = 5600.0          # mm, distance between remaining transverse stiffeners

# -----------------------------
# TRANSVERSE T-BAR (redesigned)
# -----------------------------
# Problem text: 690/100 x 10/15 mm
# interpreted as:
#   web:    690 x 10 mm
#   flange: 100 x 15 mm
h_web_tr = 690.0      # mm
t_web_tr = 10.0       # mm
b_flange_tr = 100.0   # mm
t_flange_tr = 15.0    # mm

# distance from plate reference plane to flange centroid
d_tr = h_web_tr + t_flange_tr / 2.0   # mm

# -----------------------------
# PLATE RIGIDITY
# -----------------------------
D = E * t_plate**3 / (12.0 * (1.0 - nu**2))

# -----------------------------
# LONGITUDINAL SMEARED PROPERTIES
# -----------------------------
A1 = h_long * t_long
I1 = t_long * h_long**3 / 12.0
e1 = h_long / 2.0

a1 = A1 / (b_p * t_plate)
b1 = (E * I1) / (D * b_p)
c1 = e1 / t_plate + 0.5

D1 = D * (1.0 + b1 + (12.0 * (1.0 - nu**2) * a1 * c1**2) / (1.0 + a1))
h_bar = t_plate + A1 / b_p

# -----------------------------
# HELPER
# -----------------------------
def plasticity_corrected_stress(sigma_c, sigma_y):
    if sigma_c > sigma_y / 2.0:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_c))
    return sigma_c

# ============================================================
# MODE 1: LOCAL BUCKLING OF PLATE BETWEEN STIFFENERS
# ============================================================
def sigma_c1(m, ap, bp, h, D):
    term = (m * bp / ap + ap / (m * bp))**2
    return term * D * math.pi**2 / (bp**2 * h)

def local_plate_buckling(m_max=40):
    sigma_min = float("inf")
    best_m = None

    for m in range(1, m_max + 1):
        sigma = sigma_c1(m, a_p, b_p, t_plate, D)
        if sigma < sigma_min:
            sigma_min = sigma
            best_m = m

    return best_m, sigma_min, plasticity_corrected_stress(sigma_min, sigma_y)

# ============================================================
# MODE 2: ORTHOTROPIC PANEL BETWEEN TRANSVERSE STIFFENERS
# ============================================================
def sigma_c2(m, n=1):
    term1 = (D1 / D) * (m * math.pi / a_p)**4
    term2 = 2.0 * (m * math.pi / a_p)**2 * (n * math.pi / b_pp)**2
    term3 = (n * math.pi / b_pp)**4

    return (D / h_bar) * (a_p / (m * math.pi))**2 * (term1 + term2 + term3)

def panel_between_transverse_buckling(m_max=40, n=1):
    sigma_min = float("inf")
    best_m = None

    for m in range(1, m_max + 1):
        sigma = sigma_c2(m, n)
        if sigma < sigma_min:
            sigma_min = sigma
            best_m = m

    return best_m, sigma_min, plasticity_corrected_stress(sigma_min, sigma_y)

# ============================================================
# MODE 3: COMPLETE PLATE PANEL
# Transverse stiffeners added as single beams
# ============================================================

# Effective width attached to each transverse beam
b_e = 1.1 * a_p / (1.0 + 2.0 * (a_p / b_pp)**2)

A_g = b_e * t_plate + h_web_tr * t_web_tr + b_flange_tr * t_flange_tr

e_g = (
    -0.5 * t_plate**2 * b_e
    + 0.5 * h_web_tr**2 * t_web_tr
    + b_flange_tr * t_flange_tr * d_tr
) / A_g

I_bar = (
    t_plate**3 * b_e / 3.0
    + h_web_tr**3 * t_web_tr / 3.0
    + b_flange_tr * t_flange_tr * d_tr**2
    - e_g**2 * A_g
)

I_f_tr = b_flange_tr**3 * t_flange_tr / 12.0

K_tr = (
    h_web_tr * t_web_tr**3
    + b_flange_tr * t_flange_tr**3
) / 3.0

# Remaining transverse stiffeners placed every 5600 mm
# inside the full 55.8 m panel
x_tr_list = [5600.0 * i for i in range(1, 10)]   # 5600, 11200, ..., 50400

def S_sum(m):
    return sum(math.sin(m * math.pi * x / a_pp)**2 for x in x_tr_list)

def C_sum(m):
    return sum(math.cos(m * math.pi * x / a_pp)**2 for x in x_tr_list)

def sigma_c3(m):
    alpha = m * math.pi / a_pp
    beta = math.pi / b_pp

    term1 = beta**4 * (D + (4.0 * E * I_bar / a_pp) * S_sum(m)) * alpha**(-2)
    term2 = D1 * alpha**2
    term3 = (2.0 / a_pp) * beta**2 * (
        D * a_pp + (G * K_tr + E * I_f_tr * d_tr**2 * beta**2) * C_sum(m)
    )

    return (term1 + term2 + term3) / h_bar

def complete_panel_buckling(m_max=40):
    sigma_min = float("inf")
    best_m = None

    for m in range(1, m_max + 1):
        sigma = sigma_c3(m)
        if sigma < sigma_min:
            sigma_min = sigma
            best_m = m

    return best_m, sigma_min, plasticity_corrected_stress(sigma_min, sigma_y)

# ============================================================
# MODE 4: TRIPPING OF LONGITUDINALS
# ============================================================
I_f_long = 0.0
K_long = h_long * t_long**3 / 3.0
I_p_long = h_long**3 * t_long / 3.0
d_long = h_long / 2.0

def sigma_c4(m, n=1):
    alpha = m * math.pi / a_p
    beta = n * math.pi / b_p

    numerator = (
        D * b_p * (alpha**2 + beta**2)**2
        + 2.0 * (G * K_long + E * I_f_long * d_long**2 * alpha**2) * alpha**2 * beta**2
    )

    denominator = alpha**2 * (b_p * t_plate + 2.0 * I_p_long * beta**2)

    return numerator / denominator

def tripping_buckling(m_max=40, n=1):
    sigma_min = float("inf")
    best_m = None

    for m in range(1, m_max + 1):
        sigma = sigma_c4(m, n)
        if sigma < sigma_min:
            sigma_min = sigma
            best_m = m

    return best_m, sigma_min, plasticity_corrected_stress(sigma_min, sigma_y)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    m1, s1, s1_corr = local_plate_buckling()
    m2, s2, s2_corr = panel_between_transverse_buckling()
    m3, s3, s3_corr = complete_panel_buckling()
    m4, s4, s4_corr = tripping_buckling()

    sigma_collapse = min(s3_corr, s4_corr)
    reserve = sigma_collapse - s1_corr
    reserve_ratio = reserve / s1_corr
    reserve_percent = reserve_ratio * 100.0

    print("====================================================")
    print("B3 REDESIGN - BUCKLING RESULTS")
    print("====================================================")
    print(f"D      = {D:.6e} Nmm")
    print(f"D1     = {D1:.6e} Nmm")
    print(f"h_bar  = {h_bar:.6f} mm")
    print(f"b_e    = {b_e:.3f} mm")
    print(f"I_bar  = {I_bar:.6e} mm^4")
    print(f"I_f_tr = {I_f_tr:.6e} mm^4")
    print(f"K_tr   = {K_tr:.6e} mm^4")
    print()

    print("1) Local plate buckling between stiffeners")
    print(f"   m = {m1}")
    print(f"   sigma_c1       = {s1:.3f} MPa")
    print(f"   sigma_c1,corr  = {s1_corr:.3f} MPa")
    print()

    print("2) Orthotropic panel between transverse stiffeners")
    print(f"   m = {m2}")
    print(f"   sigma_c2       = {s2:.3f} MPa")
    print(f"   sigma_c2,corr  = {s2_corr:.3f} MPa")
    print()

    print("3) Complete plate panel")
    print(f"   m = {m3}")
    print(f"   sigma_c3       = {s3:.3f} MPa")
    print(f"   sigma_c3,corr  = {s3_corr:.3f} MPa")
    print()

    print("4) Tripping of longitudinals")
    print(f"   m = {m4}")
    print(f"   sigma_c4       = {s4:.3f} MPa")
    print(f"   sigma_c4,corr  = {s4_corr:.3f} MPa")
    print()

    print("Collapse and reserve strength")
    print(f"   sigma_collapse = {sigma_collapse:.3f} MPa")
    print(f"   reserve        = {reserve:.3f} MPa")
    print(f"   reserve ratio  = {reserve_ratio:.3f}")
    print(f"   reserve [%]    = {reserve_percent:.1f}")
    print("====================================================")


    import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# B3 - 3D plot of all critical buckling modes in one figure
# ============================================================

a_pp = 55800.0
b_pp = 9950.0
a_p = 5600.0
b_p = 670.0

m1, m2, m3, m4 = 8, 1, 10, 8
n = 1

def mode_shape(X, Y, a, b, m, n=1):
    return np.sin(m * np.pi * X / a) * np.sin(n * np.pi * Y / b)

fig = plt.figure(figsize=(14, 10))

cases = [
    (a_p,  b_p,  m1, n, "Mode 1: Local plate buckling"),
    (a_p,  b_pp, m2, n, "Mode 2: Orthotropic panel"),
    (a_pp, b_pp, m3, n, "Mode 3: Complete panel"),
    (a_p,  b_p,  m4, n, "Mode 4: Tripping (simplified)")
]

for i, (a, b, m, n_val, title) in enumerate(cases, start=1):
    ax = fig.add_subplot(2, 2, i, projection="3d")

    x = np.linspace(0, a, 180)
    y = np.linspace(0, b, 100)
    X, Y = np.meshgrid(x, y)
    Z = mode_shape(X, Y, a, b, m, n_val)

    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_title(f"{title}\n$m={m}, n={n_val}$")
    ax.set_xlabel("x1 [mm]")
    ax.set_ylabel("x2 [mm]")
    ax.set_zlabel("Norm. amp.")
    ax.view_init(elev=28, azim=-130)

plt.tight_layout()
plt.show()