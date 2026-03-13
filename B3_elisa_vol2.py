import numpy as np
from math import pi

# Optional SciPy eigensolver
try:
    from scipy.linalg import eigh
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ============================================================
# B3 - COMBINED CODE
# Uses:
#   1) Local plate buckling between longitudinals
#   2) Global instability by Rayleigh-Ritz
#   3) Tripping of longitudinals
#
# Collapse load = min(global instability, tripping)
# Reserve strength = collapse / local initial buckling
# ============================================================


# ------------------------------------------------------------
# 1. MATERIAL
# ------------------------------------------------------------
E = 2.10e11          # Pa
nu = 0.30
G = E / (2.0 * (1.0 + nu))
rho = 7850.0         # kg/m^3
sigma_y = 250e6      # Pa


# ------------------------------------------------------------
# 2. PANEL GEOMETRY
# ------------------------------------------------------------
a = 55.8             # m, panel length
b = 9.95             # m, panel width
t_plate = 6.5e-3     # m

# Longitudinal flat bars
s_long = 0.670       # m
h_long = 0.160       # m
t_long = 0.012       # m

# Original transverse T-bars
s_tr_orig = 2.800    # m
hw_tr = 0.690        # m
tw_tr = 0.010        # m
bf_tr = 0.100        # m
tf_tr_orig = 0.010   # m

# Redesigned transverse T-bars
s_tr_red = 2.0 * s_tr_orig   # m = 5.6 m
tf_tr_red = 0.015            # m


# ------------------------------------------------------------
# 3. NUMERICAL SETTINGS
# ------------------------------------------------------------
M_list = list(range(1, 16))   # longitudinal Ritz terms
N_list = list(range(1, 7))    # transverse Ritz terms

m_local_max = 30
m_trip_max = 30


# ------------------------------------------------------------
# 4. BASIC HELPERS
# ------------------------------------------------------------
def D_plate_iso(t, E, nu):
    return E * t**3 / (12.0 * (1.0 - nu**2))

def rect_torsion_J(b, t):
    b1 = max(b, t)
    t1 = min(b, t)
    return (1.0 / 3.0) * b1 * t1**3

def plasticity_corrected_stress(sigma_c, sigma_y):
    """
    Johnson-Ostenfeld correction.
    Only applied if elastic buckling stress exceeds 50% of yield stress.
    """
    sigma_limit = 0.5 * sigma_y

    if sigma_c > sigma_limit:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_c))
    else:
        return sigma_c

# ------------------------------------------------------------
# 5. LONGITUDINAL + PLATE STRIP HELPERS
# ------------------------------------------------------------
def flatbar_with_attached_plate_I(s_eff, t_plate, h_st, t_st):
    """
    Combined inertia of:
      - plate strip of width s_eff
      - one flat bar stiffener
    About horizontal neutral axis.
    z measured downward from top of plate.
    """
    A_p = s_eff * t_plate
    z_p = 0.5 * t_plate
    I_p = s_eff * t_plate**3 / 12.0

    A_s = h_st * t_st
    z_s = t_plate + 0.5 * h_st
    I_s = t_st * h_st**3 / 12.0

    z_na = (A_p * z_p + A_s * z_s) / (A_p + A_s)
    I_tot = I_p + A_p * (z_p - z_na)**2 + I_s + A_s * (z_s - z_na)**2
    return I_tot, z_na

def orthotropic_rigidities_from_longitudinals(
    t_plate, s_long, h_long, t_long, E, nu
):
    """
    Orthotropic rigidities following lecture formulation.
    Plate + longitudinal stiffeners are smeared in the x-direction.
    """
    D0 = D_plate_iso(t_plate, E, nu)

    I_plate_strip = s_long * t_plate**3 / 12.0

    I_comb, z_na = flatbar_with_attached_plate_I(
        s_long, t_plate, h_long, t_long
    )

    I_add = I_comb - I_plate_strip

    D1 = D0 + E * I_add / s_long
    D2 = D0
    D12 = nu * D0

    return D0, D1, D2, D12


# ------------------------------------------------------------
# 6. TRANSVERSE T-BAR HELPERS
# ------------------------------------------------------------
def T_section_I_strong(hw, tw, bf, tf):
    """
    Bending about axis parallel to flange / plate,
    i.e. the strong axis for vertical bending.
    """
    A_w = hw * tw
    z_w = hw / 2.0

    A_f = bf * tf
    z_f = hw + tf / 2.0

    A = A_w + A_f
    z_na = (A_w * z_w + A_f * z_f) / A

    I_w = tw * hw**3 / 12.0 + A_w * (z_w - z_na)**2
    I_f = bf * tf**3 / 12.0 + A_f * (z_f - z_na)**2

    return I_w + I_f, z_na

def T_section_I_weak(hw, tw, bf, tf):
    """
    Weak-axis bending inertia about the centroidal axis
    in the plate plane and perpendicular to the stiffener line.
    For a symmetric T-section about the web centreline,
    no parallel-axis shift is needed in this direction.
    """
    Iw_web = hw * tw**3 / 12.0
    Iw_flg = tf * bf**3 / 12.0
    return Iw_web + Iw_flg

def T_section_area(hw, tw, bf, tf):
    return hw * tw + bf * tf

def transverse_beam_positions(a, spacing):
    xs = []
    x = spacing
    while x < a - 1e-12:
        xs.append(x)
        x += spacing
    return np.array(xs)

def transverse_beam_properties(hw, tw, bf, tf, E, G):
    """
    Returns properties needed for the full beam energy:

    U_beam = 1/2 ∫ [ EI_strong * (w_yy)^2
                   + GJ        * (w_xy)^2
                   + EI_weak   * (w_xy)^2 ] dy

    for a transverse beam located at x = const and spanning in y.
    """
    I_strong, _ = T_section_I_strong(hw, tw, bf, tf)
    I_weak = T_section_I_weak(hw, tw, bf, tf)
    J = rect_torsion_J(hw, tw) + rect_torsion_J(bf, tf)
    A = T_section_area(hw, tw, bf, tf)

    EI_strong = E * I_strong
    EI_weak = E * I_weak
    GJ = G * J

    return EI_strong, GJ, EI_weak, A, I_strong, I_weak, J


# ------------------------------------------------------------
# 7. RAYLEIGH-RITZ GLOBAL INSTABILITY
# ------------------------------------------------------------
def make_mode_index(M_list, N_list):
    return [(m, n) for m in M_list for n in N_list]

def build_global_buckling_matrices(
    a, b, D1, D2, D12,
    beam_xs, EI_b, GJ_b, EI_weak_b,
    M_list, N_list
):
    """
    Ritz basis:
        w = sum W_mn sin(m*pi*x/a) sin(n*pi*y/b)

    Global buckling:
        (K - Nx*Kg) q = 0

    Full transverse beam contribution:
        U_beam,j = 1/2 ∫ [ EI_b      (w_yy)^2
                         + GJ_b      (w_xy)^2
                         + EI_weak_b (w_xy)^2 ] dy
    """
    modes = make_mode_index(M_list, N_list)
    n_modes = len(modes)

    K = np.zeros((n_modes, n_modes))
    Kg = np.zeros((n_modes, n_modes))

    # Plate contribution
    for i, (m, n) in enumerate(modes):
        alpha = m * pi / a
        beta = n * pi / b

        K[i, i] += (a * b / 4.0) * (
            D1 * alpha**4 +
            2.0 * D12 * alpha**2 * beta**2 +
            D2 * beta**4
        )

        Kg[i, i] += (a * b / 4.0) * alpha**2

    # Discrete transverse beams at x = xj
    for xj in beam_xs:
        for i, (m1, n1) in enumerate(modes):
            a1 = m1 * pi / a
            b1 = n1 * pi / b

            sin1 = np.sin(a1 * xj)
            cos1 = np.cos(a1 * xj)

            for j, (m2, n2) in enumerate(modes):
                if n1 != n2:
                    continue

                a2 = m2 * pi / a
                b2 = n2 * pi / b

                sin2 = np.sin(a2 * xj)
                cos2 = np.cos(a2 * xj)

                # 1) Beam bending from w_yy
                K_bend = EI_b * (b / 2.0) * (b1**2) * (b2**2) * sin1 * sin2

                # 2) Beam torsion from w_xy
                K_tors = GJ_b * (b / 2.0) * (a1 * a2 * b1 * b2) * cos1 * cos2

                # 3) Bending about vertical/weak axis from w_xy
                K_weak = EI_weak_b * (b / 2.0) * (a1 * a2 * b1 * b2) * cos1 * cos2

                K[i, j] += K_bend + K_tors + K_weak

    K = 0.5 * (K + K.T)
    Kg = 0.5 * (Kg + Kg.T)
    return K, Kg, modes

def lowest_generalized_eigenpair(K, Kg):
    if HAVE_SCIPY:
        vals, vecs = eigh(K, Kg)
        vals = np.real(vals)

        mask = vals > 1e-9
        vals = vals[mask]
        vecs = vecs[:, mask]

        idx = np.argmin(vals)
        lam = vals[idx]
        vec = vecs[:, idx]
    else:
        A = np.linalg.solve(Kg, K)
        vals, vecs = np.linalg.eig(A)

        keep = []
        for i, val in enumerate(vals):
            if abs(val.imag) < 1e-8 and val.real > 1e-9:
                keep.append((val.real, vecs[:, i].real))

        keep.sort(key=lambda item: item[0])
        lam, vec = keep[0]

    vec = np.real(vec)
    vec = vec / np.max(np.abs(vec))
    return lam, vec


# ------------------------------------------------------------
# 8. LOCAL PLATE BUCKLING (INITIAL BUCKLING)
# ------------------------------------------------------------
def local_plate_buckling_sigma(a_loc, b_loc, t, E, nu, m_max=20):
    """
    Simply supported isotropic plate under uniaxial compression in x.
    """
    D = D_plate_iso(t, E, nu)

    best_sigma = np.inf
    best_m = None

    for m in range(1, m_max + 1):
        alpha = m * pi / a_loc
        beta = pi / b_loc

        Nx_cr = D * (alpha**2 + beta**2)**2 / alpha**2
        sigma_cr = Nx_cr / t

        if sigma_cr < best_sigma:
            best_sigma = sigma_cr
            best_m = m

    return best_sigma, best_m


# ------------------------------------------------------------
# 9. TRIPPING OF LONGITUDINALS
# ------------------------------------------------------------
def tripping_sigma(a_p, b_p, t_plate, h_long, t_long, E, nu, sigma_y, m_max=20, n=1):
    G = E / (2.0 * (1.0 + nu))
    D = D_plate_iso(t_plate, E, nu)

    I_f = 0.0
    K = h_long * t_long**3 / 3.0
    I_p = h_long**3 * t_long / 3.0
    d = h_long / 2.0

    best_sigma = np.inf
    best_sigma_corr = np.inf
    best_m_el = None
    best_m_corr = None

    for m in range(1, m_max + 1):
        alpha = m * pi / a_p
        beta = n * pi / b_p

        numerator = (
            D * b_p * (alpha**2 + beta**2)**2
            + 2.0 * (G * K + E * I_f * d**2 * alpha**2) * alpha**2 * beta**2
        )

        denominator = alpha**2 * (b_p * t_plate + 2.0 * I_p * beta**2)

        sigma = numerator / denominator
        sigma_corr = plasticity_corrected_stress(sigma, sigma_y)

        if sigma < best_sigma:
            best_sigma = sigma
            best_m_el = m

        if sigma_corr < best_sigma_corr:
            best_sigma_corr = sigma_corr
            best_m_corr = m

    return {
        "m_elastic": best_m_el,
        "sigma_elastic": best_sigma,
        "m_corr": best_m_corr,
        "sigma_corr": best_sigma_corr
    }


# ------------------------------------------------------------
# 10. WEIGHT CHECK
# ------------------------------------------------------------
A_tr_orig = T_section_area(hw_tr, tw_tr, bf_tr, tf_tr_orig)
A_tr_red = T_section_area(hw_tr, tw_tr, bf_tr, tf_tr_red)
A_long = h_long * t_long

steel_area_per_area_orig = (
    t_plate
    + A_long / s_long
    + A_tr_orig / s_tr_orig
)

steel_area_per_area_red = (
    t_plate
    + A_long / s_long
    + A_tr_red / s_tr_red
)

mass_per_area_orig = rho * steel_area_per_area_orig
mass_per_area_red = rho * steel_area_per_area_red
weight_ratio_total = mass_per_area_red / mass_per_area_orig


# ------------------------------------------------------------
# 11. BUILD REDESIGN MODEL
# ------------------------------------------------------------
D0, D1, D2, D12 = orthotropic_rigidities_from_longitudinals(
    t_plate=t_plate,
    s_long=s_long,
    h_long=h_long,
    t_long=t_long,
    E=E,
    nu=nu
)

beam_xs = transverse_beam_positions(a, s_tr_red)

EI_tr, GJ_tr, EIw_tr, A_tr, Itr_strong, Itr_weak, J_tr = transverse_beam_properties(
    hw=hw_tr, tw=tw_tr, bf=bf_tr, tf=tf_tr_red, E=E, G=G
)

K, Kg, modes = build_global_buckling_matrices(
    a=a, b=b,
    D1=D1, D2=D2, D12=D12,
    beam_xs=beam_xs,
    EI_b=EI_tr,
    GJ_b=GJ_tr,
    EI_weak_b=EIw_tr,
    M_list=M_list,
    N_list=N_list
)

Nx_instability, q_instability = lowest_generalized_eigenpair(K, Kg)
F_instability = Nx_instability * b
sigma_panel_eq = Nx_instability / t_plate
sigma_global_corr = plasticity_corrected_stress(sigma_panel_eq, sigma_y)
Nx_global_corr = sigma_global_corr * t_plate
F_global_corr = Nx_global_corr * b


# ------------------------------------------------------------
# 12. INITIAL LOCAL BUCKLING
# ------------------------------------------------------------
a_loc = s_tr_red
b_loc = s_long

sigma_local, m_local = local_plate_buckling_sigma(
    a_loc=a_loc,
    b_loc=b_loc,
    t=t_plate,
    E=E,
    nu=nu,
    m_max=m_local_max
)

Nx_local = sigma_local * t_plate
F_local = Nx_local * b


# ------------------------------------------------------------
# 13. TRIPPING CHECK
# ------------------------------------------------------------
trip = tripping_sigma(
    a_p=s_tr_red,
    b_p=s_long,
    t_plate=t_plate,
    h_long=h_long,
    t_long=t_long,
    E=E,
    nu=nu,
    sigma_y=sigma_y,
    m_max=m_trip_max,
    n=1
)

sigma_trip_corr = trip["sigma_corr"]
Nx_trip_corr = sigma_trip_corr * t_plate
F_trip_corr = Nx_trip_corr * b


# ------------------------------------------------------------
# 14. FINAL COLLAPSE CHECK
# ------------------------------------------------------------
collapse_force = min(F_global_corr, F_trip_corr)
collapse_stress = collapse_force / (b * t_plate)

if collapse_force == F_global_corr:
    governing_mode = "Global instability (Rayleigh-Ritz, corrected)"
else:
    governing_mode = "Tripping of longitudinals"

reserve_factor = collapse_force / F_local
reserve_increase_pct = (collapse_force - F_local) / F_local * 100.0


# ------------------------------------------------------------
# 15. PRINT RESULTS
# ------------------------------------------------------------
print("=" * 76)
print("B3 - COMBINED ANALYSIS")
print("=" * 76)

print("\nPANEL")
print(f"a  = {a:8.3f} m")
print(f"b  = {b:8.3f} m")
print(f"t  = {t_plate*1e3:8.3f} mm")

print("\nLONGITUDINALS")
print(f"spacing           = {s_long:8.3f} m")
print(f"flat bar          = {h_long*1e3:.0f} x {t_long*1e3:.0f} mm")

print("\nTRANSVERSE T-BARS (REDESIGN)")
print(f"new spacing       = {s_tr_red:8.3f} m")
print(f"T-bar             = {hw_tr*1e3:.0f}/{bf_tr*1e3:.0f} x {tw_tr*1e3:.0f}/{tf_tr_red*1e3:.0f} mm")
print(f"internal beams    = {len(beam_xs)}")

print("\nWEIGHT CHECK")
print(f"orig mass/area    = {mass_per_area_orig:12.4f} kg/m^2")
print(f"red  mass/area    = {mass_per_area_red:12.4f} kg/m^2")
print(f"ratio red/orig    = {weight_ratio_total:12.4f}")

print("\nORTHOTROPIC RIGIDITIES")
print(f"D0   = {D0:12.4e} N·m")
print(f"D1   = {D1:12.4e} N·m")
print(f"D2   = {D2:12.4e} N·m")
print(f"D12  = {D12:12.4e} N·m")

print("\nTRANSVERSE BEAM PROPERTIES")
print(f"EI_strong         = {EI_tr:12.4e} N·m^2")
print(f"GJ                = {GJ_tr:12.4e} N·m^2")
print(f"EI_weak           = {EIw_tr:12.4e} N·m^2")

print("\nGLOBAL INSTABILITY (RAYLEIGH-RITZ)")
print(f"Nx_instability    = {Nx_instability/1e6:12.4f} MN/m")
print(f"F_instability     = {F_instability/1e6:12.4f} MN")
print(f"eq. stress Nx/t   = {sigma_panel_eq/1e6:12.4f} MPa")

print("\nINITIAL LOCAL PLATE BUCKLING")
print(f"local field a     = {a_loc:8.3f} m")
print(f"local field b     = {b_loc:8.3f} m")
print(f"governing m       = {m_local}")
print(f"sigma_local       = {sigma_local/1e6:12.4f} MPa")
print(f"Nx_local          = {Nx_local/1e6:12.4f} MN/m")
print(f"F_local           = {F_local/1e6:12.4f} MN")

print("\nTRIPPING OF LONGITUDINALS")
print(f"governing m,corr  = {trip['m_corr']}")
print(f"sigma_trip,corr   = {sigma_trip_corr/1e6:12.4f} MPa")
print(f"Nx_trip,corr      = {Nx_trip_corr/1e6:12.4f} MN/m")
print(f"F_trip,corr       = {F_trip_corr/1e6:12.4f} MN")

print("\nFINAL COLLAPSE")
print(f"governing mode    = {governing_mode}")
print(f"collapse force    = {collapse_force/1e6:12.4f} MN")
print(f"collapse stress   = {collapse_stress/1e6:12.4f} MPa")

print("\nRESERVE STRENGTH")
print(f"reserve factor    = {reserve_factor:12.4f}")
print(f"increase          = {reserve_increase_pct:12.2f} %")

print("\nCOMMENT")
if collapse_force > F_local:
    print("There is post-buckling reserve from initial local plate buckling to collapse.")
else:
    print("No reserve strength found; redesign is likely not feasible in in-plane compression.")

print("=" * 76)


# ------------------------------------------------------------
# EXTRA CHECK: MODE 2
# Orthotropic plate between transverse stiffeners
# ------------------------------------------------------------
def mode2_sigma(a_field, b_field, D0, D1, h_bar, sigma_y, m_max=30, n=1):
    best_sigma = np.inf
    best_sigma_corr = np.inf
    best_m_el = None
    best_m_corr = None

    for m in range(1, m_max + 1):
        alpha = m * pi / a_field
        beta = n * pi / b_field

        sigma = (D0 / h_bar) * (a_field / (m * pi))**2 * (
            (D1 / D0) * alpha**4
            + 2.0 * alpha**2 * beta**2
            + beta**4
        )

        sigma_corr = plasticity_corrected_stress(sigma, sigma_y)

        if sigma < best_sigma:
            best_sigma = sigma
            best_m_el = m

        if sigma_corr < best_sigma_corr:
            best_sigma_corr = sigma_corr
            best_m_corr = m

    return {
        "m_elastic": best_m_el,
        "sigma_elastic": best_sigma,
        "m_corr": best_m_corr,
        "sigma_corr": best_sigma_corr
    }


mode2 = mode2_sigma(
    a_field=s_tr_red,
    b_field=b,
    D0=D0,
    D1=D1,
    h_bar=t_plate + (h_long * t_long) / s_long,
    sigma_y=sigma_y,
    m_max=30,
    n=1
)

print("\nMODE 2 - ORTHOTROPIC PLATE BETWEEN TRANSVERSE STIFFENERS")
print(f"governing m (elastic) = {mode2['m_elastic']}")
print(f"sigma_mode2           = {mode2['sigma_elastic']/1e6:.4f} MPa")
print(f"governing m (corr)    = {mode2['m_corr']}")
print(f"sigma_mode2,corr      = {mode2['sigma_corr']/1e6:.4f} MPa")

print("\nGLOBAL INSTABILITY (RAYLEIGH-RITZ)")
print(f"Nx_instability    = {Nx_instability/1e6:12.4f} MN/m")
print(f"F_instability     = {F_instability/1e6:12.4f} MN")
print(f"eq. stress Nx/t   = {sigma_panel_eq/1e6:12.4f} MPa")
print(f"sigma_global,corr = {sigma_global_corr/1e6:12.4f} MPa")
print(f"F_global,corr     = {F_global_corr/1e6:12.4f} MN")


# ------------------------------------------------------------
# PLOT: buckling stress as function of m
# ------------------------------------------------------------
import matplotlib.pyplot as plt

m_vals = np.arange(1, 31)

# --------------------------
# 1) Local plate buckling
# --------------------------
sigma_local_vals = []
for m in m_vals:
    alpha = m * pi / a_loc
    beta = pi / b_loc
    D = D_plate_iso(t_plate, E, nu)

    Nx_cr = D * (alpha**2 + beta**2)**2 / alpha**2
    sigma_cr = Nx_cr / t_plate
    sigma_local_vals.append(sigma_cr / 1e6)   # MPa

# --------------------------
# 2) Tripping of longitudinals
# --------------------------
sigma_trip_el_vals = []
sigma_trip_corr_vals = []

G_here = E / (2.0 * (1.0 + nu))
D_here = D_plate_iso(t_plate, E, nu)

I_f = 0.0
K_trip = h_long * t_long**3 / 3.0
I_p = h_long**3 * t_long / 3.0
d_trip = h_long / 2.0
n_trip = 1

for m in m_vals:
    alpha = m * pi / s_tr_red
    beta = n_trip * pi / s_long

    numerator = (
        D_here * s_long * (alpha**2 + beta**2)**2
        + 2.0 * (G_here * K_trip + E * I_f * d_trip**2 * alpha**2) * alpha**2 * beta**2
    )

    denominator = alpha**2 * (s_long * t_plate + 2.0 * I_p * beta**2)

    sigma_el = numerator / denominator
    sigma_corr = plasticity_corrected_stress(sigma_el, sigma_y)

    sigma_trip_el_vals.append(sigma_el / 1e6)      # MPa
    sigma_trip_corr_vals.append(sigma_corr / 1e6)  # MPa

# --------------------------
# 3) Mode 2
# --------------------------
sigma_mode2_el_vals = []
sigma_mode2_corr_vals = []

h_bar = t_plate + (h_long * t_long) / s_long
n_mode2 = 1

for m in m_vals:
    alpha = m * pi / s_tr_red
    beta = n_mode2 * pi / b

    sigma_el = (D0 / h_bar) * (s_tr_red / (m * pi))**2 * (
        (D1 / D0) * alpha**4
        + 2.0 * alpha**2 * beta**2
        + beta**4
    )

    sigma_corr = plasticity_corrected_stress(sigma_el, sigma_y)

    sigma_mode2_el_vals.append(sigma_el / 1e6)      # MPa
    sigma_mode2_corr_vals.append(sigma_corr / 1e6)  # MPa




# ------------------------------------------------------------
# COLOR PALETTE
# ------------------------------------------------------------
c1 = "#4C72B0"   # blue
c2 = "#55A868"   # green
c3 = "#C44E52"   # red

# ------------------------------------------------------------
# FIND MINIMA
# ------------------------------------------------------------
i_local = np.argmin(sigma_local_vals)
i_trip_el = np.argmin(sigma_trip_el_vals)
i_trip_corr = np.argmin(sigma_trip_corr_vals)
i_mode2_el = np.argmin(sigma_mode2_el_vals)
i_mode2_corr = np.argmin(sigma_mode2_corr_vals)

m_local_min = m_vals[i_local]
m_trip_el_min = m_vals[i_trip_el]
m_trip_corr_min = m_vals[i_trip_corr]
m_mode2_el_min = m_vals[i_mode2_el]
m_mode2_corr_min = m_vals[i_mode2_corr]

sigma_local_min = sigma_local_vals[i_local]
sigma_trip_el_min = sigma_trip_el_vals[i_trip_el]
sigma_trip_corr_min = sigma_trip_corr_vals[i_trip_corr]
sigma_mode2_el_min = sigma_mode2_el_vals[i_mode2_el]
sigma_mode2_corr_min = sigma_mode2_corr_vals[i_mode2_corr]


# --------------------------
# Plot 1: elastic curves
# --------------------------
plt.figure(figsize=(8,5))

plt.plot(m_vals, sigma_local_vals, color=c1, linewidth=2)
plt.scatter(m_vals, sigma_local_vals, color=c1, s=50, label="Mode 1")

plt.plot(m_vals, sigma_trip_el_vals, color=c2, linewidth=2)
plt.scatter(m_vals, sigma_trip_el_vals, color=c2, s=50, label="Mode 2")

plt.plot(m_vals, sigma_mode2_el_vals, color=c3, linewidth=2)
plt.scatter(m_vals, sigma_mode2_el_vals, color=c3, s=50, label="Mode 4")

# Mark minima
plt.scatter(m_local_min, sigma_local_min, color=c1, edgecolor="black", s=120, zorder=5)
plt.scatter(m_trip_el_min, sigma_trip_el_min, color=c2, edgecolor="black", s=120, zorder=5)
plt.scatter(m_mode2_el_min, sigma_mode2_el_min, color=c3, edgecolor="black", s=120, zorder=5)

# Optional text labels near minima
plt.text(m_local_min + 0.3, sigma_local_min + 10,
         f"m={m_local_min}", color=c1)
plt.text(m_trip_el_min + 0.3, sigma_trip_el_min + 10,
         f"m={m_trip_el_min}", color=c2)
plt.text(m_mode2_el_min + 0.3, sigma_mode2_el_min + 10,
         f"m={m_mode2_el_min}", color=c3)

plt.xlabel("m")
plt.ylabel("Critical stress [MPa]")
plt.title("Buckling stress as function of m")
plt.xticks(m_vals)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# --------------------------
# Plot 2: corrected curves
# --------------------------
plt.figure(figsize=(8,5))

plt.plot(m_vals, sigma_local_vals, color=c1, linewidth=2)
plt.scatter(m_vals, sigma_local_vals, color=c1, s=50, label="Mode 1")

plt.plot(m_vals, sigma_trip_corr_vals, color=c2, linewidth=2)
plt.scatter(m_vals, sigma_trip_corr_vals, color=c2, s=50, label="Mode 2")

plt.plot(m_vals, sigma_mode2_corr_vals, color=c3, linewidth=2)
plt.scatter(m_vals, sigma_mode2_corr_vals, color=c3, s=50, label="Mode 4")

# Mark minima
plt.scatter(m_local_min, sigma_local_min, color=c1, edgecolor="black", s=120, zorder=5)
plt.scatter(m_trip_corr_min, sigma_trip_corr_min, color=c2, edgecolor="black", s=120, zorder=5)
plt.scatter(m_mode2_corr_min, sigma_mode2_corr_min, color=c3, edgecolor="black", s=120, zorder=5)

# Optional text labels near minima
plt.text(m_local_min + 0.3, sigma_local_min + 10,
         f"m={m_local_min}", color=c1)
plt.text(m_trip_corr_min + 0.3, sigma_trip_corr_min + 10,
         f"m={m_trip_corr_min}", color=c2)
plt.text(m_mode2_corr_min + 0.3, sigma_mode2_corr_min + 10,
         f"m={m_mode2_corr_min}", color=c3)

plt.xlabel("m")
plt.ylabel("Critical stress [MPa]")
plt.title("Buckling stress as function of m (plasticity correction)")
plt.xticks(m_vals)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# PRINT MINIMUM VALUES
# ------------------------------------------------------------
print("\nMINIMUM VALUES FROM m-PLOTS")
print(f"Mode 1 elastic:   minimum sigma = {sigma_local_min:.3f} MPa at m = {m_local_min}")
print(f"Mode 2 elastic:   minimum sigma = {sigma_trip_el_min:.3f} MPa at m = {m_trip_el_min}")
print(f"Mode 3 elastic:   minimum sigma = {sigma_mode2_el_min:.3f} MPa at m = {m_mode2_el_min}")

print(f"Mode 1 corrected: minimum sigma = {sigma_local_min:.3f} MPa at m = {m_local_min}")
print(f"Mode 2 corrected: minimum sigma = {sigma_trip_corr_min:.3f} MPa at m = {m_trip_corr_min}")
print(f"Mode 3 corrected: minimum sigma = {sigma_mode2_corr_min:.3f} MPa at m = {m_mode2_corr_min}")


# ------------------------------------------------------------
# PLOT: Global buckling mode shape from Rayleigh-Ritz
# ------------------------------------------------------------
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Grid
nx_plot = 120
ny_plot = 120

x_vals = np.linspace(0, a, nx_plot)
y_vals = np.linspace(0, b, ny_plot)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Reconstruct mode shape
W = np.zeros_like(X)

for coeff, (m, n) in zip(q_instability, modes):
    W += coeff * np.sin(m * pi * X / a) * np.sin(n * pi * Y / b)

# Normalize for plotting only
W_plot = W / np.max(np.abs(W))

# 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, W_plot, cmap='viridis', edgecolor='none')

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('Normalized w')
ax.set_title('Global buckling mode shape (Rayleigh-Ritz)')

plt.tight_layout()
plt.show()