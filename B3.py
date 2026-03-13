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
#   1) Local plate buckling between longitudinals (your approach)
#   2) Global instability by Rayleigh-Ritz (friend's approach)
#   3) Tripping of longitudinals (your B2/B3 approach)
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

# Longitudinal flat bars (unchanged)
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
    if sigma_c > sigma_y / 2.0:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_c))
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

    # Plate rigidity
    D0 = D_plate_iso(t_plate, E, nu)

    # Plate strip inertia
    I_plate_strip = s_long * t_plate**3 / 12.0

    # Plate + stiffener inertia
    I_comb, z_na = flatbar_with_attached_plate_I(
        s_long, t_plate, h_long, t_long
    )

    # Additional inertia from stiffener
    I_add = I_comb - I_plate_strip

    # Orthotropic rigidities
    D1 = D0 + E * I_add / s_long
    D2 = D0
    D12 = nu * D0

    return D0, D1, D2, D12


# ------------------------------------------------------------
# 6. TRANSVERSE T-BAR HELPERS
# ------------------------------------------------------------
def T_section_I_strong(hw, tw, bf, tf):
    A_w = hw * tw
    z_w = hw / 2.0

    A_f = bf * tf
    z_f = hw + tf / 2.0

    A = A_w + A_f
    z_na = (A_w * z_w + A_f * z_f) / A

    I_w = tw * hw**3 / 12.0 + A_w * (z_w - z_na)**2
    I_f = bf * tf**3 / 12.0 + A_f * (z_f - z_na)**2

    return I_w + I_f, z_na

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
    I, _ = T_section_I_strong(hw, tw, bf, tf)
    J = rect_torsion_J(hw, tw) + rect_torsion_J(bf, tf)
    EI = E * I
    GJ = G * J
    A = T_section_area(hw, tw, bf, tf)
    return EI, GJ, A, I, J


# ------------------------------------------------------------
# 7. RAYLEIGH-RITZ GLOBAL INSTABILITY
# ------------------------------------------------------------
def make_mode_index(M_list, N_list):
    return [(m, n) for m in M_list for n in N_list]

def build_global_buckling_matrices(a, b, D1, D2, D12, beam_xs, EI_b, GJ_b,
                                   M_list, N_list):
    """
    Ritz basis:
        w = sum W_mn sin(m*pi*x/a) sin(n*pi*y/b)

    Global buckling:
        (K - Nx*Kg) q = 0
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

    # Discrete transverse beams
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

                # beam bending from w_yy
                Kb = EI_b * (b / 2.0) * (b1**4) * sin1 * sin2

                # beam torsion from w_xy
                Kt = GJ_b * (b / 2.0) * (a1 * a2 * b1 * b2) * cos1 * cos2

                K[i, j] += Kb + Kt

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

        Nx_cr = D * (alpha**2 + beta**2)**2 / alpha**2   # N/m
        sigma_cr = Nx_cr / t                             # Pa

        if sigma_cr < best_sigma:
            best_sigma = sigma_cr
            best_m = m

    return best_sigma, best_m


# ------------------------------------------------------------
# 9. TRIPPING OF LONGITUDINALS
# ------------------------------------------------------------
def tripping_sigma(a_p, b_p, t_plate, h_long, t_long, E, nu, sigma_y, m_max=20, n=1):
    """
    Tripping of flat-bar longitudinals with attached plate strip.
    Same logic as your previous B2/B3 tripping code.
    """
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
EI_tr, GJ_tr, A_tr, I_tr, J_tr = transverse_beam_properties(
    hw=hw_tr, tw=tw_tr, bf=bf_tr, tf=tf_tr_red, E=E, G=G
)

K, Kg, modes = build_global_buckling_matrices(
    a=a, b=b,
    D1=D1, D2=D2, D12=D12,
    beam_xs=beam_xs,
    EI_b=EI_tr, GJ_b=GJ_tr,
    M_list=M_list, N_list=N_list
)

Nx_instability, q_instability = lowest_generalized_eigenpair(K, Kg)   # N/m
F_instability = Nx_instability * b                                    # N
sigma_panel_eq = Nx_instability / t_plate                             # Pa


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
collapse_force = min(F_instability, F_trip_corr)
collapse_stress = collapse_force / (b * t_plate)

if collapse_force == F_instability:
    governing_mode = "Global instability (Rayleigh-Ritz)"
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
    a_field=s_tr_red,   # avstand mellom transverse stiffeners = 5.6 m
    b_field=b,          # panel width = 9.95 m
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




