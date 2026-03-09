import math

# --------------------------------------------------
# INPUT
# --------------------------------------------------

# geometry (mm)
a_p = 2800.0          # distance between transverse stiffeners
b_p = 670.0           # spacing between longitudinals
t_plate = 6.5         # plate thickness

# longitudinal stiffener: flat bar
h_flat = 160.0        # stiffener height
t_flat = 12.0         # stiffener thickness

# material
E = 210000.0          # MPa = N/mm^2
nu = 0.30
sigma_y = 250.0       # MPa

# buckling mode
n = 1
m_max = 20

# --------------------------------------------------
# MATERIAL
# --------------------------------------------------

G = E / (2.0 * (1.0 + nu))

# --------------------------------------------------
# PLATE RIGIDITY
# --------------------------------------------------

D = E * t_plate**3 / (12.0 * (1.0 - nu**2))

# --------------------------------------------------
# SECTION PROPERTIES FOR LONGITUDINAL FLAT BAR
# --------------------------------------------------

# no flange for a flat bar
I_f = 0.0

# torsion constant
K = h_flat * t_flat**3 / 3.0

# rotational property used in denominator
# measured from plate surface / root line
I_p = h_flat**3 * t_flat / 3.0

# distance from plate top to stiffener centroid
d = h_flat / 2.0

# --------------------------------------------------
# PLASTICITY CORRECTION
# --------------------------------------------------

def plasticity_corrected_stress(sigma_c, sigma_y):
    if sigma_c > sigma_y / 2.0:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_c))
    return sigma_c

# --------------------------------------------------
# TRIPPING OF THE LONGITUDINALS
# WITH PLATE INCLUDED
# --------------------------------------------------

def sigma_c4(m, n=1):
    alpha = m * math.pi / a_p
    beta = n * math.pi / b_p

    numerator = (
        D * b_p * (alpha**2 + beta**2)**2
        + 2.0 * (G * K + E * I_f * d**2 * alpha**2) * alpha**2 * beta**2
    )

    denominator = alpha**2 * (b_p * t_plate + 2.0 * I_p * beta**2)

    return numerator / denominator

# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":
    sigma_min = float("inf")
    sigma_corr_min = float("inf")
    best_m_elastic = None
    best_m_corr = None

    for m in range(1, m_max + 1):
        sigma = sigma_c4(m, n)
        sigma_corr = plasticity_corrected_stress(sigma, sigma_y)

        print(f"m = {m:2d}, sigma_c = {sigma:8.3f} MPa, sigma_c,corr = {sigma_corr:8.3f} MPa")

        if sigma < sigma_min:
            sigma_min = sigma
            best_m_elastic = m

        if sigma_corr < sigma_corr_min:
            sigma_corr_min = sigma_corr
            best_m_corr = m

    print(f"\nLowest elastic buckling stress: m = {best_m_elastic}, sigma_c = {sigma_min:.3f} MPa")
    print(f"Lowest plasticity-corrected buckling stress: m = {best_m_corr}, sigma_c,corr = {sigma_corr_min:.3f} MPa")