import math

# ------------------------------------------------
# INPUT
# ------------------------------------------------

E = 210000
nu = 0.3
sigma_y = 250.0  # MPa

h = 6.5

stiff_height = 160
stiff_thickness = 12

d1 = 670
a = 2800
b = 9950

n = 1
m_max = 20


# ------------------------------------------------
# PLATE RIGIDITY
# ------------------------------------------------

D = E * h**3 / (12 * (1 - nu**2))


# ------------------------------------------------
# STIFFENER PROPERTIES
# ------------------------------------------------

A1 = stiff_height * stiff_thickness
I1 = stiff_thickness * stiff_height**3 / 12
e1 = stiff_height / 2


# ------------------------------------------------
# ORTHOTROPIC PARAMETERS
# ------------------------------------------------

a1 = A1 / (d1 * h)
b1 = (E * I1) / (D * d1)
c1 = e1 / h + 0.5


# ------------------------------------------------
# ORTHOTROPIC RIGIDITY
# ------------------------------------------------

D1 = D * (1 + b1 + (12 * (1 - nu**2) * a1 * c1**2) / (1 + a1))


# ------------------------------------------------
# EFFECTIVE THICKNESS
# ------------------------------------------------

h_bar = h + A1 / d1


# ------------------------------------------------
# PLASTICITY CORRECTION FUNCTION
# ------------------------------------------------

def plasticity_corrected_stress(sigma_c, sigma_y):
    if sigma_c > sigma_y / 2:
        return sigma_y * (1 - sigma_y / (4 * sigma_c))
    return sigma_c


# ------------------------------------------------
# BUCKLING CALCULATION
# ------------------------------------------------

if __name__ == "__main__":

    sigma_min = float("inf")
    sigma_corr_min = float("inf")
    best_m_elastic = None
    best_m_corr = None

    for m in range(1, m_max + 1):

        term1 = (D1 / D) * (m * math.pi / a)**4
        term2 = 2 * (m * math.pi / a)**2 * (n * math.pi / b)**2
        term3 = (n * math.pi / b)**4

        sigma = (D / h_bar) * (a / (m * math.pi))**2 * (term1 + term2 + term3)
        sigma_corr = plasticity_corrected_stress(sigma, sigma_y)

        print(f"m = {m:2d}, sigma_c = {sigma:8.3f} MPa, sigma_c,corr = {sigma_corr:8.3f} MPa")

        if sigma < sigma_min:
            sigma_min = sigma
            best_m_elastic = m

        if sigma_corr < sigma_corr_min:
            sigma_corr_min = sigma_corr
            best_m_corr = m

    print("\nLowest elastic buckling stress:")
    print(f"m = {best_m_elastic}, sigma_c = {sigma_min:.3f} MPa")

    print("\nLowest plasticity-corrected buckling stress:")
    print(f"m = {best_m_corr}, sigma_c,corr = {sigma_corr_min:.3f} MPa")