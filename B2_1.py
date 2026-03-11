import math

def unstiffened_plate_buckling(ap_mm, bp_mm, h_mm, E=2.1e5, nu=0.3, m_max=20):
    """
    Buckling of an unstiffened plate between longitudinals and girders.

    Parameters
    ----------
    ap_mm : float
        Plate length in mm
    bp_mm : float
        Plate width in mm
    h_mm : float
        Plate thickness in mm
    E : float
        Young's modulus in MPa = N/mm^2
    nu : float
        Poisson's ratio
    m_max : int
        Maximum integer m to check

    Returns
    -------
    dict with:
        aspect_ratio
        D
        m_best
        sigma_cr_MPa
    """

    # Plate bending stiffness
    D = E * h_mm**3 / (12 * (1 - nu**2))   # Nmm

    best_sigma = float("inf")
    best_m = None

    for m in range(1, m_max + 1):
        term = (m * bp_mm / ap_mm + ap_mm / (m * bp_mm))**2
        sigma_cr = term * D * math.pi**2 / (bp_mm**2 * h_mm)   # MPa

        if sigma_cr < best_sigma:
            best_sigma = sigma_cr
            best_m = m

    return {
        "aspect_ratio": ap_mm / bp_mm,
        "D": D,
        "m_best": best_m,
        "sigma_cr_MPa": best_sigma
    }


# DINE VERDIER
ap = 2800   # mm
bp = 670    # mm
h = 6.5      # mm  

result = unstiffened_plate_buckling(ap, bp, h)

print(f"a_p/b_p = {result['aspect_ratio']:.3f}")
print(f"D = {result['D']:.3e} Nmm")
print(f"Best m = {result['m_best']}")
print(f"Critical buckling stress = {result['sigma_cr_MPa']:.2f} MPa")