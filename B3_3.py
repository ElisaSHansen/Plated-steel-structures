import math

# ------------------------------------------------
# INPUT
# ------------------------------------------------

E = 210000.0
nu = 0.3
sigma_y = 250.0   # MPa

a_pp = 55800
b_pp = 9950

a_p = 2800
c_l = 9950

# plate thickness
h = 6.5

# longitudinal stiffener (fra B2)
stiff_height = 160
stiff_thickness = 12
d1 = 670

# transverse stiffener
d = 695
t_flange_tr = 10
b_flange_tr = 100.0
t_web_tr = 10.0

# ------------------------------------------------
# PLATE RIGIDITY
# ------------------------------------------------

D = E * h**3 / (12 * (1 - nu**2))

# ------------------------------------------------
# LONGITUDINAL STIFFENER PROPERTIES
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
# FUNCTIONS
# ------------------------------------------------

def S(m):
    return 0 if m % 4 == 0 else 1

def C(m):
    return 3 if m % 4 == 0 else 1

def johnson_ostenfeld(sigma_c, sigma_y):
    if sigma_c > sigma_y / 2.0:
        return sigma_y * (1.0 - sigma_y / (4.0 * sigma_c))
    return sigma_c

# ------------------------------------------------
# EFFECTIVE WIDTH
# ------------------------------------------------

b_e = 1.1 * a_p / (1.0 + 2.0 * (a_p / c_l) ** 2)

# ------------------------------------------------
# TRANSVERSE STIFFENER PROPERTIES
# ------------------------------------------------

h_web_tr = d - 0.5 * t_flange_tr

A_g = b_e * h + h_web_tr * t_web_tr + b_flange_tr * t_flange_tr

e_g = (
    -0.5 * h**2 * b_e
    + 0.5 * h_web_tr**2 * t_web_tr
    + b_flange_tr * t_flange_tr * d
) / A_g

I_bar = (
    h**3 * b_e / 3.0
    + h_web_tr**3 * t_web_tr / 3.0
    + b_flange_tr * t_flange_tr * d**2
    - e_g**2 * A_g
)

I_f = b_flange_tr**3 * t_flange_tr / 12.0

K = (
    h_web_tr * t_web_tr**3
    + b_flange_tr * t_flange_tr**3
) / 3.0

G = E / (2.0 * (1.0 + nu))

# ------------------------------------------------
# sigma_c3
# ------------------------------------------------

def sigma_c3(m):
    term1 = (
        (math.pi / b_pp) ** 4
        * (D + (4.0 * E * I_bar / a_pp) * S(m))
        * (m * math.pi / a_pp) ** (-2)
    )

    term2 = D1 * (m * math.pi / a_pp) ** 2

    term3 = (
        (2.0 / a_pp)
        * (math.pi / b_pp) ** 2
        * (
            D * a_pp
            + (G * K + E * I_f * d**2 * (math.pi / b_pp) ** 2) * C(m)
        )
    )

    return (term1 + term2 + term3) / h_bar

# ------------------------------------------------
# CONTINUOUS m FROM EQ. (3.27)
# ------------------------------------------------

m_cont = (a_pp / b_pp) * (
    (D / D1) + (4.0 * E * I_bar) / (D1 * a_pp)
) ** 0.25

# ------------------------------------------------
# TEST NEARBY INTEGER VALUES
# ------------------------------------------------

m_floor = math.floor(m_cont)
m_ceil = math.ceil(m_cont)

candidate_range = range(max(1, m_floor - 3), m_ceil + 4)
m_candidates = [m for m in candidate_range if m % 4 != 0]

if not m_candidates:
    m_candidates = [1, 2, 3, 5, 6, 7]

results = [(m, sigma_c3(m)) for m in m_candidates]
m_min, sigma_min = min(results, key=lambda x: x[1])

# ------------------------------------------------
# JOHNSON-OSTENFELD CORRECTION OF LOWEST VALUE
# ------------------------------------------------

sigma_min_corr = johnson_ostenfeld(sigma_min, sigma_y)

# ------------------------------------------------
# OUTPUT
# ------------------------------------------------

print(f"D = {D:.6e} Nmm")
print(f"D1 = {D1:.6e} Nmm")
print(f"h_bar = {h_bar:.6f} mm")
print(f"m_cont = {m_cont:.6f}")
print(f"tested m = {m_candidates}")
print(f"m = {m_min}")
print(f"sigma_c3 = {sigma_min:.6f} MPa")
print(f"sigma_c3,corr = {sigma_min_corr:.6f} MPa")