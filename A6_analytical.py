import math

# -----------------------------
# Input values
# -----------------------------
a = 2800.0   # mm
b = 670.0    # mm
t = 6.5      # mm
E = 210000.0 # MPa = N/mm^2
nu = 0.30

# -----------------------------
# Plate rigidity
# D = E*t^3 / (12*(1-nu^2))
# -----------------------------
D = E * t**3 / (12.0 * (1.0 - nu**2))   # Nmm

# -----------------------------
# Buckling coefficient
# k = (m*b/a + a/(m*b))^2
# Check several integer values of m
# -----------------------------
best_sigma = float("inf")
best_q = None
best_m = None
best_k = None

for m in range(1, 21):
    k = (m * b / a + a / (m * b))**2
    sigma_cr = k * D * math.pi**2 / (b**2 * t)   # MPa = N/mm^2
    q_cr = sigma_cr * t                           # N/mm = kN/m

    if sigma_cr < best_sigma:
        best_sigma = sigma_cr
        best_q = q_cr
        best_m = m
        best_k = k

# -----------------------------
# Output
# -----------------------------
print(f"D = {D:.6e} Nmm")
print(f"Best m = {best_m}")
print(f"k = {best_k:.4f}")
print(f"sigma_cr = {best_sigma:.1f} MPa")
print(f"q_cr = {best_q:.1f} kN/m")