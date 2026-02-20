import numpy as np

# -----------------------------
# Input (konkrete tallverdier)
# -----------------------------
a = 2.8       # m   (2800 mm)
b = 0.67      # m   (670 mm)
t = 0.0065    # m   (6.5 mm)
E = 2.1e11    # Pa
nu = 0.30     # -
p = 10e3      # N/m^2 (10 kN/m^2)

D = E * t**3 / (12 * (1 - nu**2))  # platebøyestivhet

# -----------------------------
# Shape functions
# Simply supported i x1, clamped i x2
# phi_mn(x,y) = sin(m*pi*x/a) * (1 - cos(2*n*pi*y/b))
# -----------------------------
m_list = [1, 1, 2, 2]
n_list = [1, 2, 1, 2]
modes = list(zip(m_list, n_list))  # [(1,1), (1,2), (2,1), (2,2)]

def phi(m, n, x, y):
    return np.sin(m*np.pi*x/a) * (1 - np.cos(2*n*np.pi*y/b))

def phi_xx_plus_yy(m, n, x, y):
    # w_xx + w_yy for phi (uten koeffisienten W_mn)
    # d2/dx2 sin(m*pi*x/a) = -(m*pi/a)^2 sin(...)
    # d2/dy2 (1 - cos(2n*pi*y/b)) = (2n*pi/b)^2 cos(2n*pi*y/b)
    sx = np.sin(m*np.pi*x/a)
    cy = np.cos(2*n*np.pi*y/b)
    return (-(m*np.pi/a)**2) * sx * (1 - cy) + sx * ((2*n*np.pi/b)**2) * cy

# -----------------------------
# Numerisk integrasjon (2D) med trapesregel
# (tilstrekkelig fin mesh for stabile tall)
# -----------------------------
Nx = 801  # øk ved behov
Ny = 401  # øk ved behov

x = np.linspace(0.0, a, Nx)
y = np.linspace(0.0, b, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="xy")  # X: (Ny,Nx), Y: (Ny,Nx)

def integrate_2d(F):
    # Trapesregel 2D
    return np.trapz(np.trapz(F, x, axis=1), y, axis=0)

# -----------------------------
# Bygg K og f fra Π = U + V
# Ub = 1/2 ∬ D (w_xx + w_yy)^2 dA
# V  = - ∬ p w dA
#
# w = sum_i W_i * phi_i
#
# => Ub = 1/2 * sum_i sum_j W_i W_j * D ∬ G_i G_j dA, der G_i=(phi_i_xx+phi_i_yy)
# => V  = - sum_i W_i * ∬ p phi_i dA
#
# Minimere: dΠ/dW = 0 => K W = f
# K_ij = D ∬ G_i G_j dA
# f_i  = ∬ p phi_i dA
# -----------------------------
nm = len(modes)
K = np.zeros((nm, nm), dtype=float)
f = np.zeros(nm, dtype=float)

G = []
PHI = []

for (m, n) in modes:
    PHI.append(phi(m, n, X, Y))
    G.append(phi_xx_plus_yy(m, n, X, Y))

for i in range(nm):
    f[i] = integrate_2d(p * PHI[i])
    for j in range(nm):
        K[i, j] = D * integrate_2d(G[i] * G[j])

# Løs for W-vektor
W = np.linalg.solve(K, f)

# -----------------------------
# Utskrift: "m=... n=... gir w=..."
# Her skriver vi bidraget til nedbøyning i senter (x=a/2, y=b/2),
# samt total nedbøyning i senter og maks på et rutenett.
# -----------------------------
xc = a / 2
yc = b / 2

# Senterverdi for hver mode og total
w_center_parts = []
w_center_total = 0.0

for idx, (m, n) in enumerate(modes):
    phi_c = phi(m, n, xc, yc)
    w_part = W[idx] * phi_c
    w_center_parts.append(w_part)
    w_center_total += w_part
    print(f"m={m} n={n} gir w_senter={w_part:.6e} m   (W_{m},{n}={W[idx]:.6e})")

print(f"TOTAL: w_senter={w_center_total:.6e} m")

# Beregn w(x,y) på rutenett for å estimere maks nedbøyning
w_grid = np.zeros_like(X, dtype=float)
for idx, (m, n) in enumerate(modes):
    w_grid += W[idx] * PHI[idx]

w_max = np.max(w_grid)
w_min = np.min(w_grid)
iy, ix = np.unravel_index(np.argmax(w_grid), w_grid.shape)
x_max = x[ix]
y_max = y[iy]

print(f"w_max={w_max:.6e} m ved x={x_max:.6f} m, y={y_max:.6f} m")
print(f"w_min={w_min:.6e} m")

# Hvis du vil ha mm også:
print(f"TOTAL: w_senter={w_center_total*1e3:.6f} mm")
print(f"w_max={w_max*1e3:.6f} mm")
