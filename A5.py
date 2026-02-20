import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# INPUT (A5 – lokal jevnt fordelt last i senter)
# -------------------------------------------------
P  = 20000
a  = 28 / 10
b  = 67 / 100
u  = 2 / 10
v  = 25 / 100

t  = 65 / 10000
E  = 210 * 10**9
nu = 3 / 10

xi1 = a / 2
xi2 = b / 2

M_max = 20 #endre m_max og N_max til høyere grense ved spenningsberegning 
N_max = 20 
tol = 1e-4 #endre til 1e-7 ved spenningsberegning 

nx = 81
ny = 81

# -------------------------------------------------
# Plate stivhet
# -------------------------------------------------
D = E * t**3 / (12 * (1 - nu**2))

# -------------------------------------------------
# Pmn (kun oddetall bidrar ved senterlast)
# -------------------------------------------------
def Pmn_patch(m, n):
    if (m % 2 == 0) or (n % 2 == 0):
        return 0.0
    p0 = P / (u * v)
    return (
        16 * p0 / (np.pi**2 * m * n)
        * np.sin(m * np.pi * xi1 / a) * np.sin(m * np.pi * u / (2 * a))
        * np.sin(n * np.pi * xi2 / b) * np.sin(n * np.pi * v / (2 * b))
    )

def Wmn(m, n):
    denom = D * np.pi**4 * ((m / a)**2 + (n / b)**2)**2
    return Pmn_patch(m, n) / denom

# -------------------------------------------------
# Beregn w_max(M,N)
# -------------------------------------------------
def compute_w_max(Mmax, Nmax):
    x1 = np.linspace(0, a, nx)
    x2 = np.linspace(0, b, ny)
    w = np.zeros((ny, nx))

    sin_m = {m: np.sin(m * np.pi * x1 / a) for m in range(1, Mmax+1, 2)}
    sin_n = {n: np.sin(n * np.pi * x2 / b) for n in range(1, Nmax+1, 2)}

    for m in range(1, Mmax+1, 2):
        for n in range(1, Nmax+1, 2):
            w += Wmn(m, n) * np.outer(sin_n[n], sin_m[m])

    return float(np.max(np.abs(w)))

# -------------------------------------------------
# Konvergensflate
# -------------------------------------------------
Wsurf = np.zeros((N_max, M_max))

for M in range(1, M_max+1):
    for N in range(1, N_max+1):
        Wsurf[N-1, M-1] = compute_w_max(M, N)

w_ref = float(Wsurf[N_max-1, M_max-1])
Err = np.abs(Wsurf - w_ref) / (abs(w_ref) if abs(w_ref) > 0 else 1)

Mconv = None
Nconv = None

for k in range(1, max(M_max, N_max)+1):
    candidates = []
    for M in range(1, min(M_max, k)+1):
        N = k
        if N <= N_max and Err[N-1, M-1] <= tol:
            candidates.append((M, N))
    for N in range(1, min(N_max, k)+1):
        M = k
        if M <= M_max and Err[N-1, M-1] <= tol:
            candidates.append((M, N))
    if candidates:
        Mconv, Nconv = min(candidates, key=lambda t: (t[0]+t[1], t[0], t[1]))
        break

print(f"w_ref = {w_ref:.6e}")
print("Konvergens ved:", Mconv, Nconv)

# -------------------------------------------------
# 3D Plot
# -------------------------------------------------
M_vals = np.arange(1, M_max+1)
N_vals = np.arange(1, N_max+1)
M_mesh, N_mesh = np.meshgrid(M_vals, N_vals)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    M_mesh, N_mesh, Wsurf,
    cmap="plasma",
    edgecolor="k",
    linewidth=0.2,
    antialiased=True
)

# Colorbar med etikett på venstre side
cbar = fig.colorbar(surf, ax=ax, shrink=0.75, aspect=18, pad=0.15)
cbar.set_label("w_max [m]", rotation=90, labelpad=15)
cbar.ax.yaxis.set_label_position("left")
cbar.ax.yaxis.tick_left()

ax.set_xlabel("Mmax")
ax.set_ylabel("Nmax", labelpad=20)
ax.set_zlabel("")
ax.set_title("Konvergens for lokal last (kun oddetall bidrar)")

# Utvid z-akse
zmin = float(np.min(Wsurf))
zmax = float(np.max(Wsurf))
ax.set_zlim(zmin, zmax * 1.15)

# Mindre rød prikk
if Mconv is not None:
    Zc = float(Wsurf[Nconv-1, Mconv-1])
    eps = 1e-4 * (zmax - zmin)

    ax.scatter(
        Mconv,
        Nconv,
        Zc + eps,
        color="red",
        s=120,          # mindre prikk
        edgecolors="black",
        linewidths=1,
        depthshade=False
    )

ax.view_init(elev=30, azim=220)

plt.tight_layout()
plt.show()
