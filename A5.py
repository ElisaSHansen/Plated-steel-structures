import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# INPUT (A5 – lokal jevnt fordelt last i senter)
# -------------------------------------------------
P  = 20000          # N
a  = 28 / 10        # 2.8 m
b  = 67 / 100       # 0.67 m
u  = 2 / 10         # 0.2 m
v  = 25 / 100       # 0.25 m

t  = 65 / 10000     # 0.0065 m
E  = 210 * 10**9
nu = 3 / 10

xi1 = a / 2
xi2 = b / 2

M_max = 20
N_max = 20
tol = 1e-4

# Grid for å finne w_max over platen
nx = 81
ny = 81

# -------------------------------------------------
# Plate stivhet
# -------------------------------------------------
D = E * t**3 / (12 * (1 - nu**2))

# -------------------------------------------------
# Pmn for lokal last (kun oddetall bidrar ved senterlast)
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

# -------------------------------------------------
# Wmn
# -------------------------------------------------
def Wmn(m, n):
    Pmn = Pmn_patch(m, n)
    denom = D * np.pi**4 * ((m / a)**2 + (n / b)**2)**2
    return Pmn / denom

# -------------------------------------------------
# Beregn w_max(M,N)
# -------------------------------------------------
def compute_w_max(Mmax, Nmax):
    x1 = np.linspace(0, a, nx)
    x2 = np.linspace(0, b, ny)
    w = np.zeros((ny, nx), dtype=float)

    # Precompute sinus for fart (kun oddetall)
    sin_mx1 = {m: np.sin(m * np.pi * x1 / a) for m in range(1, Mmax + 1, 2)}
    sin_nx2 = {n: np.sin(n * np.pi * x2 / b) for n in range(1, Nmax + 1, 2)}

    for m in range(1, Mmax + 1, 2):
        for n in range(1, Nmax + 1, 2):
            w += Wmn(m, n) * np.outer(sin_nx2[n], sin_mx1[m])

    return float(np.max(np.abs(w)))

# -------------------------------------------------
# Konvergensflate Wsurf(N,M) = w_max(M,N)
# -------------------------------------------------
Wsurf = np.zeros((N_max, M_max), dtype=float)

for M in range(1, M_max + 1):
    for N in range(1, N_max + 1):
        Wsurf[N - 1, M - 1] = compute_w_max(M, N)

w_ref = float(Wsurf[N_max - 1, M_max - 1])

# Relativ feil (robust hvis w_ref skulle blitt 0)
den = abs(w_ref) if abs(w_ref) > 0 else 1.0
Err = np.abs(Wsurf - w_ref) / den

# -------------------------------------------------
# Finn konvergenspunkt (minste budsjett k = max(M,N))
# -------------------------------------------------
Mconv = None
Nconv = None

for k in range(1, max(M_max, N_max) + 1):
    candidates = []
    for M in range(1, min(M_max, k) + 1):
        N = k
        if N <= N_max and Err[N - 1, M - 1] <= tol:
            candidates.append((M, N))
    for N in range(1, min(N_max, k) + 1):
        M = k
        if M <= M_max and Err[N - 1, M - 1] <= tol:
            candidates.append((M, N))

    if candidates:
        Mconv, Nconv = min(candidates, key=lambda t: (t[0] + t[1], t[0], t[1]))
        break

print(f"Referanse: w_ref = w_max(M={M_max}, N={N_max}) = {w_ref:.6e} m")
if Mconv is None:
    print(f"Ingen konvergens innen (M,N) <= ({M_max},{N_max}) for tol={tol}.")
else:
    w_conv = float(Wsurf[Nconv - 1, Mconv - 1])
    rel_err = float(Err[Nconv - 1, Mconv - 1])
    print("Konvergens oppnådd ved:")
    print("Mconv =", Mconv)
    print("Nconv =", Nconv)
    print(f"w_max(Mconv,Nconv) = {w_conv:.6e} m")
    print(f"Relativ feil = {rel_err:.3e} (tol = {tol})")

# -------------------------------------------------
# 3D plot (med farger + flyttet colorbar + rød prikk)
# -------------------------------------------------
M_vals = np.arange(1, M_max + 1)
N_vals = np.arange(1, N_max + 1)
M_mesh, N_mesh = np.meshgrid(M_vals, N_vals)

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    M_mesh, N_mesh, Wsurf,
    cmap="plasma",
    edgecolor="k",
    linewidth=0.2,
    antialiased=True,
    alpha=0.95
)

# Colorbar med avstand slik at den ikke kræsjer med z-aksen
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.7,
    aspect=18,
    pad=0.14
)
cbar.set_label("w_max [m]")

ax.set_xlabel("Mmax")
ax.set_ylabel("Nmax", labelpad=20)
ax.set_zlabel("")  # fjern z-etikett
ax.set_title("Konvergens for lokal last (kun oddetall bidrar)")

# Utvid z-akse litt oppover
zmin = float(np.min(Wsurf))
zmax = float(np.max(Wsurf))
ax.set_zlim(zmin, zmax * 1.15)

# Rød prikk på konvergenspunktet (løftet litt over flaten)
if Mconv is not None:
    Zc = float(Wsurf[Nconv - 1, Mconv - 1])
    ax.scatter(
        Mconv, Nconv, Zc * 1.02,
        color="red",
        s=300,
        edgecolors="black",
        depthshade=False
    )

# Fast visningsvinkel (du kan endre)
ax.view_init(elev=30, azim=220)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# (Valgfritt) 2D-kart for relativ feil
# -------------------------------------------------
plt.figure(figsize=(7, 5))
cs = plt.contourf(M_mesh, N_mesh, Err, levels=30)
plt.colorbar(cs, label="Relativ feil |w_max - w_ref| / |w_ref|")
plt.xlabel("Mmax")
plt.ylabel("Nmax")
plt.title(f"Relativ-feil-kart (tol = {tol:g})")
if Mconv is not None:
    plt.scatter([Mconv], [Nconv], marker="o")
plt.tight_layout()
plt.show()
