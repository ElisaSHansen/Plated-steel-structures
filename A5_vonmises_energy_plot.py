import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Project A.5 – Rayleigh–Ritz plate with PATCH LOAD
# SSSS and CCSS boundary conditions
# Includes von Mises plots with X at maximum
# ============================================================

# -----------------------------
# 1) Inputs
# -----------------------------
a, b = 2.8, 0.67
t = 0.0065
E = 210e9
nu = 0.30
G = E/(2*(1+nu))
D = E*t**3/(12*(1-nu**2))
z = t/2

P_total = 20e3
patch_x, patch_y = 0.250, 0.200
A_patch = patch_x * patch_y
p0 = P_total / A_patch

x1c, x2c = a/2, b/2
xL1, xL2 = x1c - patch_x/2, x1c + patch_x/2
yL1, yL2 = x2c - patch_y/2, x2c + patch_y/2

M = N = 15

nx = ny = 201
xg = np.linspace(0, a, nx)
yg = np.linspace(0, b, ny)
Xg, Yg = np.meshgrid(xg, yg)

# -----------------------------
# 2) Basis functions
# -----------------------------
def SS(k, L, x):   return np.sin(k*np.pi*x/L)
def SS_d1(k, L, x): return (k*np.pi/L)*np.cos(k*np.pi*x/L)
def SS_d2(k, L, x): return -(k*np.pi/L)**2*np.sin(k*np.pi*x/L)

def CC(k, L, x): return np.sin(np.pi*x/L)*np.sin(k*np.pi*x/L)
def CC_d1(k, L, x):
    return (np.pi/L)*np.cos(np.pi*x/L)*np.sin(k*np.pi*x/L) + \
           (k*np.pi/L)*np.sin(np.pi*x/L)*np.cos(k*np.pi*x/L)
def CC_d2(k, L, x):
    return -(np.pi/L)**2*CC(k,L,x) - (k*np.pi/L)**2*CC(k,L,x)

# -----------------------------
# 3) Quadrature
# -----------------------------
def gauss(n, x1, x2):
    xi, wi = np.polynomial.legendre.leggauss(n)
    x = 0.5*(xi+1)*(x2-x1)+x1
    w = 0.5*(x2-x1)*wi
    return x, w

# -----------------------------
# 4) Rayleigh–Ritz solver
# -----------------------------
def ritz_patch(M, N, bcx="SS", bcy="SS"):
    fx, fx1, fx2 = (SS, SS_d1, SS_d2) if bcx=="SS" else (CC, CC_d1, CC_d2)
    fy, fy1, fy2 = (SS, SS_d1, SS_d2) if bcy=="SS" else (CC, CC_d1, CC_d2)

    nq = max(60, 3*M)
    xq, wx = gauss(nq, 0, a)
    yq, wy = gauss(nq, 0, b)

    X0 = np.array([fx(m,a,xq) for m in range(1,M+1)])
    X1 = np.array([fx1(m,a,xq) for m in range(1,M+1)])
    X2 = np.array([fx2(m,a,xq) for m in range(1,M+1)])
    Y0 = np.array([fy(n,b,yq) for n in range(1,N+1)])
    Y1 = np.array([fy1(n,b,yq) for n in range(1,N+1)])
    Y2 = np.array([fy2(n,b,yq) for n in range(1,N+1)])

    def I(A,B,w): return (A*w)@B.T

    Ix00, Ix11, Ix22 = I(X0,X0,wx), I(X1,X1,wx), I(X2,X2,wx)
    Iy00, Iy11, Iy22 = I(Y0,Y0,wy), I(Y1,Y1,wy), I(Y2,Y2,wy)
    Ix20 = I(X2,X0,wx); Iy20 = I(Y2,Y0,wy)

    xp, wxp = gauss(nq, xL1, xL2)
    yp, wyp = gauss(nq, yL1, yL2)
    Xp = np.array([fx(m,a,xp) for m in range(1,M+1)])
    Yp = np.array([fy(n,b,yp) for n in range(1,N+1)])
    IxP = (Xp*wxp).sum(axis=1)
    IyP = (Yp*wyp).sum(axis=1)

    nd = M*N
    A = np.zeros((nd,nd))
    B = np.zeros(nd)

    def idx(m,n): return m*N+n

    for m in range(M):
        for n in range(N):
            I = idx(m,n)
            B[I] = p0*IxP[m]*IyP[n]
            for mp in range(M):
                for np_ in range(N):
                    J = idx(mp,np_)
                    A[I,J] = D*(
                        Ix22[m,mp]*Iy00[n,np_] +
                        Ix00[m,mp]*Iy22[n,np_] +
                        nu*(Ix20[m,mp]*Iy20[n,np_] + Ix20[mp,m]*Iy20[np_,n]) +
                        2*(1-nu)*Ix11[m,mp]*Iy11[n,np_]
                    )
    W = np.linalg.solve(A,B).reshape((M,N))
    return W, (fx,fx1,fx2,fy,fy1,fy2)

# -----------------------------
# 5) Field evaluation
# -----------------------------
def eval_fields(W, basis):
    fx,fx1,fx2,fy,fy1,fy2 = basis
    M,N = W.shape

    X0 = np.array([fx(m,a,xg) for m in range(1,M+1)])
    X1 = np.array([fx1(m,a,xg) for m in range(1,M+1)])
    X2 = np.array([fx2(m,a,xg) for m in range(1,M+1)])
    Y0 = np.array([fy(n,b,yg) for n in range(1,N+1)])
    Y1 = np.array([fy1(n,b,yg) for n in range(1,N+1)])
    Y2 = np.array([fy2(n,b,yg) for n in range(1,N+1)])

    w  = np.einsum("mx,ny,mn->yx",X0,Y0,W)
    wxx= np.einsum("mx,ny,mn->yx",X2,Y0,W)
    wyy= np.einsum("mx,ny,mn->yx",X0,Y2,W)
    wxy= np.einsum("mx,ny,mn->yx",X1,Y1,W)

    s1 = -(E*z/(1-nu**2))*(wxx+nu*wyy)
    s2 = -(E*z/(1-nu**2))*(wyy+nu*wxx)
    t12= -(2*G*z)*wxy
    svm= np.sqrt(s1**2 - s1*s2 + s2**2 + 3*t12**2)

    return {"w":w,"svm":svm}

# -----------------------------
# 6) Solve
# -----------------------------
W_ssss, info_ssss = ritz_patch(M,N,"SS","SS")
W_ccss, info_ccss = ritz_patch(M,N,"CC","SS")

out_ssss = eval_fields(W_ssss, info_ssss)
out_ccss = eval_fields(W_ccss, info_ccss)

# -----------------------------
# 7) Plotting
# -----------------------------
def plot_defl(out,title):
    plt.figure(figsize=(8,5))
    plt.contourf(Xg,Yg,out["w"]*1e3,40)
    plt.colorbar(label="w [mm]")
    plt.title(title)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.show()

def plot_vm(out,title):
    svm = out["svm"]
    iy,ix = np.unravel_index(np.argmax(svm),svm.shape)
    plt.figure(figsize=(8,5))
    plt.contourf(Xg,Yg,svm*1e-6,40)
    plt.colorbar(label=r"$\sigma_{vM}$ [MPa]")
    plt.scatter(xg[ix],yg[iy],marker="x",s=120,color="black",label="Max")
    plt.legend()
    plt.title(title)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.tight_layout()
    plt.show()

plot_defl(out_ssss,"Deflection – Rayleigh–Ritz SSSS")
plot_defl(out_ccss,"Deflection – Rayleigh–Ritz CCSS")

plot_vm(out_ssss,"von Mises – Rayleigh–Ritz SSSS")
plot_vm(out_ccss,"von Mises – Rayleigh–Ritz CCSS")
