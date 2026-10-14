import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# Paramètres de simulation
L = 1000.0  # Domaine étendu pour réduire les effets de bord
N = 2**12
dx = L / N
x = np.linspace(-L / 2, L / 2, N, endpoint=False)

Tmax = 200.0
dt = 0.03
Nt = int(Tmax / dt)
g = -1.0  # Interaction répulsive si g > 0, attractive si g < 0

# Opérateur de dispersion
k = 2 * np.pi * fftfreq(N, d=dx)
D = np.exp(-0.5j * k**2 * dt)


# === PROFIL GAUSSIEN===
def initial_density(x, A=1.0, sigma=40.0):
    return A * np.exp(-(((x**2) / (2 * sigma**2)) ** 1))


def initial_density_shaped(x, A=1.0, sigma_outer=40.0, B=0.5, sigma_inner=10.0):
    """Gaussienne creusée au centre (ou pas)."""
    outer = A * np.exp(-(x**2) / (2 * sigma_outer**2))
    inner = B * np.exp(-(x**2) / (2 * sigma_inner**2))
    density = outer - inner
    return density


def initial_velocity(x, x0=0, B=1.0, sigma=2.0):
    return B * np.exp(-(((x - x0) ** 2) / (2 * sigma**2)))


# Construction de l'état initial
# rho0 = np.ones((x.shape))
sigma0 = 10.0
x00 = 0

# # with DSW
d_s = 0.6
B0 = 1.8

# # no DSW
# d_s = 0.1
# B0 = 0.9

rho0 = initial_density_shaped(x, A=1.0, sigma_outer=80.0, B=d_s, sigma_inner=sigma0)
v0 = initial_velocity(x, x0=x00, B=B0, sigma=sigma0)
phi0 = np.cumsum(v0) * dx
psi0 = np.sqrt(rho0) * np.exp(1j * phi0)


# Affichage des conditions initiales
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(x, rho0, label="Densité initiale")
ax[0].set_ylabel("Densité")
ax[0].set_ylim(0, 1.1 * np.max(rho0))
ax[1].plot(x, v0, label="Vitesse initiale")
ax[1].set_ylabel("Vitesse")
ax[1].set_ylim(-1.1 * np.max(np.abs(v0)), 1.1 * np.max(np.abs(v0)))
ax[1].set_xlabel("x")
ax[0].axvline(0, color="gray", lw=1, ls="--")
ax[1].axvline(0, color="gray", lw=1, ls="--")
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show()

# Initialisation
psi = psi0.copy()
out_density = []
out_velocity = []
out_lambda_minus = []
out_lambda_plus = []


# Stockage de l'état initial
rho = np.abs(psi) ** 2
phi = np.angle(psi)
v = np.gradient(phi, dx)
out_density.append(rho.copy())
out_velocity.append(v.copy())
out_lambda_minus.append(np.zeros(rho.shape))
out_lambda_plus.append(np.zeros(rho.shape))


for step in tqdm(range(Nt)):
    t = step * dt

    psi = D * fft(psi)
    psi = ifft(psi)
    psi *= np.exp(1j * g * np.abs(psi) ** 2 * dt)

    if step % 50 == 0:
        rho = np.abs(psi) ** 2
        phi = np.unwrap(np.angle(psi))
        v = np.gradient(phi, dx)
        lambda_minus = np.sqrt(rho) * v / 2 - np.sqrt(rho)
        lambda_plus = np.sqrt(rho) * v / 2 + np.sqrt(rho)

        out_density.append(rho.copy())
        out_velocity.append(v.copy())
        out_lambda_minus.append(lambda_minus.copy())
        out_lambda_plus.append(lambda_plus.copy())


# Animation
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(4, 8), sharex=True)
(line1,) = ax1.plot([], [], lw=2, label="Densité")
(line2,) = ax2.plot([], [], lw=2, label="Vitesse")
(line3,) = ax3.plot([], [], lw=2, label="λ⁺ = √ρv/2 + √ρ")
(line4,) = ax4.plot([], [], lw=2, label="λ⁻ = √ρv/2 - √ρ")

T = 0.02
x_rescaled = x / np.sqrt(T)
x_rescaled /= 10
x_rescaled /= 2

ax1.set_ylabel("Densité")
ax2.set_ylabel("Vitesse")
ax2.set_xlabel("x")
ax1.set_xlim(x_rescaled[0], x_rescaled[-1])
ax2.set_xlim(x_rescaled[0], x_rescaled[-1])
ax2.axhline(0, color="gray", lw=1, ls="--")
ax1.set_ylim(0, 1.5 * np.max(rho0))
ax2.set_ylim(-4 * np.max(np.abs(v0)), 4 * np.max(np.abs(v0)))
ax3.set_ylabel("λ⁺")
ax3.set_xlabel("x")
ax3.axhline(0, color="gray", lw=1, ls="--")
ax3.set_xlim(x_rescaled[0], x_rescaled[-1])
ax4.set_ylabel("λ⁻")
ax4.set_xlabel("x")
ax4.axhline(0, color="gray", lw=1, ls="--")
ax4.set_xlim(x_rescaled[0], x_rescaled[-1])
lambda_max = np.max(
    np.abs([lmb for l in out_lambda_plus for lmb in l])
)  # estimation grossière
ax3.set_ylim(-1.1 * lambda_max, 1.1 * lambda_max)
lambda_max = np.max(
    np.abs([lmb for l in out_lambda_minus for lmb in l])
)  # estimation grossière
ax4.set_ylim(-1.1 * lambda_max, 1.1 * lambda_max)


def update(frame):
    line1.set_data(x, out_density[frame])
    line2.set_data(x, out_velocity[frame])
    line3.set_data(x, out_lambda_plus[frame])
    line4.set_data(x, out_lambda_minus[frame])
    return line1, line2, line3, line4


ani = FuncAnimation(fig, update, frames=len(out_density), blit=True, interval=50)
plt.tight_layout()
plt.show()
