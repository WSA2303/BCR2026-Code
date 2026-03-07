import math

# Parâmetros fixos
C_values = [0.3, 0.9, 0.6]
Nz_values = [25, 50, 100, 200]

rho = 1560       # kg/m³
g = 9.81         # m/s²
theta = 15       # graus
sin_theta = math.sin(math.radians(theta))
cos_theta = math.cos(math.radians(theta))
cot_theta = cos_theta / sin_theta  # (mantido, mesmo não sendo usado abaixo)

K_n = 0.91       # Pa·s^n
h_0 = 0.0335    # m
n = 1            # adimensional

# Termo auxiliar A (com n=1, vira 1/K_n, mas deixei geral como no teu)
A = ((rho * g * sin_theta * h_0) ** (n - 1)) / K_n

for C in C_values:
    # Cálculo de tau_0 e z_0 (dependem de C)
    tau_0 = (rho * g * sin_theta * h_0) * C
    z_0 = h_0 - (tau_0) / (rho * g * sin_theta)  # equivale a h_0*(1-C)
    z_0_C = h_0 * (1 - C)

    # Cálculo de u_0
    fator1 = n / (n + 1)
    fator2 = ((rho * g * sin_theta) / K_n) ** (1 / n)
    fator3 = z_0 ** ((n + 1) / n)
    u_0 = fator1 * fator2 * fator3

    print("=" * 50)
    print(f"C = {C}")
    print(f"u_0  = {u_0:.4f} m/s")
    print(f"tau_0= {(tau_0):.4f} m^2/s^2")
    print("eta_0/rho (m^2/s):")

    for Nz in Nz_values:
        eta_0 = (C + (1 / Nz)) * ((Nz / A) ** (1 / n))
        print(f"  Nz = {Nz:>3d} -> eta_0 = {(eta_0):.4f}")
        #print(f"  Nz = {Nz:>3d} -> eta_0/rho = {(eta_0 / rho):.5f}")

print("=" * 50)

print(f"eta_0(Kn) = {(1000*K_n):.5f}")
