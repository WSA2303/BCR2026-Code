import math

# ============================================================
# Parâmetros de estudo
# ============================================================

C_values = [0.3, 0.9]

# Número desejado de volumes na região cisalhada
N_cis_star_values = [40, 50, 180, 200]

# ============================================================
# Parâmetros físicos fixos
# ============================================================

rho = 1560       # kg/m³
g = 9.81         # m/s²
theta = 15       # graus

sin_theta = math.sin(math.radians(theta))
cos_theta = math.cos(math.radians(theta))
cot_theta = cos_theta / sin_theta

K_n = 0.91       # Pa·s^n
h_0 = 3.35e-2    # m
n = 1            # adimensional

# ============================================================
# Função para calcular Nz(C)
# ============================================================

def calcular_Nz(C, N_cis_star):
    """
    Calcula o número total de volumes verticais Nz
    necessário para obter N_cis_star volumes na região cisalhada.
    """
    return math.ceil(N_cis_star / (1 - C))


# ============================================================
# Cálculos
# ============================================================

A = ((rho * g * sin_theta * h_0) ** (n - 1)) / K_n

for C in C_values:
    print("=" * 60)
    print(f"C = {C}")

    # Cálculo de tau_0 e z_0
    tau_0 = (rho * g * sin_theta * h_0) * C
    z_0 = h_0 - tau_0 / (rho * g * sin_theta)
    z_0_C = h_0 * (1 - C)

    # Cálculo de u_0
    fator1 = n / (n + 1)
    fator2 = ((rho * g * sin_theta) / K_n) ** (1 / n)
    fator3 = z_0 ** ((n + 1) / n)

    u_0 = fator1 * fator2 * fator3

    print(f"u_0   = {u_0:.6f} m/s")
    print(f"tau_0 = {(tau_0 / rho):.6f} m²/s²")
    print(f"z_0   = {z_0:.6f} m")
    print()

    print("Resultados para diferentes volumes cisalhados:")

    for N_cis_star in N_cis_star_values:
        Nz = calcular_Nz(C, N_cis_star)

        eta_0 = (C + (1 / Nz)) * ((Nz / A) ** (1 / n))

        print(
            f"N_cis* = {N_cis_star:>3d} "
            f"-> Nz = {Nz:>4d} "
            f"-> eta_0/rho = {(eta_0 / rho):.6f} m²/s"
        )

print("=" * 60)