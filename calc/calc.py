import math


# Parâmetros fixos
C = 0.9 
Nz = 25
rho = 1560       # kg/m³
g = 9.81         # m/s²
theta = 15       # graus
sin_theta = math.sin(math.radians(theta))
cos_theta = math.cos(math.radians(theta))
cot_theta = (cos_theta/sin_theta)
K_n = 0.91       # Pa·s^n
h_0 = 3.35e-2    # m
n = 1            # adimensional

# Cálculo de tau_0 e z_0
tau_0 = (rho * g * sin_theta * h_0) * C
z_0 = h_0 - (tau_0) / (rho * g * sin_theta)
z_0_C = h_0 * (1-C)

# Cálculo de u_0 para n = 1
fator1 = n / (n + 1)
fator2 = ((rho * g * sin_theta) / K_n) ** (1 / n)
fator3 = z_0 ** ((n + 1) / n)
#fator4 = 1 - (n / (2 * n + 1)) * (z_0 / h_0)

u_0 = fator1 * fator2 * fator3

# Cálculo \eta_0 

A = ((rho*g*sin_theta*h_0)**(n-1))/(K_n)
eta_0 = (C+(1/Nz)) * ((Nz/A)**(1/n))

print(f"{A:.3f}")

# Resultados
# print(f"u_0 = {(u_0):.5f} m/s")
print(f"tau_0/rho = {(tau_0/rho):.5f} m^2/s^2")
print(f"tau_0 = {(tau_0):.2f} Pa")
# print(f"z_0 = {(z_0):.5f} m")
# print(f"z_0(C) = {(z_0_C):.5f} m")
print(f"eta_0 = {(eta_0/rho):.5f}")

# n = 1            # adimensional
# Nz= 25 ## ATENCAO!!!!

# A=((rho*g*sin_theta*h_0)**(1-n))/(K_n)

# eta_0 = ((C+(1/Nz))*Nz/A)/rho
# print(f"eta_0 = {eta_0}")
