from __future__ import annotations

import math
import numpy as np


def u_vertical(ty, Kn, n, rho, th, h, dz, adm=False):
    vel = []      # m/s
    gamm = []     # 1/s
    eta_arr = []  # Pa.s (no seu código você divide depois por rho)
    g = 9.81      # m/s²
    z0 = h - ty / (rho * g * math.sin(th))  # m

    if adm is False:
        if dz < 1:
            for z in np.arange(0, h, dz):
                if z <= z0:
                    uz = (n / (n + 1)) * (
                        ((rho * g * math.sin(th) * (z0 ** (n + 1))) / Kn) ** (1 / n)
                    ) * (1 - (1 - z / z0) ** ((n + 1) / n))
                    gamma = ((rho * g * math.sin(th) * (z0 - z)) / Kn) ** (1 / n)
                    eta = (ty / gamma + Kn * (gamma ** (n - 1))) / rho
                else:
                    uz = vel[-1]
                    gamma = 0
                    eta = math.inf

                vel.append(uz)
                gamm.append(gamma)
                eta_arr.append(eta)

        elif dz == 1:
            z = h
            if z <= z0:
                uz = (n / (n + 1)) * (
                    ((rho * g * math.sin(th) * (z0 ** (n + 1))) / Kn) ** (1 / n)
                ) * (1 - (1 - z / z0) ** ((n + 1) / n))
                gamma = ((rho * g * math.sin(th) * (z0 - z)) / Kn) ** (1 / n)
                eta = (ty / gamma + Kn * (gamma ** (n - 1))) / rho
            else:
                uz = (n / (n + 1)) * (
                    ((rho * g * math.sin(th) * (z0 ** (n + 1))) / Kn) ** (1 / n)
                )
                gamma = 0
                eta = math.inf

            vel = uz
            gamm = gamma
            eta_arr = eta

    else:
        if dz < 1:
            G = (h * rho * g * math.sin(th)) / ty
            for z in np.arange(0, h / z0, dz / z0):
                if z < 1:
                    uz = 1 - (1 - z) ** ((n + 1) / n)
                elif z <= G / (G - 1):
                    uz = 1
                vel.append(uz)

    if dz < 1:
        if adm:
            u_avg = (G - 1) / G * (G / (G - 1) - n / (2 * n + 1))
        else:
            u_avg = (n / (n + 1)) * (((rho * g * math.sin(th)) / Kn) ** (1 / n)) * (
                (z0) ** ((n + 1) / n)
            ) * (1 - (n / (2 * n + 1)) * (z0 / h))
    else:
        u_avg = vel

    return vel, u_avg, z0, gamm, eta_arr
