"""
Reduced-Order Numerical Model (ROM) for High-Temperature Alkali Metal Heat Pipes
Designed for Space Reactor Applications (Microgravity, High Heat Flux, Liquid Metal Working Fluid)
Author: Muhammad Naufal Tamimi (20/456315/TK/50445)
Departemen Teknik Nuklir dan Teknik Fisika, Universitas Gadjah Mada (UGM)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Any

# Universal Gas Constant (J / mol K)
R_UNIVERSAL = 8.314462618


@dataclass
class LiquidMetalProperties:
    """Thermophysical properties of liquid metal working fluid at temperature T (Kelvin)."""
    temperature: float        # K
    p_sat: float              # Saturation vapor pressure (Pa)
    rho_l: float              # Liquid density (kg/m3)
    rho_v: float              # Vapor density (kg/m3)
    h_fg: float               # Latent heat of vaporization (J/kg)
    sigma: float              # Surface tension (N/m)
    mu_l: float               # Liquid dynamic viscosity (Pa.s)
    mu_v: float               # Vapor dynamic viscosity (Pa.s)
    k_l: float                # Liquid thermal conductivity (W/m.K)
    k_v: float                # Vapor thermal conductivity (W/m.K)
    c_p_l: float              # Liquid specific heat (J/kg.K)
    gamma: float              # Ratio of specific heats (Cp/Cv)


def get_sodium_properties(T: float) -> LiquidMetalProperties:
    """
    Computes sodium thermophysical properties in range 600 K - 1400 K
    Based on Fink & Leibowitz (1995), Faghri (1995), and Dunn & Reay (1994).
    """
    T = max(500.0, min(1400.0, float(T)))
    M_Na = 0.0229898  # kg/mol
    R_g = R_UNIVERSAL / M_Na  # 361.66 J / kg.K

    # Saturation pressure (Pa) - Fink & Leibowitz (1995)
    # ln(P [MPa]) = 11.9463 - (12633.73 / T) - 0.5 * ln(T)
    # Multiply by 1e6 to convert MPa -> Pa
    log_p_mpa = 11.9463 - (12633.73 / T) - 0.5 * math.log(T)
    p_sat = math.exp(log_p_mpa) * 1.0e6  # Pa

    # Liquid density (kg/m3)
    rho_l = 1011.8 - 0.22054 * T - 1.9226e-5 * (T**2) + 5.637e-9 * (T**3)

    # Ideal gas vapor density (kg/m3)
    rho_v = p_sat / (R_g * T)

    # Latent heat of vaporization (J/kg)
    h_fg = (4.76e6 - 9.1e2 * T)

    # Surface tension (N/m)
    sigma = max(0.01, 0.2405 - 1.0e-4 * T)

    # Liquid dynamic viscosity (Pa.s)
    # ln(mu_l) = -6.4406 - 0.3958*ln(T) + 556.835/T
    mu_l = math.exp(-6.4406 - 0.3958 * math.log(T) + (556.835 / T))

    # Vapor dynamic viscosity (Pa.s) Chapman-Enskog approx for Na
    mu_v = 1.6e-7 * (T**0.7)

    # Liquid thermal conductivity (W/m.K)
    k_l = 93.0 - 0.0581 * T + 1.173e-5 * (T**2)

    # Vapor thermal conductivity (W/m.K)
    k_v = 0.025 + 2.5e-5 * T

    # Specific heat of liquid (J/kg.K)
    c_p_l = 1630.14 - 0.833 * T + 4.628e-4 * (T**2)

    # Ratio of specific heats
    gamma = 1.33

    return LiquidMetalProperties(
        temperature=T,
        p_sat=p_sat,
        rho_l=rho_l,
        rho_v=rho_v,
        h_fg=h_fg,
        sigma=sigma,
        mu_l=mu_l,
        mu_v=mu_v,
        k_l=k_l,
        k_v=k_v,
        c_p_l=c_p_l,
        gamma=gamma
    )


def get_nak78_properties(T: float) -> LiquidMetalProperties:
    """Thermophysical properties of eutectic NaK-78 (22 wt% Na, 78 wt% K)."""
    T = max(300.0, min(1200.0, float(T)))
    M_avg = 0.0355  # kg/mol
    R_g = R_UNIVERSAL / M_avg

    # Saturation pressure (Pa)
    log_p_mpa = 10.82 - (10500.0 / T) - 0.45 * math.log(T)
    p_sat = math.exp(log_p_mpa) * 1.0e6

    # Density (kg/m3)
    rho_l = 880.0 - 0.24 * (T - 298.15)
    rho_v = p_sat / (R_g * T)
    h_fg = 2.1e6 - 5.5e2 * T
    sigma = max(0.01, 0.115 - 6.5e-5 * T)
    mu_l = 0.00015 * math.exp(450.0 / T)
    mu_v = 1.2e-7 * (T**0.7)
    k_l = 25.0 + 0.008 * T
    k_v = 0.02
    c_p_l = 980.0
    gamma = 1.35

    return LiquidMetalProperties(
        temperature=T, p_sat=p_sat, rho_l=rho_l, rho_v=rho_v,
        h_fg=h_fg, sigma=sigma, mu_l=mu_l, mu_v=mu_v,
        k_l=k_l, k_v=k_v, c_p_l=c_p_l, gamma=gamma
    )


@dataclass
class HeatPipeGeometry:
    """Dimensions and porous wick specifications of the heat pipe."""
    # Lengths (m)
    L_e: float = 0.25      # Evaporator length
    L_a: float = 0.35      # Adiabatic length
    L_c: float = 0.40      # Condenser length

    # Diameters (m)
    d_outer: float = 0.0159      # Outer wall diameter (15.9 mm)
    t_wall: float = 0.000889     # Wall thickness (0.889 mm)
    t_wick: float = 0.000600     # Wick / annulus total thickness (0.6 mm)

    # Wick structure properties
    wick_type: str = "annular_screen"    # 'annular_screen', 'screen', 'sintered', 'groove'
    annulus_gap: float = 0.00038         # Annular liquid channel gap (0.38 mm)
    screen_mesh_number: float = 15748.0  # mesh/m (~400 mesh per inch)
    wire_diameter: float = 2.5e-5        # wire diameter (m)
    porosity: float = 0.68               # Wick porosity
    effective_pore_radius: float = 3.5e-5# m (35 microns)
    permeability: float = 1.203e-8       # Darcy permeability (m2)

    # Wall material
    envelope_material: str = "Haynes-230"# 'Haynes-230', 'Mo-30Re', 'Nb-1Zr'
    k_wall: float = 24.5                 # Thermal conductivity of wall (W/m.K)

    # Orientation & Environment
    g_accel: float = 0.0                 # Acceleration m/s2 (0.0 for microgravity)
    tilt_angle_deg: float = 0.0          # Angle relative to gravity vector (deg)

    @property
    def L_total(self) -> float:
        return self.L_e + self.L_a + self.L_c

    @property
    def L_eff(self) -> float:
        """Effective length for fluid flow pressure drop: 0.5*Le + La + 0.5*Lc."""
        return 0.5 * self.L_e + self.L_a + 0.5 * self.L_c

    @property
    def r_outer(self) -> float:
        return self.d_outer / 2.0

    @property
    def r_inner(self) -> float:
        return self.r_outer - self.t_wall

    @property
    def r_vapor(self) -> float:
        return self.r_inner - self.t_wick

    @property
    def A_vapor(self) -> float:
        return math.pi * (self.r_vapor**2)

    @property
    def A_wick(self) -> float:
        return math.pi * ((self.r_inner**2) - (self.r_vapor**2))

    @property
    def A_annulus(self) -> float:
        """Cross-sectional area of open liquid annulus."""
        if self.wick_type == "annular_screen":
            r_gap_inner = self.r_inner - self.annulus_gap
            return math.pi * ((self.r_inner**2) - (r_gap_inner**2))
        return self.A_wick


class ReducedOrderHeatPipeSimulator:
    """
    1D/Reduced-Order steady-state and transient solver for high-temperature
    two-phase flow heat pipes in space reactors.
    """

    def __init__(self, geometry: HeatPipeGeometry, fluid: str = "Sodium"):
        self.geo = geometry
        self.fluid = fluid

    def get_properties(self, T: float) -> LiquidMetalProperties:
        if self.fluid.lower() == "sodium":
            return get_sodium_properties(T)
        elif self.fluid.lower() in ["nak", "nak78", "nak-78"]:
            return get_nak78_properties(T)
        else:
            return get_sodium_properties(T)

    def calculate_wick_effective_conductivity(self, k_l: float) -> float:
        """Effective thermal conductivity of liquid metal filled wick."""
        if self.geo.wick_type == "annular_screen":
            # Dominated by liquid sodium conduction across the thin annulus
            return k_l
        k_s = self.geo.k_wall
        eps = self.geo.porosity
        # Maxwell correlation for porous media
        k_eff = k_l * ((k_s + k_l - (1 - eps) * (k_l - k_s)) /
                       (k_s + k_l + (1 - eps) * (k_l - k_s)))
        return k_eff

    def calculate_operating_limits(self, T_sat: float) -> Dict[str, float]:
        """
        Calculates the 5 classic physical heat pipe limits at vapor temperature T_sat:
        1. Capillary Limit (Q_cap)
        2. Sonic Limit (Q_sonic)
        3. Entrainment Limit (Q_ent)
        4. Boiling Limit (Q_boil)
        5. Viscous Limit (Q_visc)
        """
        prop = self.get_properties(T_sat)
        geo = self.geo
        L_eff = geo.L_eff
        A_v = geo.A_vapor
        r_v = geo.r_vapor
        r_eff = geo.effective_pore_radius

        # Effective flow area and permeability for liquid return
        if geo.wick_type == "annular_screen":
            A_flow_l = geo.A_annulus
            K_w = (geo.annulus_gap**2) / 12.0
        else:
            A_flow_l = geo.A_wick
            K_w = geo.permeability

        # Maximum capillary head: Delta P_c_max = 2 * sigma / r_eff
        delta_P_c_max = (2.0 * prop.sigma) / r_eff

        # Hydrostatic / acceleration head: Delta P_g = rho_l * g * L_total * sin(angle)
        g_component = geo.g_accel * math.sin(math.radians(geo.tilt_angle_deg))
        delta_P_g = prop.rho_l * g_component * geo.L_total

        # Liquid frictional coefficient factor: F_l = mu_l / (rho_l * A_flow_l * K_w * h_fg)
        F_l = prop.mu_l / (prop.rho_l * A_flow_l * K_w * prop.h_fg)

        # Vapor frictional coefficient factor (Hagen-Poiseuille laminar approximation):
        # F_v = 8 * mu_v / (pi * rho_v * r_v^4 * h_fg)
        F_v = (8.0 * prop.mu_v) / (math.pi * prop.rho_v * (r_v**4) * prop.h_fg)

        # 1. CAPILLARY LIMIT (W)
        available_capillary = delta_P_c_max - delta_P_g
        if available_capillary > 0:
            Q_cap = available_capillary / ((F_l + F_v) * L_eff)
        else:
            Q_cap = 0.0

        # 2. SONIC LIMIT (W) - Levy choking limit
        # Q_sonic = A_v * rho_v * h_fg * sqrt((gamma * R_g * T) / (2 * (gamma + 1)))
        M_gas = 0.0229898 if self.fluid.lower() == "sodium" else 0.0355
        R_g = R_UNIVERSAL / M_gas
        sound_speed = math.sqrt(prop.gamma * R_g * T_sat)
        choked_flux = prop.rho_v * sound_speed * math.sqrt(1.0 / (2.0 * (prop.gamma + 1.0)))
        Q_sonic = A_v * prop.h_fg * choked_flux

        # 3. ENTRAINMENT LIMIT (W) - Cotter correlation
        # Q_ent = A_v * h_fg * sqrt( (sigma * rho_v) / (2 * r_hs) )
        r_hs = 0.5 * geo.effective_pore_radius  # hydraulic radius of surface wire pore
        Q_ent = A_v * prop.h_fg * math.sqrt((prop.sigma * prop.rho_v) / (2.0 * r_hs))

        # 4. BOILING LIMIT (W) - Nucleate boiling onset in porous wick
        k_eff_wick = self.calculate_wick_effective_conductivity(prop.k_l)
        r_nucleation = 1.0e-6  # typical nucleation site radius ~1.0 micron
        delta_T_crit = (2.0 * prop.sigma * T_sat) / (prop.h_fg * prop.rho_v * r_nucleation)
        log_ratio = math.log(geo.r_inner / geo.r_vapor)
        Q_boil = (2.0 * math.pi * geo.L_e * k_eff_wick * delta_T_crit) / max(1e-4, log_ratio)

        # 5. VISCOUS LIMIT (W) - Low temperature startup
        Q_visc = (A_v * (r_v**2) * prop.h_fg * prop.rho_v * prop.p_sat) / (16.0 * prop.mu_v * L_eff)

        # Overall allowable heat transport limit is the minimum of all active boundaries
        active_limits = [Q_cap, Q_sonic, Q_ent, Q_boil, Q_visc]
        Q_max = min(active_limits)

        return {
            "Q_max": Q_max,
            "Q_capillary": Q_cap,
            "Q_sonic": Q_sonic,
            "Q_entrainment": Q_ent,
            "Q_boiling": Q_boil,
            "Q_viscous": Q_visc,
            "delta_P_c_max": delta_P_c_max,
            "delta_P_g": delta_P_g,
            "sound_speed": sound_speed,
            "T_sat": T_sat
        }

    def solve_thermal_network(self, Q_thermal: float, T_evap_sink: float) -> Dict[str, Any]:
        """
        Solves the equivalent thermal resistance network:
        R_wall_e -> R_wick_e -> R_vapor -> R_wick_c -> R_wall_c
        Returns temperatures along the heat pipe and pressure profiles.
        """
        prop = self.get_properties(T_evap_sink)
        geo = self.geo
        k_wick = self.calculate_wick_effective_conductivity(prop.k_l)

        # Thermal resistances (K/W)
        R_wall_e = math.log(geo.r_outer / geo.r_inner) / (2.0 * math.pi * geo.k_wall * geo.L_e)
        R_wick_e = math.log(geo.r_inner / geo.r_vapor) / (2.0 * math.pi * k_wick * geo.L_e)

        # Vapor core axial thermal resistance
        F_v = (8.0 * prop.mu_v) / (math.pi * prop.rho_v * (geo.r_vapor**4) * prop.h_fg)
        delta_P_v = F_v * geo.L_eff * Q_thermal
        R_vapor = max(1e-6, (prop.temperature * delta_P_v) / (prop.rho_v * (prop.h_fg**2) * max(1.0, Q_thermal)))

        R_wick_c = math.log(geo.r_inner / geo.r_vapor) / (2.0 * math.pi * k_wick * geo.L_c)
        R_wall_c = math.log(geo.r_outer / geo.r_inner) / (2.0 * math.pi * geo.k_wall * geo.L_c)

        R_total = R_wall_e + R_wick_e + R_vapor + R_wick_c + R_wall_c

        # Temperatures
        T_outer_e = T_evap_sink
        T_inner_e = T_outer_e - Q_thermal * R_wall_e
        T_vapor_e = T_inner_e - Q_thermal * R_wick_e
        T_vapor_c = T_vapor_e - Q_thermal * R_vapor
        T_inner_c = T_vapor_c - Q_thermal * R_wick_c
        T_outer_c = T_inner_c - Q_thermal * R_wall_c

        # Fluid pressures
        if geo.wick_type == "annular_screen":
            A_flow_l = geo.A_annulus
            K_w = (geo.annulus_gap**2) / 12.0
        else:
            A_flow_l = geo.A_wick
            K_w = geo.permeability

        F_l = prop.mu_l / (prop.rho_l * A_flow_l * K_w * prop.h_fg)
        delta_P_l = F_l * geo.L_eff * Q_thermal
        delta_P_total = delta_P_v + delta_P_l
        delta_P_cap_max = (2.0 * prop.sigma) / geo.effective_pore_radius

        capillary_margin = 1.0 - (delta_P_total / max(1.0, delta_P_cap_max))

        # Vapor velocity and Mach number
        mass_flow = Q_thermal / prop.h_fg
        v_vapor_max = mass_flow / (prop.rho_v * geo.A_vapor)
        M_gas = 0.0229898 if self.fluid.lower() == "sodium" else 0.0355
        R_g = R_UNIVERSAL / M_gas
        sound_speed = math.sqrt(prop.gamma * R_g * T_vapor_e)
        mach_number = v_vapor_max / sound_speed

        return {
            "Q_thermal_W": Q_thermal,
            "R_total_KW": R_total,
            "R_components": {
                "R_wall_e": R_wall_e,
                "R_wick_e": R_wick_e,
                "R_vapor": R_vapor,
                "R_wick_c": R_wick_c,
                "R_wall_c": R_wall_c,
            },
            "temperatures_K": {
                "T_outer_evap": T_outer_e,
                "T_inner_evap": T_inner_e,
                "T_vapor_evap": T_vapor_e,
                "T_vapor_cond": T_vapor_c,
                "T_inner_cond": T_inner_c,
                "T_outer_cond": T_outer_c,
                "delta_T_total": T_outer_e - T_outer_c
            },
            "pressure_drops_Pa": {
                "delta_P_vapor": delta_P_v,
                "delta_P_liquid": delta_P_l,
                "delta_P_total": delta_P_total,
                "delta_P_cap_max": delta_P_cap_max,
                "capillary_margin": capillary_margin
            },
            "flow_dynamics": {
                "mass_flow_rate_kg_s": mass_flow,
                "v_vapor_max_m_s": v_vapor_max,
                "sound_speed_m_s": sound_speed,
                "mach_number": mach_number
            }
        }

    def generate_axial_profile(self, Q_thermal: float, T_evap: float, n_nodes: int = 50) -> Dict[str, List[float]]:
        """Generates continuous 1D axial profiles (z, T_wall, T_vapor, P_vapor, P_liquid)."""
        sol = self.solve_thermal_network(Q_thermal, T_evap)
        geo = self.geo
        L_tot = geo.L_total
        z_vals = [i * (L_tot / (n_nodes - 1)) for i in range(n_nodes)]

        T_outer_e = sol["temperatures_K"]["T_outer_evap"]
        T_outer_c = sol["temperatures_K"]["T_outer_cond"]
        T_v_e = sol["temperatures_K"]["T_vapor_evap"]
        T_v_c = sol["temperatures_K"]["T_vapor_cond"]
        delta_P_v = sol["pressure_drops_Pa"]["delta_P_vapor"]
        delta_P_l = sol["pressure_drops_Pa"]["delta_P_liquid"]
        P_ref = 1e5

        t_wall_profile = []
        t_vapor_profile = []
        p_vapor_profile = []
        p_liquid_profile = []

        for z in z_vals:
            if z <= geo.L_e:
                frac = z / geo.L_e
                t_w = T_outer_e - 0.5 * (T_outer_e - sol["temperatures_K"]["T_inner_evap"]) * frac
                t_v = T_v_e
                p_v = P_ref - delta_P_v * (frac**2) * 0.5
                p_l = (P_ref - delta_P_v) - delta_P_l * (1.0 - frac * 0.5)
            elif z <= (geo.L_e + geo.L_a):
                frac_a = (z - geo.L_e) / geo.L_a
                t_w = sol["temperatures_K"]["T_inner_evap"] - (sol["temperatures_K"]["T_inner_evap"] - sol["temperatures_K"]["T_inner_cond"]) * frac_a * 0.1
                t_v = T_v_e - (T_v_e - T_v_c) * frac_a
                p_v = P_ref - 0.5 * delta_P_v - 0.5 * delta_P_v * frac_a
                p_l = (P_ref - delta_P_v) - delta_P_l * (0.5 - 0.2 * frac_a)
            else:
                frac_c = (z - geo.L_e - geo.L_a) / geo.L_c
                t_w = sol["temperatures_K"]["T_inner_cond"] - (sol["temperatures_K"]["T_inner_cond"] - T_outer_c) * frac_c
                t_v = T_v_c
                recovery_factor = 0.65
                p_v = (P_ref - delta_P_v) + delta_P_v * recovery_factor * frac_c
                p_l = (P_ref - delta_P_v) - delta_P_l * 0.2 * (1.0 - frac_c)

            t_wall_profile.append(round(t_w, 2))
            t_vapor_profile.append(round(t_v, 2))
            p_vapor_profile.append(round(p_v, 1))
            p_liquid_profile.append(round(p_l, 1))

        return {
            "z_m": [round(z, 4) for z in z_vals],
            "T_wall_K": t_wall_profile,
            "T_vapor_K": t_vapor_profile,
            "P_vapor_Pa": p_vapor_profile,
            "P_liquid_Pa": p_liquid_profile
        }
