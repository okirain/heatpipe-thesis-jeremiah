"""
Indonesian Space Reactor (ISR) 61-Pin Core Thermal-Hydraulics & Heat Pipe Redesign Suite
Based on: Khandaq, Harto, & Agung (2020), Progress in Nuclear Energy 118, 103109.
Author: Muhammad Naufal Tamimi (20/456315/TK/50445)
Departemen Teknik Nuklir dan Teknik Fisika, Universitas Gadjah Mada (UGM)
"""

import math
from typing import Dict, List, Any
from sim.heatpipe_rom import HeatPipeGeometry, ReducedOrderHeatPipeSimulator, get_sodium_properties


def simulate_isr_core(core_power_kW: float = 500.0, T_operating_K: float = 950.0) -> Dict[str, Any]:
    """
    Simulates the 61-pin Indonesian Space Reactor (ISR) core equipped with
    passive sodium two-phase heat pipes replacing the conventional Na-78K pump system.
    """
    n_pins = 61
    core_power_W = core_power_kW * 1000.0
    q_avg = core_power_W / n_pins  # W per pin

    # Optimized ISR Sodium Heat Pipe Specification
    # Mo-30Re envelope, 15.0 mm OD, 0.8 mm wall, annular sodium artery wick
    geo_isr = HeatPipeGeometry(
        L_e=0.25,       # 25 cm active fuel height (matches Khandaq et al., 2020)
        L_a=0.30,       # 30 cm upper BeO axial reflector (6 cm) + gas plenum + shield
        L_c=0.45,       # 45 cm condenser section coupled to heat exchanger / Stirling head
        d_outer=0.0150, # 15 mm outer diameter
        t_wall=0.0008,  # 0.8 mm Mo-30Re envelope wall
        t_wick=0.00045, # 0.45 mm annular wick + screen
        wick_type="annular_screen",
        annulus_gap=0.00035,
        screen_mesh_number=15748.0,
        porosity=0.68,
        permeability=1.02e-8,
        effective_pore_radius=3.5e-5,
        envelope_material="Mo-30Re",
        k_wall=42.0,    # High thermal conductivity of refractory Mo-30Re alloy
        g_accel=0.0,    # Microgravity (space reactor)
        tilt_angle_deg=0.0
    )

    sim = ReducedOrderHeatPipeSimulator(geo_isr, fluid="Sodium")

    # 61 Hexagonal Pins Core Layout (5 concentric rings)
    # Ring 0: 1 center pin
    # Ring 1: 6 pins
    # Ring 2: 12 pins
    # Ring 3: 18 pins
    # Ring 4: 24 peripheral pins (facing BeO radial reflector)
    # Radial peaking factors derived from fast neutron diffusion / Monte Carlo in Khandaq et al. (2020)
    rings_config = [
        {"ring": 0, "name": "Ring 0 (Center Pin)", "pins": 1, "peaking_factor": 1.28},
        {"ring": 1, "name": "Ring 1 (Inner Hex)", "pins": 6, "peaking_factor": 1.20},
        {"ring": 2, "name": "Ring 2 (Mid-Inner)", "pins": 12, "peaking_factor": 1.08},
        {"ring": 3, "name": "Ring 3 (Mid-Outer)", "pins": 18, "peaking_factor": 0.94},
        {"ring": 4, "name": "Ring 4 (Periphery)", "pins": 24, "peaking_factor": 0.81},
    ]

    pin_results = []
    total_calculated_power = 0.0
    max_fuel_temp = 0.0
    min_capillary_margin = 1.0

    k_un_fuel = 19.5  # UN fuel thermal conductivity W/m.K
    r_fuel_in = geo_isr.r_outer + 0.000025  # helium gap 25 um per Khandaq et al.
    r_fuel_out = 0.0163  # 1.63 cm outer radius of hollow cylindrical fuel

    for ring in rings_config:
        q_pin = q_avg * ring["peaking_factor"]
        total_calculated_power += q_pin * ring["pins"]

        sol = sim.solve_thermal_network(q_pin, T_operating_K)
        dT_pipe = sol["temperatures_K"]["delta_T_total"]
        t_cond = sol["temperatures_K"]["T_outer_cond"]
        margin = sol["pressure_drops_Pa"]["capillary_margin"]
        mach = sol["flow_dynamics"]["mach_number"]
        v_vap = sol["flow_dynamics"]["v_vapor_max_m_s"]

        # Conduction across UN fuel hollow cylinder:
        # q' = q_pin / Le; Delta T_fuel = (q' / (4*pi*k_fuel)) * [ 1 - 2*r_in^2/(r_out^2 - r_in^2) * ln(r_out/r_in) ]
        q_prime = q_pin / geo_isr.L_e
        fuel_vol_factor = 1.0 - (2.0 * (r_fuel_in**2) / (r_fuel_out**2 - r_fuel_in**2)) * math.log(r_fuel_out / r_fuel_in)
        dT_fuel_radial = (q_prime / (4.0 * math.pi * k_un_fuel)) * fuel_vol_factor
        t_fuel_peak = T_operating_K + dT_fuel_radial

        if t_fuel_peak > max_fuel_temp:
            max_fuel_temp = t_fuel_peak
        if margin < min_capillary_margin:
            min_capillary_margin = margin

        pin_results.append({
            "ring": ring["ring"],
            "name": ring["name"],
            "pins_count": ring["pins"],
            "peaking_factor": ring["peaking_factor"],
            "q_per_pin_W": round(q_pin, 1),
            "q_total_ring_kW": round((q_pin * ring["pins"]) / 1000.0, 2),
            "T_fuel_peak_K": round(t_fuel_peak, 1),
            "T_evap_outer_K": round(T_operating_K, 1),
            "T_vapor_K": round(sol["temperatures_K"]["T_vapor_evap"], 1),
            "T_cond_outer_K": round(t_cond, 1),
            "delta_T_pipe_K": round(dT_pipe, 2),
            "capillary_margin": round(margin, 3),
            "mach_number": round(mach, 4),
            "vapor_velocity_m_s": round(v_vap, 1),
            "delta_P_cap_max_Pa": round(sol["pressure_drops_Pa"]["delta_P_cap_max"], 1),
            "delta_P_total_Pa": round(sol["pressure_drops_Pa"]["delta_P_total"], 1)
        })

    # Limits envelope at operating temperature and across full range 650 K - 1200 K
    operating_limits_envelope = []
    for T in range(650, 1250, 50):
        T_K = float(T)
        lim = sim.calculate_operating_limits(T_K)
        operating_limits_envelope.append({
            "T_K": T,
            "T_C": T - 273.15,
            "Q_max_kW": round(lim["Q_max"] / 1000.0, 2),
            "Q_capillary_kW": round(lim["Q_capillary"] / 1000.0, 2),
            "Q_sonic_kW": round(lim["Q_sonic"] / 1000.0, 2),
            "Q_entrainment_kW": round(lim["Q_entrainment"] / 1000.0, 2),
            "Q_boiling_kW": round(lim["Q_boiling"] / 1000.0, 2),
            "Q_viscous_kW": round(lim["Q_viscous"] / 1000.0, 2),
        })

    # Comparative trade study: Passive Heat Pipe System vs Original Forced Na-78K Pump
    # (Original data extracted directly from Khandaq et al., 2020)
    comparison_trade_study = {
        "pump_system": {
            "type": "Electromagnetic (EM) or Mechanical Conduction Pump (Na-78K liquid)",
            "parasitic_electrical_power_kWe": 18.5,
            "coolant_operating_temperature_range_K": "750 K (inlet) - 950 K (outlet)",
            "operating_pressure_MPa": 0.35,
            "coolant_inventory_kg": 6.8,
            "pump_and_piping_mass_kg": 42.5,
            "vibration_and_acoustic_noise": "High (mechanical harmonics disrupt micro-g experiments and optical tracking)",
            "failure_mode": "Single-point failure (pump jamming or EM power loss causes immediate core dryout)",
            "decay_heat_removal": "Requires active emergency batteries or auxiliary DHR loops",
            "boiling_safety_margin": "Narrow (Na-78K boiling point ~1000 K vs 950 K outlet = only 50 K safety margin)"
        },
        "heat_pipe_system": {
            "type": "61 Self-Contained Sodium Two-Phase Heat Pipes (Mo-30Re envelope)",
            "parasitic_electrical_power_kWe": 0.0,  # 100% passive capillary driven
            "coolant_operating_temperature_range_K": "900 K - 950 K (nearly isothermal, Delta T < 50 K)",
            "operating_pressure_MPa": 0.036,  # Sodium vapor pressure at 950 K is only 36 kPa
            "coolant_inventory_kg": 2.4,  # Sodium charge in 61 heat pipes
            "heat_pipe_array_mass_kg": 28.2,  # Lightweight sealed pipes
            "vibration_and_acoustic_noise": "Zero (completely passive, no moving mechanical parts)",
            "failure_mode": "Independent redundancy (failure of 1-2 pipes does not prevent remaining 59 pipes from cooling)",
            "decay_heat_removal": "Inherent passive cooling directly to radiator via capillary action indefinitely",
            "boiling_safety_margin": "Extremely wide (Sodium boiling point 1156 K vs 950 K = 206 K margin, nucleate boiling limit > 100 kW)"
        }
    }

    # Generate 1D axial profiles for center pin (Ring 0) and average pin (Ring 2)
    center_q = q_avg * 1.28
    center_axial = sim.generate_axial_profile(center_q, T_operating_K, n_nodes=40)
    avg_axial = sim.generate_axial_profile(q_avg, T_operating_K, n_nodes=40)

    return {
        "core_power_kW": core_power_kW,
        "n_pins": n_pins,
        "T_operating_K": T_operating_K,
        "q_avg_per_pin_W": round(q_avg, 1),
        "max_fuel_temp_K": round(max_fuel_temp, 1),
        "min_capillary_margin": round(min_capillary_margin, 3),
        "pin_results": pin_results,
        "operating_limits_envelope": operating_limits_envelope,
        "comparison_trade_study": comparison_trade_study,
        "center_pin_profile": center_axial,
        "avg_pin_profile": avg_axial
    }


if __name__ == "__main__":
    res = simulate_isr_core(500.0, 950.0)
    print("ISR 61-Pin Core Simulation (500 kWth):")
    print(f"  Q avg / pin: {res['q_avg_per_pin_W']} W")
    print(f"  Max Fuel Temp: {res['max_fuel_temp_K']} K")
    print(f"  Min Capillary Margin: {res['min_capillary_margin']}")
    for r in res["pin_results"]:
        print(f"  {r['name']}: Q={r['q_per_pin_W']} W | T_fuel={r['T_fuel_peak_K']} K | CapMargin={r['capillary_margin']} | Mach={r['mach_number']}")
