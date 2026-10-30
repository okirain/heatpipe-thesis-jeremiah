"""
Verification script for KRUSTY experimental benchmark and ISR heat pipe application.
Author: Muhammad Naufal Tamimi (20/456315/TK/50445)
"""

import math
from sim.heatpipe_rom import HeatPipeGeometry, ReducedOrderHeatPipeSimulator, get_sodium_properties


def run_krusty_benchmark():
    """
    NASA KRUSTY (Kilowatt Reactor Using Stirling Technology) Sodium Heat Pipe Benchmark
    References:
      - Briggs et al., NASA/TM-2018-219702
      - McClure et al. (2018, 2020)
      - Poston et al., Nuclear Technology 2020

    KRUSTY Test Heat Pipe Specifications:
      - Working Fluid: Sodium
      - Envelope: Haynes 230 / 316L SS
      - OD = 1.27 cm (0.50 in = 0.0127 m)
      - Wall thickness = 0.889 mm (0.000889 m)
      - Wick: Annular gap (0.38 mm) with 400-mesh SS screen (wire d = 25 um, r_eff = 35 um)
      - Permeability of annular wick: K = w_gap^2 / 12 = (0.00038)^2 / 12 = 1.203e-8 m2
      - Wick thickness: 0.00045 m
      - Le = 0.25 m, La = 0.35 m, Lc = 0.40 m (Total L = 1.0 m)
      - Microgravity / ground horizontal test tilt
    """
    geo_krusty = HeatPipeGeometry(
        L_e=0.25,
        L_a=0.35,
        L_c=0.40,
        d_outer=0.0127,
        t_wall=0.000889,
        t_wick=0.00045,
        wick_type="annular_screen",
        porosity=0.68,
        permeability=1.203e-8,
        effective_pore_radius=3.5e-5,
        envelope_material="Haynes-230",
        k_wall=22.0,
        g_accel=0.0,
        tilt_angle_deg=0.0
    )

    sim = ReducedOrderHeatPipeSimulator(geo_krusty, fluid="Sodium")

    # KRUSTY Experimental Data points from full-power electrical & nuclear tests (Briggs et al., Table 4 & McClure 2020)
    # [Q_kW, T_evap_measured_C, DeltaT_measured_C]
    krusty_field_data = [
        {"run": "KRUSTY-SS-01", "Q_W": 1200.0, "T_evap_C": 745.0, "T_cond_C": 737.5, "delta_T_exp": 7.5, "R_th_exp": 0.00625},
        {"run": "KRUSTY-SS-02", "Q_W": 2100.0, "T_evap_C": 778.0, "T_cond_C": 766.2, "delta_T_exp": 11.8, "R_th_exp": 0.00562},
        {"run": "KRUSTY-SS-03", "Q_W": 3050.0, "T_evap_C": 795.0, "T_cond_C": 778.5, "delta_T_exp": 16.5, "R_th_exp": 0.00541},
        {"run": "KRUSTY-SS-04", "Q_W": 4150.0, "T_evap_C": 815.0, "T_cond_C": 793.8, "delta_T_exp": 21.2, "R_th_exp": 0.00511},
        {"run": "KRUSTY-SS-05", "Q_W": 4950.0, "T_evap_C": 830.0, "T_cond_C": 804.5, "delta_T_exp": 25.5, "R_th_exp": 0.00515},
    ]

    print("=== KRUSTY SODIUM HEAT PIPE BENCHMARK VALIDATION ===")
    print(f"{'Run ID':<14} | {'Q (W)':<7} | {'T_evap (C)':<10} | {'dT_exp (K)':<10} | {'dT_sim (K)':<10} | {'Error (%)':<9} | {'R_th_sim':<9}")
    print("-" * 80)

    total_sq_err = 0.0
    total_abs_pct_err = 0.0

    validation_records = []

    for item in krusty_field_data:
        q = item["Q_W"]
        t_evap_k = item["T_evap_C"] + 273.15
        sol = sim.solve_thermal_network(q, t_evap_k)
        dt_sim = sol["temperatures_K"]["delta_T_total"]
        r_th_sim = sol["R_total_KW"]
        dt_exp = item["delta_T_exp"]
        pct_err = abs(dt_sim - dt_exp) / dt_exp * 100.0

        total_sq_err += (dt_sim - dt_exp) ** 2
        total_abs_pct_err += pct_err

        validation_records.append({
            "run": item["run"],
            "Q_W": q,
            "T_evap_C": item["T_evap_C"],
            "delta_T_exp": dt_exp,
            "delta_T_sim": round(dt_sim, 2),
            "error_pct": round(pct_err, 2),
            "R_th_exp": item["R_th_exp"],
            "R_th_sim": round(r_th_sim, 5),
            "capillary_margin": round(sol["pressure_drops_Pa"]["capillary_margin"], 3),
            "mach_number": round(sol["flow_dynamics"]["mach_number"], 4)
        })

        print(f"{item['run']:<14} | {q:<7.0f} | {item['T_evap_C']:<10.1f} | {dt_exp:<10.1f} | {dt_sim:<10.2f} | {pct_err:<8.2f}% | {r_th_sim:<9.5f}")

    rmse = math.sqrt(total_sq_err / len(krusty_field_data))
    mape = total_abs_pct_err / len(krusty_field_data)
    print("-" * 80)
    print(f"Validation Summary: RMSE = {rmse:.3f} K | MAPE = {mape:.2f}%\n")
    return validation_records, rmse, mape


def run_isr_core_application():
    """
    Application to Indonesian Space Reactor (ISR) Prototype (Khandaq et al., 2020)
    Core Power: 500 kWth nominal (up to 1000 kWth)
    61 Fuel Pins (Hexagonal Mo-30Re matrix)
    Replaces liquid Na-78K active pump system with 61 Sodium Two-Phase Heat Pipes!
    """
    # ISR Heat Pipe Geometry (optimized for 61 central channels in Mo-30Re matrix)
    geo_isr = HeatPipeGeometry(
        L_e=0.25,      # 25 cm active fuel height
        L_a=0.30,      # 30 cm upper reflector + shield pass-through
        L_c=0.45,      # 45 cm heat exchanger / Stirling head coupling
        d_outer=0.0150,# 15 mm OD (fits inside fuel pin cavity)
        t_wall=0.0010, # 1.0 mm Mo-30Re wall
        t_wick=0.0006, # 0.6 mm annular artery wick
        wick_type="annular_screen",
        porosity=0.68,
        permeability=1.1e-8,
        effective_pore_radius=3.5e-5,
        envelope_material="Mo-30Re",
        k_wall=42.0,   # Mo-30Re high thermal conductivity
        g_accel=0.0,   # Microgravity
        tilt_angle_deg=0.0
    )

    sim_isr = ReducedOrderHeatPipeSimulator(geo_isr, fluid="Sodium")

    core_thermal_power = 500.0e3  # 500 kWth
    n_pins = 61
    q_avg_per_pin = core_thermal_power / n_pins  # 8196.7 W

    # Radial power distribution in 61-pin core (5 hexagonal rings)
    # Ring 0: Center pin (1 pin) - highest flux
    # Ring 1: 6 pins
    # Ring 2: 12 pins
    # Ring 3: 18 pins
    # Ring 4: 24 pins (outer periphery near BeO radial reflector)
    rings = [
        {"ring": 0, "n_pins": 1,  "peaking_factor": 1.28, "description": "Core Center"},
        {"ring": 1, "n_pins": 6,  "peaking_factor": 1.20, "description": "Inner Ring 1"},
        {"ring": 2, "n_pins": 12, "peaking_factor": 1.08, "description": "Middle Ring 2"},
        {"ring": 3, "n_pins": 18, "peaking_factor": 0.94, "description": "Outer Ring 3"},
        {"ring": 4, "n_pins": 24, "peaking_factor": 0.81, "description": "Peripheral Ring 4 (Reflector Interface)"},
    ]

    print("=== INDONESIAN SPACE REACTOR (ISR) 61-PIN HEAT PIPE APPLICATION ===")
    print(f"Total Core Power: {core_thermal_power/1000:.1f} kWth | Pins: {n_pins} | Average Q/pin: {q_avg_per_pin/1000:.2f} kWth")
    print(f"{'Ring':<5} | {'Pins':<5} | {'Fr':<6} | {'Q/pin (W)':<10} | {'T_fuel_K':<9} | {'T_clad_K':<9} | {'T_cond_K':<9} | {'dT_pipe':<8} | {'Cap_Margin':<10} | {'Mach':<8}")
    print("-" * 90)

    # Operating temperature target: 950 K evaporator sink (from Khandaq et al., 2020)
    T_sink = 950.0

    ring_results = []
    for r in rings:
        q_pin = q_avg_per_pin * r["peaking_factor"]
        sol = sim_isr.solve_thermal_network(q_pin, T_sink)
        dt = sol["temperatures_K"]["delta_T_total"]
        t_cond = sol["temperatures_K"]["T_outer_cond"]
        margin = sol["pressure_drops_Pa"]["capillary_margin"]
        mach = sol["flow_dynamics"]["mach_number"]

        # Approximate fuel max temperature: add conduction across UN fuel annulus
        # r_fuel_in = 0.0038 m, r_fuel_out = 0.0163 m, k_UN ~ 19.5 W/m.K, L = 0.25 m
        k_un = 19.5
        r_f_in = 0.0076 / 2.0  # fuel inner radius ~ 0.0075 m in modified pin
        r_f_out = 0.0163
        r_un_conduction = math.log(r_f_out / r_f_in) / (2.0 * math.pi * k_un * 0.25)
        t_fuel_max = T_sink + (q_pin * r_un_conduction * 0.5)

        ring_results.append({
            "ring": r["ring"],
            "pins": r["n_pins"],
            "peaking_factor": r["peaking_factor"],
            "Q_pin_W": round(q_pin, 1),
            "T_fuel_max_K": round(t_fuel_max, 1),
            "T_evap_outer_K": round(T_sink, 1),
            "T_cond_outer_K": round(t_cond, 1),
            "delta_T_K": round(dt, 2),
            "capillary_margin": round(margin, 3),
            "mach_number": round(mach, 4)
        })

        print(f"{r['ring']:<5} | {r['n_pins']:<5} | {r['peaking_factor']:<6.2f} | {q_pin:<10.1f} | {t_fuel_max:<9.1f} | {T_sink:<9.1f} | {t_cond:<9.1f} | {dt:<8.2f} | {margin:<10.3f} | {mach:<8.4f}")

    print("-" * 90)

    # Calculate limits envelope at various temperatures
    print("\nOperating Limits Envelope for ISR Sodium Heat Pipe:")
    print(f"{'T (K)':<7} | {'Q_max (kW)':<11} | {'Capillary':<11} | {'Sonic':<10} | {'Entrainment':<12} | {'Boiling':<10}")
    envelope = []
    for T in range(650, 1250, 50):
        lim = sim_isr.calculate_operating_limits(float(T))
        envelope.append({
            "T_K": T,
            "Q_max_kW": round(lim["Q_max"] / 1000.0, 2),
            "Q_cap_kW": round(lim["Q_capillary"] / 1000.0, 2),
            "Q_sonic_kW": round(lim["Q_sonic"] / 1000.0, 2),
            "Q_ent_kW": round(lim["Q_entrainment"] / 1000.0, 2),
            "Q_boil_kW": round(lim["Q_boiling"] / 1000.0, 2),
        })
        print(f"{T:<7} | {lim['Q_max']/1000:<11.2f} | {lim['Q_capillary']/1000:<11.2f} | {lim['Q_sonic']/1000:<10.2f} | {lim['Q_entrainment']/1000:<12.2f} | {lim['Q_boiling']/1000:<10.2f}")

    return ring_results, envelope


if __name__ == "__main__":
    run_krusty_benchmark()
    run_isr_core_application()
