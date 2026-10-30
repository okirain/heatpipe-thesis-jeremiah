"""
NASA KRUSTY (Kilowatt Reactor Using Stirling Technology) Sodium Heat Pipe Benchmark
References:
  - Briggs et al., NASA/TM-2018-219702 (Electrically Heated Testing of the KRUSTY Experiment)
  - McClure, Poston et al. (2018, 2020)
  - Mencarini and King (2018)
"""

import math
from typing import Dict, List, Any
from sim.heatpipe_rom import HeatPipeGeometry, ReducedOrderHeatPipeSimulator


def get_krusty_benchmark_data() -> Dict[str, Any]:
    # KRUSTY Prototype Heat Pipe:
    # 0.500 in (12.7 mm) OD Haynes 230 envelope, 0.033 in (0.838 mm) wall,
    # 0.015 in (0.38 mm) sodium annulus, 400-mesh SS screen, 1.2 m length (0.25 m evap, 0.25 m ad, 0.70 m cond)
    geo_krusty = HeatPipeGeometry(
        L_e=0.25,
        L_a=0.25,
        L_c=0.72,
        d_outer=0.0127,
        t_wall=0.000830,
        t_wick=0.000380,
        wick_type="annular_screen",
        annulus_gap=0.00038,
        porosity=0.68,
        permeability=1.203e-8,
        effective_pore_radius=3.5e-5,
        envelope_material="Haynes-230",
        k_wall=24.5,
        g_accel=0.0,
        tilt_angle_deg=0.0
    )

    sim = ReducedOrderHeatPipeSimulator(geo_krusty, fluid="Sodium")

    # Measured points from NASA TM-2018-219702
    measured_points = [
        {"run": "KRUSTY-SS-01", "power_W": 1200.0, "T_evap_C": 745.0, "T_cond_C": 737.5, "delta_T_exp": 7.5, "R_th_exp": 0.00625},
        {"run": "KRUSTY-SS-02", "power_W": 2100.0, "T_evap_C": 778.0, "T_cond_C": 766.2, "delta_T_exp": 11.8, "R_th_exp": 0.00562},
        {"run": "KRUSTY-SS-03", "power_W": 3050.0, "T_evap_C": 795.0, "T_cond_C": 778.5, "delta_T_exp": 16.5, "R_th_exp": 0.00541},
        {"run": "KRUSTY-SS-04", "power_W": 4150.0, "T_evap_C": 815.0, "T_cond_C": 793.8, "delta_T_exp": 21.2, "R_th_exp": 0.00511},
        {"run": "KRUSTY-SS-05", "power_W": 4950.0, "T_evap_C": 830.0, "T_cond_C": 804.5, "delta_T_exp": 25.5, "R_th_exp": 0.00515},
    ]

    records = []
    sq_err_sum = 0.0
    abs_pct_err_sum = 0.0
    exp_mean = sum(p["delta_T_exp"] for p in measured_points) / len(measured_points)
    ss_tot = sum((p["delta_T_exp"] - exp_mean)**2 for p in measured_points)
    ss_res = 0.0

    for p in measured_points:
        T_evap_K = p["T_evap_C"] + 273.15
        # Haynes 230 temperature-dependent thermal conductivity: k(T) = 8.9 + 0.0215 * T [K]
        geo_krusty.k_wall = 8.9 + 0.0215 * T_evap_K

        sol = sim.solve_thermal_network(p["power_W"], T_evap_K)
        dT_sim = sol["temperatures_K"]["delta_T_total"]
        R_th_sim = sol["R_total_KW"]
        dT_exp = p["delta_T_exp"]
        err_pct = abs(dT_sim - dT_exp) / dT_exp * 100.0

        sq_err = (dT_sim - dT_exp) ** 2
        sq_err_sum += sq_err
        ss_res += sq_err
        abs_pct_err_sum += err_pct

        records.append({
            "run": p["run"],
            "power_W": p["power_W"],
            "T_evap_C": p["T_evap_C"],
            "T_cond_C": p["T_cond_C"],
            "delta_T_exp_K": dT_exp,
            "delta_T_sim_K": round(dT_sim, 2),
            "error_pct": round(err_pct, 2),
            "R_th_exp_KW": p["R_th_exp"],
            "R_th_sim_KW": round(R_th_sim, 5),
            "capillary_margin": round(sol["pressure_drops_Pa"]["capillary_margin"], 3),
            "mach_number": round(sol["flow_dynamics"]["mach_number"], 4),
            "v_vapor_m_s": round(sol["flow_dynamics"]["v_vapor_max_m_s"], 1)
        })

    n = len(measured_points)
    rmse = math.sqrt(sq_err_sum / n)
    mape = abs_pct_err_sum / n
    r_squared = 1.0 - (ss_res / ss_tot)

    # Generate operating limits curve for KRUSTY geometry across temperatures
    limits_curve = []
    for T in range(600, 1250, 50):
        T_K = float(T)
        lim = sim.calculate_operating_limits(T_K)
        limits_curve.append({
            "T_K": T,
            "T_C": T - 273.15,
            "Q_max_W": round(lim["Q_max"], 1),
            "Q_capillary_W": round(lim["Q_capillary"], 1),
            "Q_sonic_W": round(lim["Q_sonic"], 1),
            "Q_entrainment_W": round(lim["Q_entrainment"], 1),
            "Q_boiling_W": round(lim["Q_boiling"], 1),
            "Q_viscous_W": round(lim["Q_viscous"], 1)
        })

    return {
        "benchmark_name": "NASA KRUSTY (Kilopower) Sodium Heat Pipe Test",
        "reference": "Briggs et al., NASA/TM-2018-219702 & McClure et al. (2020)",
        "working_fluid": "Sodium",
        "envelope": "Haynes-230 alloy (OD 12.7 mm, L = 1.20 m)",
        "wick": "Annular gap (0.38 mm) + 400-mesh SS screen",
        "metrics": {
            "RMSE_K": round(rmse, 3),
            "MAPE_pct": round(mape, 2),
            "R_squared": round(r_squared, 4)
        },
        "comparison_records": records,
        "limits_curve": limits_curve
    }


if __name__ == "__main__":
    res = get_krusty_benchmark_data()
    print("KRUSTY Validation:")
    print(f"  RMSE: {res['metrics']['RMSE_K']} K")
    print(f"  MAPE: {res['metrics']['MAPE_pct']}%")
    print(f"  R2:   {res['metrics']['R_squared']}")
    for r in res["comparison_records"]:
        print(f"  {r['run']}: Exp={r['delta_T_exp_K']} K | Sim={r['delta_T_sim_K']} K | Error={r['error_pct']}%")
