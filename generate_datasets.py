"""
Data Generator script: produces JSON and CSV artifacts for the Quarto Thesis and Frontend.
Author: Muhammad Naufal Tamimi (20/456315/TK/50445)
"""

import json
import csv
import os
from sim.krusty_benchmark import get_krusty_benchmark_data
from sim.isr_core_heatpipe import simulate_isr_core
from sim.heatpipe_rom import HeatPipeGeometry, ReducedOrderHeatPipeSimulator, get_sodium_properties, get_nak78_properties


def generate_all_artifacts():
    out_dir = "public/data"
    os.makedirs(out_dir, exist_ok=True)

    # 1. KRUSTY Benchmark Dataset
    krusty_data = get_krusty_benchmark_data()
    with open(os.path.join(out_dir, "krusty_benchmark.json"), "w") as f:
        json.dump(krusty_data, f, indent=2)

    with open(os.path.join(out_dir, "krusty_validation.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run_ID", "Power_W", "T_evap_C", "T_cond_C", "DeltaT_exp_K", "DeltaT_sim_K", "Error_pct", "R_th_exp_KW", "R_th_sim_KW", "Capillary_Margin", "Mach_Number"])
        for r in krusty_data["comparison_records"]:
            writer.writerow([r["run"], r["power_W"], r["T_evap_C"], r["T_cond_C"], r["delta_T_exp_K"], r["delta_T_sim_K"], r["error_pct"], r["R_th_exp_KW"], r["R_th_sim_KW"], r["capillary_margin"], r["mach_number"]])

    # 2. ISR 61-Pin Core Simulation Dataset
    isr_data = simulate_isr_core(core_power_kW=500.0, T_operating_K=980.0)
    with open(os.path.join(out_dir, "isr_core_simulation.json"), "w") as f:
        json.dump(isr_data, f, indent=2)

    with open(os.path.join(out_dir, "isr_pin_distribution.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ring", "Ring_Name", "Pins_Count", "Peaking_Factor", "Q_per_pin_W", "T_fuel_peak_K", "T_evap_K", "T_cond_K", "DeltaT_pipe_K", "Capillary_Margin", "Mach_Number"])
        for p in isr_data["pin_results"]:
            writer.writerow([p["ring"], p["name"], p["pins_count"], p["peaking_factor"], p["q_per_pin_W"], p["T_fuel_peak_K"], p["T_evap_outer_K"], p["T_cond_outer_K"], p["delta_T_pipe_K"], p["capillary_margin"], p["mach_number"]])

    # 3. Thermophysical Properties Table (Sodium & NaK-78)
    prop_records = []
    with open(os.path.join(out_dir, "sodium_nak_properties.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Temperature_K", "Na_Psat_kPa", "Na_rho_l_kg_m3", "Na_rho_v_kg_m3", "Na_hfg_kJ_kg", "Na_sigma_N_m", "Na_mu_l_mPa_s", "Na_k_l_W_mK", "NaK_Psat_kPa", "NaK_rho_l_kg_m3"])
        for T in range(600, 1350, 50):
            na = get_sodium_properties(float(T))
            nak = get_nak78_properties(float(T))
            row = [
                T,
                round(na.p_sat / 1000.0, 3),
                round(na.rho_l, 1),
                round(na.rho_v, 4),
                round(na.h_fg / 1000.0, 1),
                round(na.sigma, 4),
                round(na.mu_l * 1000.0, 4),
                round(na.k_l, 2),
                round(nak.p_sat / 1000.0, 3),
                round(nak.rho_l, 1)
            ]
            writer.writerow(row)
            prop_records.append({
                "T_K": T,
                "Na_Psat_kPa": round(na.p_sat / 1000.0, 3),
                "Na_rho_l": round(na.rho_l, 1),
                "Na_rho_v": round(na.rho_v, 4),
                "Na_hfg_kJ_kg": round(na.h_fg / 1000.0, 1),
                "Na_sigma": round(na.sigma, 4),
                "Na_mu_l_mPa_s": round(na.mu_l * 1000.0, 4),
                "Na_k_l": round(na.k_l, 2),
                "NaK_Psat_kPa": round(nak.p_sat / 1000.0, 3),
                "NaK_rho_l": round(nak.rho_l, 1)
            })

    with open(os.path.join(out_dir, "thermophysical_properties.json"), "w") as f:
        json.dump(prop_records, f, indent=2)

    print("All dataset artifacts successfully generated in public/data/")


if __name__ == "__main__":
    generate_all_artifacts()
