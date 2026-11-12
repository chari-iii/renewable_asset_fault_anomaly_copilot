"""
Standalone Training and Evaluation Script.
Trains models, evaluates with chronological splits, and generates evaluation report.
Usage:
    python backend/train_and_evaluate.py
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline import RenewableAssetPipeline, MODELS_DIR

def main():
    print("=" * 70)
    print("GREENKO RENEWABLE ASSET FAULT & ANOMALY COPILOT - EVALUATION SUITE")
    print("=" * 70)

    pipeline = RenewableAssetPipeline()
    metrics = pipeline.metrics

    print("\n[1] EXPECTED POWER GENERATION MODELS (VARIANCE BASELINE)")
    print("-" * 70)
    s_rf = metrics['solar_power_forecasting']['chosen_model_rf']
    s_base = metrics['solar_power_forecasting']['baseline_linear']
    print(f"Solar Inverter (SOLAR_INV_01) - Chronological Test Split ({metrics['solar_power_forecasting']['test_samples']} points):")
    print(f"  * Chosen Model (RandomForestRegressor): MAE = {s_rf['mae_mw']:.4f} MW | RMSE = {s_rf['rmse_mw']:.4f} MW | R2 = {s_rf['r2_score']:.4f}")
    print(f"  * Baseline Model (Linear Physics):      MAE = {s_base['mae_mw']:.4f} MW | RMSE = {s_base['rmse_mw']:.4f} MW | R2 = {s_base['r2_score']:.4f}")

    w_rf = metrics['wind_power_forecasting']['chosen_model_rf']
    w_base = metrics['wind_power_forecasting']['baseline_linear']
    print(f"\nWind Turbine (WTG_07) - Chronological Test Split ({metrics['wind_power_forecasting']['test_samples']} points):")
    print(f"  * Chosen Model (RandomForestRegressor): MAE = {w_rf['mae_mw']:.4f} MW | RMSE = {w_rf['rmse_mw']:.4f} MW | R2 = {w_rf['r2_score']:.4f}")
    print(f"  * Baseline Model (Linear Wind Speed):   MAE = {w_base['mae_mw']:.4f} MW | RMSE = {w_base['rmse_mw']:.4f} MW | R2 = {w_base['r2_score']:.4f}")

    print("\n[2] ANOMALY DETECTION EVALUATION (ISOLATION FOREST vs GROUND TRUTH)")
    print("-" * 70)
    s_iso = metrics['solar_anomaly_detection']
    print(f"Solar Anomaly Detection:")
    print(f"  * Algorithm: {s_iso['algorithm']} (contamination={s_iso['contamination_setting']}, n_estimators={s_iso['n_estimators']})")
    print(f"  * Known Fault Recall: {s_iso['known_fault_capture']} ({s_iso['recall']*100:.1f}%) | Precision: {s_iso['precision']*100:.1f}% | F1: {s_iso['f1_score']:.3f}")
    print(f"  * Flagged Timestamps: {s_iso['flagged_anomalies']}")

    w_iso = metrics['wind_anomaly_detection']
    print(f"\nWind Anomaly Detection:")
    print(f"  * Algorithm: {w_iso['algorithm']} (contamination={w_iso['contamination_setting']}, n_estimators={w_iso['n_estimators']})")
    print(f"  * Known Fault Recall: {w_iso['known_fault_capture']} ({w_iso['recall']*100:.1f}%) | Precision: {w_iso['precision']*100:.1f}% | F1: {w_iso['f1_score']:.3f}")
    print(f"  * Flagged Timestamps: {w_iso['flagged_anomalies']}")

    # Save metrics JSON to models directory
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[OK] Evaluation metrics saved to: {metrics_path}")

    # Display detected anomalies
    anomalies = pipeline.get_detected_anomalies()
    print(f"\n[3] CONFIRMED CRITICAL FAULT EVENTS ({len([a for a in anomalies if a.severity == 'CRITICAL'])})")
    print("-" * 70)
    for a in anomalies:
        if a.severity == 'CRITICAL':
            print(f"  [CRITICAL] Asset {a.asset_id} (TS #{a.timestamp_id:03d}, Hour {a.hour:02d}:00):")
            print(f"     Temp: {a.temperature_c}°C | Vib: {a.vibration_mm_s} mm/s | Power: {a.power_mw} MW (Exp: {a.expected_power_mw} MW)")
            print(f"     Fault Diagnosis: {a.fault_type}")

    print("\n" + "=" * 70)
    print("EVALUATION & VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
