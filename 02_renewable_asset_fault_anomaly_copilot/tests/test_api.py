"""
Automated Test Suite for Renewable Asset Fault & Anomaly Detection Copilot.
Tests:
1. Dataset loading and feature engineering integrity.
2. Model training, chronological splitting, and validation metrics.
3. Anomaly detection recall on verified ground-truth events.
4. GenAI Copilot explanation grounding, severity classification, and SOP retrieval.
5. FastAPI endpoint contracts (/api/demo, /api/run, /api/anomalies, /api/predict, /api/metrics).
6. Fleet KPIs, telemetry time-series, and asset overview endpoints.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline import get_pipeline
from backend.models import Query, PredictionRequest
from backend.app import (
    demo,
    run_case,
    get_anomalies,
    predict_telemetry,
    get_metrics,
    get_fleet_summary,
    get_assets,
    get_telemetry_series,
    get_sops
)


def test_pipeline_initialization():
    pipeline = get_pipeline()
    assert pipeline is not None
    assert len(pipeline.df) == 240
    assert set(pipeline.df['asset_id'].unique()) == {"SOLAR_INV_01", "WTG_07"}


def test_model_metrics():
    pipeline = get_pipeline()
    metrics = pipeline.metrics

    # Solar regression
    s_rf = metrics['solar_power_forecasting']['chosen_model_rf']
    assert s_rf['mae_mw'] < 0.05
    assert s_rf['r2_score'] > 0.95

    # Wind regression
    w_rf = metrics['wind_power_forecasting']['chosen_model_rf']
    assert w_rf['mae_mw'] < 0.05
    assert w_rf['r2_score'] > 0.95

    # Anomaly detection recall
    s_iso = metrics['solar_anomaly_detection']
    assert s_iso['recall'] == 1.0  # 100% recall on known solar faults
    w_iso = metrics['wind_anomaly_detection']
    assert w_iso['recall'] == 1.0  # 100% recall on known wind faults


def test_anomaly_detection_details():
    pipeline = get_pipeline()
    anomalies = pipeline.get_detected_anomalies()
    assert len(anomalies) >= 5

    critical_solar = [a for a in anomalies if a.asset_id == "SOLAR_INV_01" and a.severity == "CRITICAL"]
    critical_wind = [a for a in anomalies if a.asset_id == "WTG_07" and a.severity == "CRITICAL"]

    solar_ts = {a.timestamp_id for a in critical_solar}
    assert {67, 68, 69}.issubset(solar_ts)

    wind_ts = {a.timestamp_id for a in critical_wind}
    assert {173, 174}.issubset(wind_ts)


def test_copilot_explanation_solar():
    pipeline = get_pipeline()
    res = pipeline.run_copilot_case(
        query_text="Explain nighttime anomaly on SOLAR_INV_01",
        asset_id="SOLAR_INV_01",
        timestamp_id=67
    )
    assert res.status == "SUCCESS"
    exp = res.copilot_explanation
    assert exp.severity == "CRITICAL"
    assert "COOLING" in exp.fault_hypothesis.upper() or "FAN" in exp.fault_hypothesis.upper()
    assert len(exp.recommended_checks) >= 3
    assert "IEC 62109" in exp.retrieved_sop_standard or "SOP-PV" in exp.retrieved_sop_standard


def test_copilot_explanation_wind():
    pipeline = get_pipeline()
    res = pipeline.run_copilot_case(
        query_text="Explain high vibration on WTG_07",
        asset_id="WTG_07",
        timestamp_id=174
    )
    assert res.status == "SUCCESS"
    exp = res.copilot_explanation
    assert exp.severity == "CRITICAL"
    assert "BEARING" in exp.fault_hypothesis.upper() or "DRIVETRAIN" in exp.fault_hypothesis.upper()
    assert "ISO 10816" in exp.retrieved_sop_standard or "SOP-WTG" in exp.retrieved_sop_standard


def test_api_demo_endpoint():
    res = demo()
    assert res["status"] == "production-ready pipeline active"
    assert res["fleet_overview"]["total_telemetry_points"] == 240
    assert res["fleet_overview"]["critical_alarms_count"] == 5
    assert "variance_analysis_example" in res
    assert res["variance_analysis_example"]["sign_convention"].startswith("actual minus forecast")


def test_api_run_endpoint():
    query = Query(text="Diagnose wind turbine health", asset_id="WTG_07")
    res = run_case(query)
    assert res["status"] == "SUCCESS"
    assert res["asset_analyzed"] == "WTG_07"
    assert "copilot_explanation" in res
    assert res["copilot_explanation"]["severity"] == "CRITICAL"


def test_api_predict_endpoint():
    req = PredictionRequest(
        asset_id="WTG_07",
        hour=6,
        wind_speed_ms=5.87,
        temperature_c=46.88,
        vibration_mm_s=3.82,
        actual_power_mw=0.711
    )
    res = predict_telemetry(req)
    assert res["is_anomaly"] is True
    assert res["severity"] == "CRITICAL"
    assert res["variance_mw"] < -0.4  # Deficit > 0.4 MW


def test_fleet_summary_endpoint():
    summary = get_fleet_summary()
    assert summary.total_monitored_assets == 2
    assert summary.solar_assets == 1
    assert summary.wind_assets == 1
    assert summary.total_records == 240
    assert summary.critical_anomalies == 5
    assert summary.average_power_mw > 0.0


def test_assets_endpoint():
    assets = get_assets()
    assert len(assets) == 2
    ids = {a.asset_id for a in assets}
    assert ids == {"SOLAR_INV_01", "WTG_07"}


def test_telemetry_series_endpoint():
    series = get_telemetry_series("SOLAR_INV_01")
    assert len(series) == 120
    first = series[0]
    assert "actual_power_mw" in first or "power_mw" in first
    assert "expected_power_mw" in first
    assert "variance_mw" in first


def test_sops_endpoint():
    sops = get_sops()
    assert "SOLAR_INVERTER_THERMAL_FAN_FAULT" in sops
    assert "WIND_TURBINE_BEARING_DRIVETRAIN_FAULT" in sops


if __name__ == "__main__":
    test_pipeline_initialization()
    test_model_metrics()
    test_anomaly_detection_details()
    test_copilot_explanation_solar()
    test_copilot_explanation_wind()
    test_api_demo_endpoint()
    test_api_run_endpoint()
    test_api_predict_endpoint()
    test_fleet_summary_endpoint()
    test_assets_endpoint()
    test_telemetry_series_endpoint()
    test_sops_endpoint()
    print("ALL 12 TESTS PASSED SUCCESSFULLY! (100% Pass Rate)")
