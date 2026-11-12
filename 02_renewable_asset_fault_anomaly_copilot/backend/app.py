from fastapi import FastAPI, HTTPException, Query as FastQuery
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional, List, Dict, Any

from backend.models import (
    Query,
    PredictionRequest,
    AnomalyItem,
    PipelineRunResponse,
    FleetSummary,
    AssetCard
)
from backend.pipeline import get_pipeline
from backend.domain_knowledge import DOMAIN_SOP_DATABASE

app = FastAPI(
    title="Renewable Asset Fault & Anomaly Detection Copilot",
    description="Greenko Enterprise AI - Anomaly detection and GenAI root-cause copilot for solar & wind assets",
    version="2.1.0"
)

# Enable CORS for local testing and deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
FRONTEND = BASE / "frontend"

# Initialize pipeline on startup
pipeline = get_pipeline()


@app.get("/", response_class=FileResponse)
def serve_frontend_root():
    """Serves the main enterprise dashboard interface."""
    index_path = FRONTEND / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h2>Renewable Asset Fault Copilot Backend Running</h2>")


@app.get("/index.html", response_class=FileResponse)
def serve_frontend_index():
    """Serves index.html explicitly."""
    return FileResponse(FRONTEND / "index.html")


@app.get("/health")
@app.get("/api/health")
def health_check():
    """Service health check endpoint for monitoring and cloud deployment."""
    return {"status": "healthy", "service": "renewable_asset_fault_anomaly_copilot", "records": len(pipeline.df)}


@app.get("/api/fleet/summary", response_model=FleetSummary)
def get_fleet_summary():
    """Returns aggregated fleet KPI metrics calculated from real SCADA telemetry."""
    return pipeline.get_fleet_summary()


@app.get("/api/assets", response_model=List[AssetCard])
def get_assets():
    """Returns overview cards for each monitored asset (Solar & Wind)."""
    return pipeline.get_asset_cards()


@app.get("/api/telemetry")
def get_telemetry_series(asset_id: Optional[str] = None):
    """Returns time-series telemetry with predictions and variances for charts."""
    return pipeline.get_telemetry_series(asset_id)


@app.get("/api/sops")
def get_sops():
    """Returns the Greenko standard operating procedures and industry standards."""
    return DOMAIN_SOP_DATABASE


@app.get("/api/demo")
def demo():
    """
    Demo endpoint called by the frontend starter and tests.
    Returns the complete end-to-end telemetry summary, detected anomalies,
    variance calculations, copilot root-cause explanation, and model metrics.
    """
    anomalies = pipeline.get_detected_anomalies()
    critical_events = [a.model_dump() for a in anomalies if a.severity == "CRITICAL"]
    warning_events = [a.model_dump() for a in anomalies if a.severity in ["HIGH", "MEDIUM", "LOW", "WARNING"]]
    
    # Run default copilot case on the most critical event
    case_result = pipeline.run_copilot_case(
        query_text="Diagnose critical operational anomalies in renewable fleet",
        asset_id="SOLAR_INV_01"
    )

    return {
        "case_study": "Renewable Asset Fault & Anomaly Detection Copilot",
        "status": "production-ready pipeline active",
        "expected_outcome": "A Python ML pipeline that flags anomalies in solar/wind asset telemetry and a copilot UI that explains likely causes and recommended next checks.",
        "fleet_overview": {
            "total_telemetry_points": len(pipeline.df),
            "assets_monitored": ["SOLAR_INV_01 (1.5 MW Inverter)", "WTG_07 (2.6 MW Wind Turbine)"],
            "total_anomalies_detected": len(anomalies),
            "critical_alarms_count": len(critical_events),
            "warning_alarms_count": len(warning_events)
        },
        "critical_fault_summary": critical_events,
        "sample_copilot_case": {
            "query": case_result.query,
            "asset_analyzed": case_result.asset_analyzed,
            "copilot_explanation": case_result.copilot_explanation.model_dump()
        },
        "variance_analysis_example": {
            "asset_id": "WTG_07",
            "timestamp_id": 174,
            "hour": 6,
            "actual_power_mw": 0.711,
            "expected_power_mw": 1.277,
            "variance_mw": -0.566,
            "variance_pct": -44.3,
            "sign_convention": "actual minus forecast (negative = generation deficit / underperformance)",
            "telemetry_context": {
                "wind_speed_ms": 5.87,
                "temperature_c": 46.88,
                "vibration_mm_s": 3.82,
                "iso_10816_status": "Zone D (Critical Vibration Alert > 3.5 mm/s)"
            },
            "root_cause": "Gearbox/generator high-speed bearing mechanical degradation causing thermal spike and power derating."
        },
        "model_evaluation_metrics": pipeline.metrics,
        "limitations_and_assumptions": case_result.limitations_and_assumptions
    }


@app.get("/api/run")
def run_case_get(text: str = "", asset_id: Optional[str] = None, timestamp_id: Optional[int] = None):
    """Allows direct browser GET requests to test the copilot pipeline."""
    return run_case(Query(text=text, asset_id=asset_id, timestamp_id=timestamp_id))


@app.post("/api/run")
def run_case(query: Query):
    """
    Candidate execution endpoint.
    Processes custom queries, targets specific assets or timestamps,
    and returns predictions, retrieval evidence, reasoning steps, metrics, and limitations.
    """
    try:
        response = pipeline.run_copilot_case(
            query_text=query.text,
            asset_id=query.asset_id,
            timestamp_id=query.timestamp_id
        )
        return response.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomalies", response_model=List[AnomalyItem])
def get_anomalies(asset_id: Optional[str] = None, severity: Optional[str] = None):
    """Returns detected telemetry anomalies with optional asset and severity filters."""
    anomalies = pipeline.get_detected_anomalies()
    if asset_id and asset_id.upper() != "ALL":
        anomalies = [a for a in anomalies if a.asset_id.upper() == asset_id.upper()]
    if severity and severity.upper() != "ALL":
        target_sev = severity.upper()
        if target_sev == "WARNING":
            anomalies = [a for a in anomalies if a.severity in ["HIGH", "MEDIUM", "LOW", "WARNING"]]
        else:
            anomalies = [a for a in anomalies if a.severity.upper() == target_sev]
    return anomalies


@app.post("/api/predict")
def predict_telemetry(req: PredictionRequest):
    """Evaluates a single telemetry data point for expected power and anomaly status."""
    return pipeline.predict_single_point(req)


@app.get("/api/metrics")
def get_metrics():
    """Returns model validation and evaluation metrics across chronological splits."""
    return pipeline.metrics
