from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class TelemetryPoint(BaseModel):
    timestamp_id: int
    asset_id: str
    hour: int
    irradiance_wm2: float
    wind_speed_ms: float
    temperature_c: float
    vibration_mm_s: float
    power_mw: float


class Query(BaseModel):
    text: str = Field(default="", description="Natural language prompt or query for the copilot")
    asset_id: Optional[str] = Field(default=None, description="Optional target asset ID (SOLAR_INV_01 or WTG_07)")
    timestamp_id: Optional[int] = Field(default=None, description="Optional specific timestamp ID to inspect")


class PredictionRequest(BaseModel):
    asset_id: str
    hour: int
    irradiance_wm2: float = 0.0
    wind_speed_ms: float = 0.0
    temperature_c: float = 30.0
    vibration_mm_s: float = 1.3
    actual_power_mw: Optional[float] = None


class AnomalyItem(BaseModel):
    timestamp_id: int
    asset_id: str
    hour: int
    temperature_c: float
    vibration_mm_s: float
    power_mw: float
    expected_power_mw: float
    variance_mw: float
    variance_pct: float
    anomaly_score: float
    severity: str
    fault_type: str


class CopilotExplanation(BaseModel):
    headline: str
    status: str
    severity: str
    fault_hypothesis: str
    telemetry_evidence: List[str]
    variance_analysis: str
    recommended_checks: List[str]
    retrieved_sop_standard: str
    safeguard_note: str


class PipelineRunResponse(BaseModel):
    query: str
    status: str
    asset_analyzed: str
    total_telemetry_points: int
    anomalies_detected: int
    anomaly_summary: List[AnomalyItem]
    copilot_explanation: CopilotExplanation
    evaluation_metrics: Dict[str, Any]
    limitations_and_assumptions: List[str]
