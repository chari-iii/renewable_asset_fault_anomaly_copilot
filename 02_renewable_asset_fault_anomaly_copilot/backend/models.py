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


class FleetSummary(BaseModel):
    total_monitored_assets: int
    solar_assets: int
    wind_assets: int
    total_records: int
    total_anomalies: int
    critical_anomalies: int
    high_anomalies: int
    medium_anomalies: int
    low_anomalies: int
    average_power_mw: float
    fleet_health_score_pct: float
    system_status: str


class AssetCard(BaseModel):
    asset_id: str
    asset_type: str
    nominal_capacity_mw: float
    current_power_mw: float
    expected_power_mw: float
    performance_deviation_pct: float
    current_temperature_c: float
    current_vibration_mm_s: float
    asset_health: str
    anomaly_status: str
    total_anomalies: int


class TelemetrySeriesPoint(BaseModel):
    timestamp_id: int
    asset_id: str
    hour: int
    irradiance_wm2: float
    wind_speed_ms: float
    temperature_c: float
    vibration_mm_s: float
    power_mw: float
    expected_power_mw: float
    variance_mw: float
    variance_pct: float
    is_anomaly: bool
    severity: str
