"""
Renewable Energy Domain Knowledge Base & Illustrative Standard Operating Procedures (SOPs).
DISCLAIMER: The SOP entries below (SOP-PV-042, SOP-WTG-089, etc.) are illustrative operational
knowledge templates synthesized from public international standards (IEC 62109-1, ISO 10816-21,
IEC 61724-1, IEC 61400-1) for this case study, rather than official Greenko internal proprietary documents.
"""

from typing import Dict, Any, List

DOMAIN_SOP_DATABASE: Dict[str, Dict[str, Any]] = {
    "NOMINAL_OPERATION": {
        "id": "SOP-GEN-001",
        "title": "Nominal Asset Operation & Routine Preventative Monitoring",
        "standard_reference": "IEC 61724-1 / IEC 61400-1 (Nominal Operating Parameters)",
        "applicable_asset_type": "ALL",
        "trigger_conditions": {
            "operational_state": "All telemetry parameters within normal operational limits"
        },
        "root_cause_analysis": (
            "No active physical anomaly or performance deficit detected. "
            "Asset is operating within nominal baseline parameters."
        ),
        "impact_and_risk": "Zero elevated risk. Continuous normal power generation.",
        "recommended_operator_actions": [
            "1. Maintain standard scheduled preventative maintenance intervals.",
            "2. Continue automated SCADA monitoring and alarm surveillance.",
            "3. Log routine operational parameters in plant shift diary."
        ]
    },
    "SOLAR_INVERTER_THERMAL_FAN_FAULT": {
        "id": "SOP-PV-042",
        "title": "Solar Inverter Cooling System & Thermal Runaway Mitigation",
        "standard_reference": "IEC 62109-1 / IEC 61724 (Photovoltaic System Performance)",
        "applicable_asset_type": "SOLAR_INV_01",
        "trigger_conditions": {
            "temperature_threshold_c": 45.0,
            "vibration_threshold_mm_s": 2.5,
            "operational_state": "Zero or low irradiance during evening/night cooling phase"
        },
        "root_cause_analysis": (
            "During non-generating evening hours (irradiance = 0 W/m²), inverter temperature should "
            "decay toward ambient (<30°C). Simultaneous temperature spike (>46°C) and elevated "
            "vibration (>3.5 mm/s) indicates forced-air cooling fan bearing failure, blower motor "
            "stalling/mechanical rub, clogged intake air filters, or auxiliary cooling transformer resonance."
        ),
        "impact_and_risk": (
            "Risk of IGBT semiconductor thermal breakdown, DC link bus capacitor degradation, "
            "or inverter fire hazard. Can lead to prolonged morning outage and lost peak generation."
        ),
        "recommended_operator_actions": [
            "1. Dispatch site electrical technician with handheld FLIR thermal camera to inspect inverter enclosure.",
            "2. Verify status of forced-air cooling fan circuit breakers and blower motor contactors.",
            "3. Inspect cooling fan blades for physical deformation, foreign debris, or bearing seizure.",
            "4. Inspect and clean external air intake sand/dust filters.",
            "5. Review internal SCADA alarms for IGBT heatsink over-temperature and DC link capacitor telemetry."
        ]
    },
    "WIND_TURBINE_BEARING_DRIVETRAIN_FAULT": {
        "id": "SOP-WTG-089",
        "title": "Wind Turbine Drivetrain & Bearing Vibration Alert Procedure",
        "standard_reference": "ISO 10816-21 (Mechanical Vibration in Wind Turbines) / IEC 61400-1",
        "applicable_asset_type": "WTG_07",
        "trigger_conditions": {
            "temperature_threshold_c": 43.0,
            "vibration_threshold_mm_s": 3.5,
            "operational_state": "Continuous generation during low-to-medium wind (5-6 m/s)"
        },
        "root_cause_analysis": (
            "Under ISO 10816-21, drivetrain RMS vibration exceeding 3.5 mm/s falls into Zone D "
            "(unacceptable vibration; imminent mechanical distress). Combined with elevated bearing "
            "temperature (>45°C) during cool morning hours, this signature indicates generator "
            "drive-end (DE) or non-drive-end (NDE) bearing spalling, gearbox high-speed shaft (HSS) "
            "bearing distress, lubrication starvation, or high-speed coupling misalignment."
        ),
        "impact_and_risk": (
            "Catastrophic bearing seizure leading to gearbox failure, generator rotor eccentricity, "
            "drivetrain lockup, and months of unscheduled downtime with high crane replacement costs."
        ),
        "recommended_operator_actions": [
            "1. Issue temporary power curtailment or derating if vibration exceeds 3.5 mm/s continuously.",
            "2. Lock-out/Tag-out (LOTO) turbine nacelle and execute visual and acoustic drivetrain inspection.",
            "3. Perform high-frequency vibration FFT spectral analysis (check for BPFO/BPFI defect harmonics).",
            "4. Check automatic lubrication system oil pressure, reservoir levels, and line blockages.",
            "5. Extract oil/grease samples from gearbox and generator bearings for ferrography wear-particle test.",
            "6. Check flexible coupling elastomeric elements and verify shaft laser alignment."
        ]
    },
    "SOLAR_UNDERPERFORMANCE_CURTAILMENT": {
        "id": "SOP-PV-018",
        "title": "Solar Generation Deficit & Grid Curtailment Triage",
        "standard_reference": "IEC 61724-1 Class A Performance Ratio Monitoring",
        "applicable_asset_type": "SOLAR_INV_01",
        "trigger_conditions": {
            "variance_mw_deficit": -0.2,
            "irradiance_threshold_wm2": 300.0
        },
        "root_cause_analysis": (
            "Actual generation significantly below expected output given current solar irradiance. "
            "Potential causes include DC combiner box fuse trip, inverter maximum power point tracking "
            "(MPPT) clipping, grid curtailment order, or heavy soiling on photovoltaic strings."
        ),
        "impact_and_risk": "Direct commercial revenue loss and failure to meet PPA contractual generation targets.",
        "recommended_operator_actions": [
            "1. Cross-check grid dispatch instruction log for automated utility curtailment.",
            "2. Check inverter DC input string currents to detect blown fuses or disconnected strings.",
            "3. Inspect pyranometer cleanliness and verify irradiance calibration."
        ]
    },
    "WIND_TURBINE_AERODYNAMIC_UNDERPERFORMANCE": {
        "id": "SOP-WTG-012",
        "title": "Wind Turbine Power Curve Deficit Investigation",
        "standard_reference": "IEC 61400-12-1 (Power Performance Measurements)",
        "applicable_asset_type": "WTG_07",
        "trigger_conditions": {
            "variance_mw_deficit": -0.3,
            "wind_speed_threshold_ms": 5.0
        },
        "root_cause_analysis": (
            "Turbine generating below nominal power curve for prevailing wind velocity. "
            "Typically caused by blade pitch calibration drift, anemometer calibration error, "
            "yaw misalignment (wind direction tracking lag), or mechanical brake drag."
        ),
        "impact_and_risk": "Reduced annual energy production (AEP) and excessive asymmetric blade fatigue loads.",
        "recommended_operator_actions": [
            "1. Check turbine yaw error and wind vane alignment relative to nacelle heading.",
            "2. Verify blade pitch angle sensor calibration and hydraulic/electric pitch drive response.",
            "3. Inspect rotor blades for leading-edge erosion or aerodynamic contamination."
        ]
    }
}


def retrieve_domain_context(asset_id: str, temperature_c: float, vibration_mm_s: float, variance_mw: float, hour: int) -> Dict[str, Any]:
    """
    RAG Domain Knowledge Retriever:
    Matches operational telemetry anomalies with industry standards (ISO 10816, IEC)
    and returns relevant Standard Operating Procedures (SOPs).
    Returns nominal baseline procedure when no anomaly thresholds are breached.
    """
    if "SOLAR" in asset_id.upper():
        if temperature_c > 44.0 or vibration_mm_s > 2.5:
            return DOMAIN_SOP_DATABASE["SOLAR_INVERTER_THERMAL_FAN_FAULT"]
        elif variance_mw < -0.15:
            return DOMAIN_SOP_DATABASE["SOLAR_UNDERPERFORMANCE_CURTAILMENT"]
        else:
            return DOMAIN_SOP_DATABASE["NOMINAL_OPERATION"]
    else:  # Wind WTG
        if vibration_mm_s > 2.5 or temperature_c > 42.0:
            return DOMAIN_SOP_DATABASE["WIND_TURBINE_BEARING_DRIVETRAIN_FAULT"]
        elif variance_mw < -0.2:
            return DOMAIN_SOP_DATABASE["WIND_TURBINE_AERODYNAMIC_UNDERPERFORMANCE"]
        else:
            return DOMAIN_SOP_DATABASE["NOMINAL_OPERATION"]
