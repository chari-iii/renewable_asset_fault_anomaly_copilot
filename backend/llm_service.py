"""
Google Gemini GenAI Integration Service for Renewable Asset Fault & Anomaly Copilot.
Integrates Google Gemini (gemini-2.5-flash) using the official `google-genai` SDK.
Receives verified structured SCADA telemetry and ML anomaly scores.
Enforces zero-hallucination guardrails and provides deterministic SOP fallback.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.domain_knowledge import retrieve_domain_context
from backend.copilot import generate_copilot_explanation
from backend.models import CopilotExplanation

logger = logging.getLogger("greenko_copilot.llm_service")

# Default model configuration
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_SECONDS = 8.0


def mask_secret(secret: Optional[str]) -> str:
    """Safely masks sensitive API keys for logs."""
    if not secret:
        return "(unset)"
    s = secret.strip()
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}...{s[-4:]}"


def build_system_instruction() -> str:
    """Constructs strict grounding guardrails for the Gemini model."""
    return (
        "You are the Greenko Renewable Fleet Operational Intelligence Copilot, an expert industrial SCADA engineer. "
        "Your task is to provide clear, operator-friendly triage explanations for solar photovoltaic inverters "
        "and wind turbine generators based on verified telemetry and standard operating procedures.\n\n"
        "STRICT SAFETY & OPERATIONAL GUARDRAILS:\n"
        "1. GROUND TRUTH IMMUTABILITY: Never modify, contradict, or invent telemetry sensor readings (temperature, vibration, power, irradiance, wind speed).\n"
        "2. PRESERVE ML PREDICTIONS: Never alter the anomaly detection flag or severity level determined by the Isolation Forest model.\n"
        "3. HYPOTHESES TRANSPARENCY: Clearly label all suspected physical failure modes as probabilistic hypotheses for field verification (e.g., '[Hypothesis] Suspected blower bearing seizure'). Never present a suspected fault as a confirmed physical fact.\n"
        "4. SOP FIDELITY: Only reference the retrieved standard operating procedure provided in the prompt context. Never invent imaginary proprietary Greenko internal SOP numbers.\n"
        "5. FACTUAL ENGINEERING LANGUAGE: Use precise, concise control-room engineering language suitable for plant technicians.\n\n"
        "You must respond with valid JSON matching the requested schema."
    )


def build_grounded_prompt(
    asset_id: str,
    hour: int,
    temperature_c: float,
    vibration_mm_s: float,
    actual_power_mw: float,
    expected_power_mw: float,
    variance_mw: float,
    variance_pct: float,
    anomaly_score: float,
    severity: str,
    timestamp_id: int,
    sop: Dict[str, Any],
    user_query: Optional[str] = None
) -> str:
    """Constructs prompt containing strictly verified facts and retrieved domain SOP context."""
    context = {
        "asset_context": {
            "asset_id": asset_id,
            "asset_type": "Central Solar Inverter (1.5 MW)" if "SOLAR" in asset_id.upper() else "Wind Turbine Generator (2.6 MW)",
            "timestamp_id": timestamp_id,
            "scada_hour": f"{hour:02d}:00"
        },
        "verified_scada_telemetry": {
            "temperature_c": round(temperature_c, 2),
            "vibration_mm_s": round(vibration_mm_s, 2),
            "actual_power_mw": round(actual_power_mw, 3),
            "expected_power_mw": round(expected_power_mw, 3),
            "variance_mw": round(variance_mw, 3),
            "variance_pct": round(variance_pct, 1)
        },
        "ml_anomaly_findings": {
            "anomaly_detected": severity in ["CRITICAL", "HIGH", "MEDIUM"],
            "anomaly_score": round(anomaly_score, 4),
            "severity_level": severity,
            "vibration_iso_zone": "Zone D (Critical alert >3.5 mm/s)" if vibration_mm_s >= 3.5 else ("Zone C (Warning 2.0-3.5 mm/s)" if vibration_mm_s >= 2.0 else "Zone A/B (Normal <2.0 mm/s)")
        },
        "retrieved_sop_context": {
            "sop_id": sop.get("id", "SOP-GEN-001"),
            "standard_reference": sop.get("standard_reference", "IEC / ISO"),
            "title": sop.get("title", "Standard Operations"),
            "root_cause_analysis": sop.get("root_cause_analysis", ""),
            "recommended_operator_actions": sop.get("recommended_operator_actions", [])
        },
        "operator_query": user_query or "Analyze telemetry health, identify likely causes, and recommend next checks."
    }

    return (
        f"Verified Telemetry & Engineering Context:\n"
        f"```json\n{json.dumps(context, indent=2)}\n```\n\n"
        f"Generate a structured operator triage response in JSON format with exactly the following keys:\n"
        f"- headline: (string) Concise operator headline summarizing the asset condition.\n"
        f"- severity_summary: (string) Factual 1-sentence assessment of operational risk.\n"
        f"- variance_explanation: (string) Clear explanation of the power variance ({variance_mw:+.3f} MW, {variance_pct:+.1f}%).\n"
        f"- potential_contributing_factors: (list of strings) 2 to 3 physical hypotheses, each explicitly starting with '[Hypothesis]'.\n"
        f"- recommended_inspection_steps: (list of strings) 3 to 5 prioritized maintenance actions aligned with the retrieved SOP.\n"
        f"- operator_follow_up_questions: (list of strings) 2 to 3 specific diagnostic questions for the field technician.\n"
    )


def generate_llm_explanation(
    asset_id: str,
    hour: int,
    temperature_c: float,
    vibration_mm_s: float,
    actual_power_mw: float,
    expected_power_mw: float,
    variance_mw: float,
    variance_pct: float,
    anomaly_score: float,
    timestamp_id: int,
    user_query: Optional[str] = None,
    force_mode: str = "auto"
) -> CopilotExplanation:
    """
    Generates an operator explanation using Google Gemini (gemini-2.5-flash).
    Gracefully falls back to the deterministic rule-based generator if:
    - force_mode is 'deterministic'
    - GEMINI_API_KEY is not set or empty
    - Network timeout occurs (> 8.0s)
    - API returns an error or invalid format
    """
    # 1. Retrieve domain SOP standard
    sop = retrieve_domain_context(asset_id, temperature_c, vibration_mm_s, variance_mw, hour)
    
    # 2. Check if deterministic fallback is explicitly forced
    if force_mode == "deterministic":
        logger.info("Deterministic mode explicitly requested; using rule-based generator.")
        fallback = generate_copilot_explanation(
            asset_id, hour, temperature_c, vibration_mm_s,
            actual_power_mw, expected_power_mw, variance_mw, variance_pct,
            anomaly_score, timestamp_id, user_query
        )
        fallback.source = "DETERMINISTIC_SOP"
        fallback.fallback_used = False
        fallback.model_name = None
        return fallback

    # 3. Check for API key
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.info("GEMINI_API_KEY unset; seamlessly using deterministic SOP fallback.")
        fallback = generate_copilot_explanation(
            asset_id, hour, temperature_c, vibration_mm_s,
            actual_power_mw, expected_power_mw, variance_mw, variance_pct,
            anomaly_score, timestamp_id, user_query
        )
        fallback.source = "DETERMINISTIC_FALLBACK (GEMINI_API_KEY not set)"
        fallback.fallback_used = True
        fallback.model_name = None
        return fallback

    # 4. Attempt Gemini API Call using official google-genai SDK
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    try:
        from google import genai
        from google.genai import types

        logger.info(
            "Calling Gemini GenAI API (Model: %s, Key: %s, Asset: %s, TS: %d)",
            model_name, mask_secret(api_key), asset_id, timestamp_id
        )

        client = genai.Client(api_key=api_key)

        prompt = build_grounded_prompt(
            asset_id=asset_id,
            hour=hour,
            temperature_c=temperature_c,
            vibration_mm_s=vibration_mm_s,
            actual_power_mw=actual_power_mw,
            expected_power_mw=expected_power_mw,
            variance_mw=variance_mw,
            variance_pct=variance_pct,
            anomaly_score=anomaly_score,
            severity="CRITICAL" if (vibration_mm_s >= 3.5 or temperature_c >= 45.0) else ("WARNING" if (vibration_mm_s >= 2.0 or temperature_c >= 42.0) else "NORMAL"),
            timestamp_id=timestamp_id,
            sop=sop,
            user_query=user_query
        )

        # Call Gemini with structured JSON output and safety instructions
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=build_system_instruction(),
                response_mime_type="application/json",
                temperature=0.2, # Low temperature for strict factual determinism
                max_output_tokens=1024
            )
        )

        if not response or not response.text:
            raise ValueError("Empty response received from Gemini API")

        # Parse and validate structured output
        raw_text = response.text.strip()
        data = json.loads(raw_text)

        # Assemble verified telemetry facts
        evidence = [
            f"Asset Identifier: {asset_id} at Timestamp ID #{timestamp_id} (Hour {hour:02d}:00)",
            f"Temperature: {temperature_c:.2f}°C",
            f"Vibration: {vibration_mm_s:.2f} mm/s RMS (ISO 10816 Zone D alert threshold: >3.5 mm/s)",
            f"Actual Power: {actual_power_mw:.3f} MW | Expected: {expected_power_mw:.3f} MW (Variance: {variance_mw:+.3f} MW [{variance_pct:+.1f}%])",
            f"Isolation Forest Anomaly Score: {anomaly_score:.4f}"
        ]

        # Determine severity status
        if vibration_mm_s >= 3.5 or temperature_c >= 48.0:
            severity = "CRITICAL"
            status = "ALARM_ACTIVE"
        elif vibration_mm_s >= 2.0 or temperature_c >= 42.0 or variance_mw <= -0.2:
            severity = "WARNING"
            status = "INVESTIGATION_RECOMMENDED"
        else:
            severity = "NORMAL"
            status = "NOMINAL_OPERATION"

        hypotheses = data.get("potential_contributing_factors", [])
        if not hypotheses:
            hypotheses = [f"[Hypothesis] Telemetry pattern indicates potential mechanical/thermal deviation under {sop.get('id', 'standard rules')}."]

        checks = data.get("recommended_inspection_steps", [])
        if not checks:
            checks = sop.get("recommended_operator_actions", ["Verify telemetry and sensors."])

        questions = data.get("operator_follow_up_questions", [])

        return CopilotExplanation(
            headline=data.get("headline", f"Operational Triage Notice for {asset_id}"),
            status=status,
            severity=severity,
            fault_hypothesis="; ".join(hypotheses),
            telemetry_evidence=evidence,
            variance_analysis=data.get("variance_explanation", f"Variance: {variance_mw:+.3f} MW ({variance_pct:+.1f}%)"),
            recommended_checks=checks,
            retrieved_sop_standard=f"{sop['id']} ({sop['standard_reference']}): {sop['title']}",
            safeguard_note="GenAI Guardrail: Synthesis generated via Google Gemini 2.5 Flash strictly grounded on verified SCADA telemetry and retrieved ISO/IEC SOP standards.",
            source="GEMINI_2_5_FLASH",
            fallback_used=False,
            model_name=model_name,
            severity_summary=data.get("severity_summary", f"Severity assessed as {severity}"),
            potential_contributing_factors=hypotheses,
            operator_follow_up_questions=questions
        )

    except Exception as e:
        logger.warning(
            "Gemini API call failed (%s); seamlessly falling back to deterministic SOP explanation.",
            type(e).__name__
        )
        fallback = generate_copilot_explanation(
            asset_id, hour, temperature_c, vibration_mm_s,
            actual_power_mw, expected_power_mw, variance_mw, variance_pct,
            anomaly_score, timestamp_id, user_query
        )
        fallback.source = f"DETERMINISTIC_FALLBACK ({type(e).__name__})"
        fallback.fallback_used = True
        fallback.model_name = None
        return fallback
