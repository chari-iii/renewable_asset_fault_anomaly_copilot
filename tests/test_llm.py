"""
Unit Tests for Google Gemini GenAI Integration & Deterministic Fallback.
All tests use mock fixtures and deterministic fallbacks.
NO paid API calls are performed.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from backend.llm_service import (
    generate_llm_explanation,
    mask_secret,
    build_grounded_prompt,
    build_system_instruction
)
from backend.models import Query
from backend.pipeline import get_pipeline


def test_mask_secret():
    """Verifies that secrets are masked safely for logs."""
    assert mask_secret(None) == "(unset)"
    assert mask_secret("") == "(unset)"
    assert mask_secret("short") == "***"
    assert mask_secret("AIzaSyAbcdef123456789") == "AIza...6789"


def test_gemini_fallback_when_api_key_unset(monkeypatch):
    """Verifies that if GEMINI_API_KEY is unset, deterministic SOP fallback is returned."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    res = generate_llm_explanation(
        asset_id="WTG_07",
        hour=6,
        temperature_c=46.88,
        vibration_mm_s=3.82,
        actual_power_mw=0.711,
        expected_power_mw=1.277,
        variance_mw=-0.566,
        variance_pct=-44.3,
        anomaly_score=0.174,
        timestamp_id=174
    )
    
    assert res.fallback_used is True
    assert "DETERMINISTIC_FALLBACK" in res.source
    assert res.severity == "CRITICAL"
    assert "ISO 10816" in res.retrieved_sop_standard or "SOP-WTG" in res.retrieved_sop_standard
    assert len(res.recommended_checks) >= 3


def test_gemini_forced_deterministic_mode(monkeypatch):
    """Verifies that if force_mode='deterministic', fallback is immediately returned even if key exists."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake_mock_key_12345")
    
    res = generate_llm_explanation(
        asset_id="SOLAR_INV_01",
        hour=19,
        temperature_c=53.48,
        vibration_mm_s=3.86,
        actual_power_mw=0.0,
        expected_power_mw=0.004,
        variance_mw=-0.004,
        variance_pct=0.0,
        anomaly_score=0.1581,
        timestamp_id=67,
        force_mode="deterministic"
    )
    
    assert res.source == "DETERMINISTIC_SOP"
    assert res.fallback_used is False
    assert res.severity == "CRITICAL"
    assert "IEC 62109" in res.retrieved_sop_standard or "SOP-PV" in res.retrieved_sop_standard


def test_gemini_successful_mocked_llm_response(monkeypatch):
    """Verifies complete end-to-end processing of a mocked Gemini 2.5 Flash structured response."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key_test_6789")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    mock_llm_output = {
        "headline": "CRITICAL DRIVETRAIN DEGRADATION on WTG_07: High Vibration and Thermal Spike",
        "severity_summary": "Immediate operational risk of high-speed shaft bearing lockup under continuous generation.",
        "variance_explanation": "Actual power generation of 0.711 MW is 0.566 MW (-44.3%) below aerodynamic expectation.",
        "potential_contributing_factors": [
            "[Hypothesis] Generator drive-end bearing spalling causing localized mechanical friction.",
            "[Hypothesis] High-speed shaft coupling misalignment leading to excessive radial displacement.",
            "[Hypothesis] Lubrication line blockage starving bearing raceway."
        ],
        "recommended_inspection_steps": [
            "1. Issue temporary power curtailment or derating.",
            "2. Execute Lock-Out/Tag-Out (LOTO) and inspect nacelle drivetrain.",
            "3. Collect grease/oil ferrography sample to detect metal wear particulates.",
            "4. Perform high-frequency vibration FFT spectral analysis for BPFI/BPFO defect frequencies."
        ],
        "operator_follow_up_questions": [
            "When was the automatic lubrication reservoir last replenished?",
            "Has there been a recent grid curtailment or sudden pitch emergency stop event?",
            "Are there audible acoustic knocking sounds inside the nacelle at rated speed?"
        ]
    }

    mock_response = MagicMock()
    mock_response.text = json.dumps(mock_llm_output)

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client_instance):
        res = generate_llm_explanation(
            asset_id="WTG_07",
            hour=6,
            temperature_c=46.88,
            vibration_mm_s=3.82,
            actual_power_mw=0.711,
            expected_power_mw=1.277,
            variance_mw=-0.566,
            variance_pct=-44.3,
            anomaly_score=0.174,
            timestamp_id=174
        )

        assert res.source == "GEMINI_2_5_FLASH"
        assert res.fallback_used is False
        assert res.model_name == "gemini-2.5-flash"
        assert "WTG_07" in res.headline
        assert len(res.potential_contributing_factors) == 3
        for factor in res.potential_contributing_factors:
            assert factor.startswith("[Hypothesis]")
        assert len(res.recommended_checks) >= 3
        assert len(res.operator_follow_up_questions) == 3
        assert "ISO 10816" in res.retrieved_sop_standard or "SOP-WTG" in res.retrieved_sop_standard


def test_gemini_api_timeout_triggers_fallback(monkeypatch):
    """Verifies that if Gemini API times out or raises an exception, fallback is returned seamlessly."""
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key_timeout_test")

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = TimeoutError("Simulated 8.0s API Timeout")

    with patch("google.genai.Client", return_value=mock_client_instance):
        res = generate_llm_explanation(
            asset_id="SOLAR_INV_01",
            hour=19,
            temperature_c=53.48,
            vibration_mm_s=3.86,
            actual_power_mw=0.0,
            expected_power_mw=0.004,
            variance_mw=-0.004,
            variance_pct=0.0,
            anomaly_score=0.1581,
            timestamp_id=67
        )

        assert res.fallback_used is True
        assert "TimeoutError" in res.source
        assert res.severity == "CRITICAL"
        assert "IEC 62109" in res.retrieved_sop_standard or "SOP-PV" in res.retrieved_sop_standard


def test_immutable_ml_ground_truth(monkeypatch):
    """Verifies that the LLM service never alters sensor readings or ML anomaly scores."""
    pipeline = get_pipeline()
    case = pipeline.run_copilot_case(
        query_text="Diagnose WTG_07",
        asset_id="WTG_07",
        timestamp_id=174,
        mode="deterministic"
    )

    exp = case.copilot_explanation
    # Evidence must contain exact verified numerical values
    evidence_text = " ".join(exp.telemetry_evidence)
    assert "46.88°C" in evidence_text
    assert "3.82 mm/s" in evidence_text
    assert "0.711 MW" in evidence_text
    assert "0.174" in evidence_text


def test_copilot_explain_endpoints():
    """Tests the /api/copilot/explain GET and POST endpoints."""
    from backend.app import copilot_explain_get, copilot_explain_post
    
    # GET
    get_res = copilot_explain_get(asset_id="WTG_07", timestamp_id=174, mode="deterministic")
    assert get_res["status"] == "SUCCESS"
    assert get_res["asset_analyzed"] == "WTG_07"
    assert "copilot_explanation" in get_res
    assert get_res["copilot_explanation"]["severity"] == "CRITICAL"

    # POST
    post_res = copilot_explain_post(Query(text="Inspect Solar TS 67", asset_id="SOLAR_INV_01", timestamp_id=67, mode="deterministic"))
    assert post_res["status"] == "SUCCESS"
    assert post_res["asset_analyzed"] == "SOLAR_INV_01"
    assert post_res["copilot_explanation"]["severity"] == "CRITICAL"
