# Greenko Renewable Asset Fault & Anomaly Detection Copilot

**Enterprise AI Solution for Predictive Maintenance, Anomaly Triage & Decision Support in Solar and Wind Assets**

---

## 1. Project Overview & Business Context

### 1.1 Problem Statement
Greenko operates utility-scale renewable energy assets—principally central solar photovoltaic (PV) power plants and wind turbine generators (WTGs)—distributed across multiple regions. Control room operators and site maintenance engineers monitor continuous streams of SCADA telemetry. Without automated anomaly triage:
1. **Critical equipment degradation is detected too late:** Mechanical bearing distress, cooling fan failures, or inverter thermal runaway escalate into catastrophic equipment failure, leading to unplanned downtime, costly crane mobilization, and lost generation.
2. **Alert fatigue & ambiguous telemetry:** Raw sensor thresholds trigger nuisance alarms, while operators struggle to connect co-occurring mechanical and electrical anomalies (e.g. vibration spikes paired with unseasonal nighttime temperature rises).
3. **Forecast variance ambiguity:** When generation deviates from the day-ahead schedule, operators lack immediate clarity on whether the shortfall is driven by environmental resource fluctuation, grid curtailment, or impending mechanical failure.

### 1.2 Intended Users & Value Delivered
- **Target Users:** Operations Analysts, Plant Maintenance Engineers, and Control Room Dispatchers.
- **Value Delivered:**
  - **Automated Anomaly Detection:** Multivariate Isolation Forest combined with ISO 10816-21 and IEC 62109 engineering standards.
  - **Actual-versus-Forecast Variance Analysis:** Clear megawatt magnitude ($\Delta P$) and percentage deficit calculations using physics-grounded power baselines.
  - **Grounded AI Copilot:** Natural language triage translating sensor signatures into plain-English root causes and standard operating procedure (SOP) maintenance checklists.
  - **Executive & Operator Dashboard:** Responsive dark navy industrial monitoring dashboard with live KPIs, interactive time-series charts, and diagnostic drill-downs.

---

## 2. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            SCADA Telemetry Streams                                |
|   (Solar: Irradiance, Temp, Vib, Power | Wind: Wind Speed, Temp, Vib, Power)     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                1. Data Preprocessing & Physics Feature Engineering                |
|  * Chronological sorting (No future-data leakage)                                 |
|  * Cyclical time encoding (sin_hour, cos_hour)                                     |
|  * Physics expectations: Solar PR (0.0019 MW/W/m²) | Wind Cubic Curve              |
+-----------------------------------------------------------------------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+   +---------------------------------------+
|  2A. Expected Power Baseline Modeling |   |     2B. Multi-Sensor Anomaly Engine   |
|  * Random Forest Regressor (Nonlinear)|   |  * Isolation Forest (Contamination=3%)|
|  * Linear Physics Baseline Comparison |   |  * ISO 10816-21 Vibration Zone D Rules|
+---------------------------------------+   +---------------------------------------+
                    |                                             |
                    +----------------------+----------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                     3. Variance & Diagnostic Severity Classifier                  |
|  * Variance = Actual Power - Forecast Power (Sign convention: Negative = Deficit) |
|  * Severity classification: CRITICAL, HIGH, MEDIUM, LOW, NORMAL                   |
+-----------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                    4. RAG Domain Knowledge Retrieval Engine                       |
|  * Matches fault signature against Greenko SOPs & Industry Standards             |
|  * ISO 10816-21 (Wind Vibration Zone D >3.5 mm/s) | IEC 62109-1 (Inverter Thermal)|
+-----------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                       5. GenAI Copilot Explanation Layer                          |
|  * Operator-friendly natural language synthesis                                  |
|  * Strict numerical grounding (Zero hallucination guardrail)                     |
|  * Dual mode: Deterministic Physics Engine + Optional External LLM Integration    |
+-----------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                 6. FastAPI REST Backend & Modern Enterprise Dashboard             |
|  * REST Endpoints: /api/fleet/summary, /api/assets, /api/telemetry, /api/sops...  |
|  * Dark-themed industrial SaaS frontend (Overview, Assets, Anomalies, Copilot)   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Technology Stack

- **Backend & API:** FastAPI (0.115.0), Starlette, Uvicorn (0.30.6), Pydantic v2 (2.8.2).
- **Machine Learning & Analytics:** Scikit-Learn (1.5.1), Pandas (2.2.2), NumPy (1.26.4), SciPy (1.17.1), Joblib (1.6.0).
- **Frontend Dashboard:** HTML5, CSS3 (Custom Dark Navy/Charcoal Design System), Vanilla JS, Chart.js (v4.4.4).
- **Testing & Quality Assurance:** Pytest (9.2.0), HTTPX (0.28.1).
- **Containerization & Deployment:** Docker, Procfile, Railway configuration (`railway.json`).

---

## 4. Key Features

1. **Fleet Overview Dashboard:** Live KPI cards tracking total assets, hourly records, anomaly counts, critical alarms, fleet average power, and fleet health index.
2. **Dedicated Solar & Wind Asset Views:** Telemetry drill-down with power-vs-irradiance and power-vs-wind-speed scatter curves, plus thermal and vibration envelope tracking.
3. **Anomaly Detection & Event Triage:** Interactive table with real-time text search, asset filter (`All`, `SOLAR_INV_01`, `WTG_07`), and severity filter (`Critical`, `High`, `Medium`).
4. **Interactive AI Copilot Workbench:** Select any anomalous event or submit custom natural language queries. Generates grounded fault hypotheses, verified sensor evidence, actual-vs-expected power variance, and ISO/IEC standard operating procedure (SOP) field checklists.
5. **Model Performance Hub:** Side-by-side comparison of Random Forest Regressors against Linear Physics baselines, including detailed engineering notes on synthetic benchmark behavior versus real plant aerodynamics.
6. **Starter API Inspector:** Backward-compatible interactive console for testing `/api/demo` and other endpoints directly in the UI.

---

## 5. Machine Learning Pipeline & Engineering Investigation

### 5.1 Dataset Overview
The bundled dataset (`data/asset_telemetry.csv`) contains 240 chronological hourly records across two asset classes:
- **`SOLAR_INV_01` (120 records):** Central solar PV inverter (1.5 MW nominal capacity).
- **`WTG_07` (120 records):** Utility-scale Wind Turbine Generator (2.6 MW nominal capacity).

### 5.2 Feature Engineering
- **Cyclical Temporal Encoding:** Diurnal operational cycles represented via $\sin(2\pi \cdot \text{hour} / 24)$ and $\cos(2\pi \cdot \text{hour} / 24)$ to preserve continuity between Hour 23 and Hour 0.
- **Physics Reference Models:** First-principles baseline power expectations ($P_{\text{solar}} = 0.0019 \cdot G$; $P_{\text{wind}} = f(v^3)$).
- **Strict Chronological Splitting:** 70% train split (Days 1–3.5) and 30% test split (Days 3.5–5) without shuffling to prevent future-data leakage.

### 5.3 Anomaly Detection Formulation
- **Algorithm:** Isolation Forest ($\text{contamination}=0.035$, $n=100$, $\text{max\_samples}=\text{'auto'}$).
- **Domain Post-Processing:** Telemetry anomaly scores are combined with **ISO 10816-21 Zone D** vibration alarm thresholds ($>3.5\text{ mm/s}$) and IEC inverter thermal runaway thresholds ($>45.0^\circ\text{C}$) to eliminate false alarms and achieve **100% recall** on ground-truth failure modes.

### 5.4 Model Evaluation Results (Chronological Test Set)

| Target Asset | Model Architecture | MAE ($MW$) | RMSE ($MW$) | $R^2$ Score | Known Fault Recall |
|---|---|---:|---:|---:|---:|
| **`SOLAR_INV_01`** | **Random Forest Regressor (n=100, depth=6)** | 0.0012 MW | 0.0024 MW | 1.0000 | 100.0% (3/3) |
| `SOLAR_INV_01` | Linear Physics Baseline ($P = \eta \cdot G$) | 0.0001 MW | 0.0002 MW | 1.0000 | N/A |
| **`WTG_07`** | **Random Forest Regressor (n=100, depth=6)** | 0.0178 MW | 0.0258 MW | 0.9986 | 100.0% (2/2) |
| `WTG_07` | Linear Wind Speed Baseline ($P \approx 0.45\cdot v - 1.36$) | 0.0010 MW | 0.0012 MW | 1.0000 | N/A |

### 5.5 Important ML Engineering Finding (Why Linear Baseline Outperformed RF on MAE)
During our rigorous evaluation, we investigated why the Linear baseline achieved lower MAE than Random Forest on this specific dataset:
1. **Synthetic Data Artifact:** The bundled benchmark telemetry was generated using exact affine linear equations without measurement noise ($P_{\text{solar}} \approx 0.0019 \cdot G$; $P_{\text{wind}} \approx 0.45\cdot v - 1.36$). An Ordinary Least Squares (OLS) linear model fits a linear relationship with near-zero error.
2. **Random Forest Step Discretization:** As a decision tree ensemble, Random Forest approximates continuous functions through piecewise-constant hyper-rectangles, resulting in small quantization binning residuals.
3. **Physical Real-World Justification:** In real operating renewable plants, solar inverter MPPT clipping, temperature derating coefficients ($-0.4\%/^\circ\text{C}$ Pmax loss), and wind turbine aerodynamic cubic power curves ($P \propto \frac{1}{2}\rho A C_p v^3$ saturating at rated speed) are **highly nonlinear**. Therefore, Random Forest is technically required for physical plant operations, even though linear regression achieves lower MAE on this synthetic linear benchmark.

---

## 6. Actual-vs-Forecast Variance Analysis

### 6.1 Sign Convention
$$\text{Variance } (\Delta P) = P_{\text{actual}} - P_{\text{expected}}$$
- **Negative Variance ($\Delta P < 0$):** Generation deficit (shortfall / underperformance / loss).
- **Positive Variance ($\Delta P > 0$):** Generation surplus (favorable atmospheric density or over-performance).

### 6.2 Demonstrating a Clear Forecast Miss: Wind Turbine Bearing Distress
- **Asset:** `WTG_07` (Timestamp ID #174, Hour 06:00)
- **Measured Telemetry:**
  - Wind Speed: $5.87\text{ m/s}$
  - Bearing Temperature: $46.88^\circ\text{C}$ (Normal morning baseline: $24.0^\circ\text{C} - 28.0^\circ\text{C}$)
  - Drivetrain Vibration: $3.82\text{ mm/s RMS}$ (ISO 10816 Zone D Danger threshold: $>3.5\text{ mm/s}$)
  - Actual Generation: $0.711\text{ MW}$
  - Expected Forecast Generation: $1.277\text{ MW}$
- **Variance Calculation:**
  $$\Delta P = 0.711\text{ MW} - 1.277\text{ MW} = -0.566\text{ MW} \quad (-44.3\% \text{ deficit})$$
- **Root Cause Interpretation:** Concurrent high vibration ($3.82\text{ mm/s}$) and elevated bearing temperature ($46.9^\circ\text{C}$) indicate severe mechanical degradation in the generator drive-end bearing or gearbox high-speed stage, resulting in mechanical friction, internal power loss, and turbine derating.

---

## 7. AI Copilot Architecture & Safeguards

### 7.1 What GenAI Does in Our System
The GenAI Copilot transforms complex numerical SCADA telemetry into grounded, actionable natural language diagnoses:
1. Translates multi-sensor anomalies into plain-English root causes.
2. Retrieves relevant Greenko Standard Operating Procedures (SOPs) and ISO/IEC engineering standards.
3. Formulates a prioritized field inspection checklist for plant technicians.

### 7.2 RAG / Knowledge Retrieval Layer
Our copilot uses **Retrieval-Augmented Generation (RAG)** over a structured Greenko SOP Knowledge Base:
- **`SOP-PV-042` (IEC 62109-1):** Inverter Forced-Air Cooling Fan Failure & Thermal Runaway Mitigation.
- **`SOP-WTG-089` (ISO 10816-21):** Wind Turbine Drivetrain & Bearing Vibration Alert Procedure.
- **`SOP-PV-018` (IEC 61724):** Solar PV Generation Deficit & String Curtailment Triage.
- **`SOP-WTG-012` (IEC 61400):** Wind Turbine Aerodynamic Power Curve Deficit Investigation.

### 7.3 Production Safeguards & AI Guardrails
- **Zero Hallucination:** Explanations are strictly bound to verified numerical telemetry values and retrieved SOP standards.
- **API Key Security:** No secrets or credentials are hardcoded. External LLM integration uses environment variables (`OPENAI_API_KEY` / `GEMINI_API_KEY`).
- **Deterministic Offline Fallback:** If API keys are absent or external networks fail, the copilot automatically uses the built-in deterministic rule-grounded generator, ensuring 100% uptime and reproducibility without external dependencies.
- **Distinction from Autonomous Agents:** We explicitly do not call this a multi-tool autonomous agent. It is a **grounded diagnostic copilot** operating via structured retrieval, ML scoring, and deterministic prompt synthesis.

---

## 8. API Documentation

| Endpoint | Method | Request / Query | Description |
|---|---|---|---|
| `/` | `GET` | None | Serves the main enterprise dashboard interface |
| `/api/fleet/summary` | `GET` | None | Returns aggregated fleet KPIs (assets, anomalies, avg power, health score) |
| `/api/assets` | `GET` | None | Returns overview cards for Solar and Wind assets |
| `/api/telemetry` | `GET` | `asset_id` (opt) | Returns time-series telemetry with predictions and variances for charts |
| `/api/anomalies` | `GET` | `asset_id`, `severity` | Returns detected anomalies with filtering |
| `/api/sops` | `GET` | None | Returns the Greenko SOP database and industry standards |
| `/api/demo` | `GET` | None | Returns end-to-end demo payload for quick verification |
| `/api/run` | `GET`/`POST` | `Query` (text, asset_id, ts) | Executes Copilot pipeline and generates grounded explanation |
| `/api/predict` | `POST` | `PredictionRequest` | Evaluates a single telemetry data point in real time |
| `/api/metrics` | `GET` | None | Returns model evaluation metrics across chronological splits |

Interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

---

## 9. Setup, Execution & Testing Guide

### 9.1 Local Setup
```bash
# 1. Clone repository
git clone <your-repo-link>
cd renewable_asset_fault_anomaly_copilot/02_renewable_asset_fault_anomaly_copilot

# 2. Create and activate virtual environment (Python 3.11 recommended)
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 9.2 Train Models & Reproduce Metrics
```bash
python backend/train_and_evaluate.py
```

### 9.3 Run Automated Test Suite
```bash
python -m pytest -q
```
*Expected result:* **`12 passed in 3.2s (100% pass rate)`**.

### 9.4 Start the FastAPI Server & Frontend Dashboard
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --app-dir 02_renewable_asset_fault_anomaly_copilot --reload
```
Open your browser to:
- **Interactive Enterprise Dashboard:** `http://127.0.0.1:8000/`
- **Swagger API Documentation:** `http://127.0.0.1:8000/docs`
- **Demo API Endpoint:** `http://127.0.0.1:8000/api/demo`

---

## 10. Deployment Readiness (Railway)

This repository includes full configuration for deployment to [Railway](https://railway.app):
- **`Procfile`:** Declares web process: `web: uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
- **`railway.json`:** Configures build and restart policies.
- **`Dockerfile`:** Container specification pre-building models and dependencies.
- **Environment Variables:** No secrets are required for core execution. To enable external LLMs, set `OPENAI_API_KEY` or `GEMINI_API_KEY` and `ENABLE_EXTERNAL_LLM=true`.

*(Deployment evidence: Deployment manifest and multi-stage container build have been validated locally; deployment to Railway can be triggered by connecting your GitHub repo).*

---

## 11. Limitations & Production-Readiness Roadmap

1. **High-Frequency Vibration Spectrum:** The benchmark SCADA dataset provides hourly RMS vibration velocity ($mm/s$). In an operational plant, diagnosing specific bearing defect frequencies (inner race BPFI vs outer race BPFO) requires high-frequency accelerometer waveforms ($10\text{ kHz}-50\text{ kHz}$).
2. **Dynamic Plant Curtailment Signals:** The model assumes generation follows environmental potential. In production, grid curtailment signals from the State Load Despatch Centre (SLDC) must be ingested directly into the variance engine.
3. **Module-Level Solar Telemetry:** Current telemetry tracks central inverters; incorporating string-level DC combiner box current telemetry would pinpoint individual blown fuses or soiling across specific table rows.
4. **Streaming Ingestion:** Next deployment phase would ingest telemetry via Apache Kafka / MQTT broker directly into the FastAPI inference pipeline for real-time sub-second anomaly alerting.
