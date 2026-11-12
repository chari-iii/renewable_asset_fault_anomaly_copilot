# Greenko Renewable Asset Fault & Anomaly Detection Copilot

**Industrial Operational Intelligence for Predictive Maintenance, Anomaly Triage & Decision Support in Solar and Wind Fleet Assets**

---

## 1. Project Overview
Greenko operates utility-scale renewable energy portfolios comprising central solar photovoltaic (PV) power plants and wind turbine generators (WTGs). In large-scale renewable generation, operational anomalies—such as inverter cooling blower failures or wind turbine drivetrain bearing degradation—often manifest early through subtle multi-sensor telemetry signatures before escalating into catastrophic plant equipment failure.

The **Renewable Asset Fault & Anomaly Detection Copilot** is an enterprise-grade operational intelligence system that ingests hourly SCADA telemetry, computes expected generation baselines to perform actual-versus-forecast variance analysis, flags anomalous multi-sensor patterns using Isolation Forest and ISO 10816 standards, and provides plant operators with deterministic diagnostic triage and standard operating procedure (SOP) maintenance checklists.

---

## 2. Problem Statement
In renewable energy plant control rooms and field substations:
1. **Unplanned Downtime & Equipment Damage:** Mechanical bearing degradation, lubrication starvation, and inverter cooling blower seizures cause prolonged outages, expensive crane mobilizations, and commercial revenue loss.
2. **Alert Fatigue & Ambiguous Root Causes:** Conventional single-sensor threshold alarms trigger nuisance notifications without isolating underlying multi-sensor relationships (e.g. connecting an abnormal nighttime temperature surge with cooling fan bearing rub).
3. **Forecast Variance Uncertainty:** When generation deviates from scheduled forecasts, control-room operators require rapid triage to distinguish environmental resource deficits from mechanical degradation or utility curtailment.

---

## 3. Objectives
- **Multi-Sensor Anomaly Detection:** Detect multi-sensor operational anomalies across solar inverters and wind turbines using machine learning (Isolation Forest) combined with ISO 10816-21 vibration thresholds.
- **Physical Variance Analysis:** Provide continuous expected generation forecasts to quantify active power variances ($\Delta P = P_{\text{actual}} - P_{\text{expected}}$) in megawatts ($MW$) and percentage terms.
- **Operator Decision Support:** Deliver deterministic diagnostic explanations and field troubleshooting checklists retrieved from Greenko plant SOPs, ISO 10816-21, and IEC standards.
- **Interactive Fleet Visibility:** Provide an industrial dashboard for real-time fleet overview, individual asset telemetry curves, anomaly event logs, and model validation inspection.

---

## 4. Technology Stack
- **Backend Framework:** FastAPI (0.115.0), Starlette (0.38.6), Uvicorn (0.30.6).
- **Data Validation & Schemas:** Pydantic v2 (2.8.2).
- **Machine Learning & Analytics:** Scikit-Learn (1.5.1), Pandas (2.2.2), NumPy (1.26.4), SciPy (1.17.1), Joblib (1.6.0).
- **Frontend Dashboard:** HTML5, CSS3 (Enterprise Dark Navy/Charcoal Design System), Vanilla JavaScript (ES6+), Chart.js (4.4.4).
- **Testing & Quality Assurance:** Pytest (8.0.0+ / 9.2.0), HTTPX (0.28.1).
- **Containerization & Deployment:** Docker, Procfile, Railway configuration (`railway.json`).

---

## 5. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            SCADA Telemetry Streams                                |
|   (Solar: Irradiance, Temp, Vib, Power | Wind: Wind Speed, Temp, Vib, Power)     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                1. Data Preprocessing & Physics Feature Engineering                |
|  * Strict chronological sorting (No temporal lookahead leakage)                   |
|  * Cyclical diurnal time encoding (hour_sin, hour_cos)                            |
|  * Physics expectations: Solar PR (0.0019 MW/W/m²) | Wind Power Curve Baseline    |
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
|                       5. Operator Copilot Explanation Layer                       |
|  * Structured engineering explanation: headline, fault hypothesis, evidence       |
|  * Deterministic Physics Engine grounded in verified SCADA telemetry & SOPs      |
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

## 6. Dataset Description
The system is evaluated on the benchmark SCADA telemetry dataset located at `data/asset_telemetry.csv`:
- **Total Records:** 240 chronological hourly records.
- **Assets Monitored:**
  - `SOLAR_INV_01` (120 records): Central solar PV inverter (1.5 MW nominal AC capacity).
  - `WTG_07` (120 records): Utility-scale Wind Turbine Generator (2.6 MW nominal capacity).
- **Telemetry Channels:**
  - `timestamp_id`: Sequential hourly observation index (1 to 240).
  - `asset_id`: Asset identifier string.
  - `hour`: Diurnal hour (0 to 23).
  - `irradiance_wm2`: Plane-of-array solar irradiance ($W/m^2$).
  - `wind_speed_ms`: Nacelle anemometer wind speed ($m/s$).
  - `temperature_c`: Inverter cabinet / generator bearing temperature ($^\circ C$).
  - `vibration_mm_s`: RMS vibration velocity ($mm/s$).
  - `power_mw`: Active power generation injected into the plant bus ($MW$).

---

## 7. Data Preprocessing
- **Chronological Ordering:** Records are sorted strictly by sequential timestamp ID and diurnal hour. No random shuffling is performed, preserving continuous time-series structure.
- **Data Integrity:** The SCADA dataset is validated with zero null values, zero negative power readings during nighttime hours, and consistent physical units.
- **Chronological Train/Test Split (Expected Power Baseline):**
  - **Solar Clean Split:** 81 clean training observations (first 70% of 117 clean records; 3 benchmark fault timestamps excluded from baseline fitting), 36 test observations (last 30%).
  - **Wind Clean Split:** 82 clean training observations (first 70% of 118 clean records; 2 benchmark fault timestamps excluded from baseline fitting), 36 test observations (last 30%).
  - **Zero Future Leakage:** Models are fitted strictly on historical observations before evaluating on future time intervals.

---

## 8. Feature Engineering
- **Diurnal Circularity:** Standard numeric hours (0 to 23) create an artificial boundary between 23:00 and 00:00. Circular trigonometric encoding preserves day/night cyclicality:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
- **Solar Physics Baseline:** Expected solar power derived from plane-of-array irradiance using nominal performance ratio ($0.0019\text{ MW}/(W/m^2)$):
  $$P_{\text{solar\_expected}} = \max(0.0, G \times 0.0019)$$
- **Wind Power Curve Baseline:** Expected aerodynamic power modeled from wind speed ($v$) with cut-in ($3.0\text{ m/s}$), cubic growth ($P \propto v^3$), and rated saturation ($2.6\text{ MW}$):
  $$P_{\text{wind\_expected}} = \text{clip}(0.00582 \times v^3, 0.0, 2.6)$$

---

## 9. ML Algorithms
1. **Power Forecast Baseline — Random Forest Regressor:**
   - Architecture: `RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_split=3, random_state=42)`.
   - Captures multi-variable interactions between weather inputs, ambient temperature, and diurnal cycles without parametric assumptions.
2. **Reference Baseline — Ordinary Least Squares (OLS) Linear Regression:**
   - Evaluated alongside Random Forest to establish a transparent first-principles benchmark.
3. **Multi-Sensor Anomaly Detection — Isolation Forest:**
   - Architecture: `IsolationForest(contamination=0.035, n_estimators=100, max_samples='auto', random_state=42)`.
   - Isolates anomalous operating points in high-dimensional telemetry space ($T$, $v$, $P$, weather) by measuring tree partition path length.
4. **Engineering Vibration Boundary — ISO 10816-21:**
   - Deterministic vibration thresholds: Zone A/B ($<2.0\text{ mm/s}$ Normal), Zone C ($2.0-3.5\text{ mm/s}$ Warning/Drift), Zone D ($>3.5\text{ mm/s}$ Critical Distress).

---

## 10. Solar Anomaly Detection
- **Asset:** `SOLAR_INV_01` (1.5 MW Central Inverter).
- **Physical Signature:** Photovoltaic inverters exhibit high heat dissipation during peak daylight and natural convective cooling toward ambient ($\approx 25^\circ\text{C}$) during non-generating nighttime hours (irradiance $= 0\text{ W/m}^2$).
- **Injected Benchmark Fault (Timestamps 67, 68, 69; Hours 19:00 - 21:00):**
  - Cabinet temperature surged to $53.48^\circ\text{C}$ (normally $<30^\circ\text{C}$ at night).
  - Blower vibration rose to $3.98\text{ mm/s}$ (normally $<1.0\text{ mm/s}$).
  - Active power was $0.0\text{ MW}$ (darkness).
- **Detection Result:** Isolation Forest flags all three points ($TS=67, 68, 69$) with high anomaly scores ($>0.12$). Classified as `CRITICAL` severity under IEC 62109-1 due to auxiliary blower motor failure / mechanical rub.

---

## 11. Wind Anomaly Detection
- **Asset:** `WTG_07` (2.6 MW Wind Turbine Generator).
- **Physical Signature:** Drivetrain vibration velocity ($mm/s\text{ RMS}$) measured on the generator drive-end and gearbox high-speed shaft bearings.
- **Injected Benchmark Fault (Timestamps 173, 174; Hours 05:00 - 06:00):**
  - At Timestamp 174, wind speed was $5.87\text{ m/s}$ (expected generation $\approx 1.28\text{ MW}$).
  - Actual generation dropped to $0.711\text{ MW}$ (a $-0.566\text{ MW}$ or $-44.3\%$ deficit).
  - Drivetrain vibration surged to $3.82\text{ mm/s}$, exceeding the ISO 10816-21 Zone D threshold ($>3.5\text{ mm/s}$).
  - Bearing temperature reached $46.88^\circ\text{C}$ during cool morning hours.
- **Detection Result:** Flagged by Isolation Forest and categorized as `CRITICAL` under ISO 10816-21 Zone D. Diagnosed as high-speed shaft / generator bearing mechanical degradation.

---

## 12. Model Evaluation
The model evaluation is conducted via `backend/train_and_evaluate.py` using strict chronological splits:

### 12.1 Generation Forecasting Metrics (36-Point Chronological Test Split)

| Target Asset | Model Architecture | MAE ($MW$) | RMSE ($MW$) | $R^2$ Score |
|---|---|---:|---:|---:|
| **`SOLAR_INV_01`** | **Random Forest Regressor (depth=6)** | **0.0012 MW** | **0.0024 MW** | **1.0000** |
| `SOLAR_INV_01` | Linear Physics Baseline ($P = \eta \cdot G$) | 0.0001 MW | 0.0002 MW | 1.0000 |
| **`WTG_07`** | **Random Forest Regressor (depth=6)** | **0.0178 MW** | **0.0258 MW** | **0.9986** |
| `WTG_07` | Linear Wind Speed Baseline ($P \approx 0.45\cdot v - 1.36$) | 0.0010 MW | 0.0012 MW | 1.0000 |

### 12.2 Anomaly Detection Benchmark Capture

| Asset Target | Algorithm | Predefined Benchmark Faults | Captured | Precision | F1 Score |
|---|---|---|:---:|---:|---:|
| **`SOLAR_INV_01`** | Isolation Forest (contam=0.035) | Blower Failure (TS #67, 68, 69) | **3 / 3 (100.0%)** | 60.0% | 0.750 |
| **`WTG_07`** | Isolation Forest (contam=0.035) | Bearing Distress (TS #173, 174) | **2 / 2 (100.0%)** | 40.0% | 0.571 |

### 12.3 Investigation Finding: Why Linear Baseline Outperforms Random Forest on Benchmark Data
- **Synthetic Data Characteristics:** In the provided benchmark dataset `asset_telemetry.csv`, generation values were synthesized using exact affine formulas ($P_{\text{solar}} = 0.0019 \times G$; $P_{\text{wind}} \approx 0.45 \times v - 1.36$) with zero simulated environmental noise.
- **Model Mechanics:** Ordinary Least Squares (OLS) fits straight affine lines with near-zero mathematical error. Random Forest creates orthogonal step partitions, leaving small piecewise residuals.
- **Physical Plant Justification:** In real operating utility plants, power relationships are non-linear: solar panels exhibit negative temperature derating ($\approx -0.38\%/\text{°C}$) and inverter MPPT clipping, while wind turbines follow cubic aerodynamic power curves ($P \propto v^3$) with pitch regulation. Random Forest is engineering-justified for real plant operations.

---

## 13. SOP Retrieval Workflow
The Standard Operating Procedure (SOP) retrieval mechanism operates in [backend/domain_knowledge.py](backend/domain_knowledge.py) as a **deterministic rule-grounded retriever** against structured engineering knowledge:
1. **Input Signature:** Receives asset ID, active temperature ($T$), RMS vibration ($v$), generation variance ($\Delta P$), and diurnal hour.
2. **Matching Criteria:** Evaluates telemetry against domain boundaries:
   - If Solar and ($T > 44.0^\circ\text{C}$ or $v > 2.5\text{ mm/s}$): Selects `SOP-PV-042`.
   - If Solar and ($\Delta P < -0.15\text{ MW}$): Selects `SOP-PV-018`.
   - If Wind and ($v > 2.5\text{ mm/s}$ or $T > 42.0^\circ\text{C}$): Selects `SOP-WTG-089`.
   - If Wind and ($\Delta P < -0.20\text{ MW}$): Selects `SOP-WTG-012`.
   - **Nominal Baseline:** When no abnormal temperature, vibration, or variance threshold is breached, selects `SOP-GEN-001` (Routine Preventative Monitoring) rather than a false alarm.
3. **Retrieved Standards (Illustrative Domain Knowledge Templates):**
   - `SOP-PV-042` (IEC 62109-1 / IEC 61724): Central Inverter Thermal Runaway & Cooling Fan Seizure.
   - `SOP-WTG-089` (ISO 10816-21 / IEC 61400-1): Drivetrain & Bearing Vibration Alert Procedure.
   - `SOP-PV-018` (IEC 61724-1): Solar Generation Deficit & DC String Triage.
   - `SOP-WTG-012` (IEC 61400-12-1): Wind Turbine Power Curve Deficit Investigation.
   - `SOP-GEN-001` (IEC 61724 / IEC 61400 Nominal Parameters): Nominal Asset Operation & Routine Monitoring.
4. **Architecture & Scope Notes:**
   - **Rule-Grounded Deterministic Matching:** Retrieval uses deterministic boundary evaluation directly against engineering specifications rather than an external vector database or embedding model.
   - **Illustrative Knowledge Base Disclaimer:** These SOP templates are synthesized from international industry standards (IEC 62109, ISO 10816, IEC 61724, IEC 61400) for this case study, rather than official Greenko internal proprietary documents.

---

## 14. Copilot Explanation Process
The Copilot layer in [backend/copilot.py](backend/copilot.py) synthesizes structured operator guidance through four deterministic steps:
1. **Variance Sign Convention:** Formats $\Delta P = P_{\text{actual}} - P_{\text{expected}}$ where negative values indicate generation deficit/underperformance.
2. **Telemetry Evidence Assembly:** Compiles verified numerical facts (temperature, vibration, power, weather condition, ISO 10816 zone classification).
3. **Engineering Hypothesis Formulation:** Connects the multi-sensor signature to physical mechanisms (e.g., cooling blower motor rub or high-speed shaft bearing spalling).
4. **Checklist Generation:** Returns standard maintenance steps from the retrieved SOP (e.g., FLIR thermal scan, LOTO nacelle inspection, FFT spectrum analysis for BPFO/BPFI defect harmonics).
5. **Deterministic Safeguard:** Output is strictly derived from verified SCADA limits and standard operating procedures. No external LLM is wired; the system operates with zero external API dependencies.

---

## 15. API Endpoint Documentation

| Endpoint | Method | Input Parameters | Output Model | Description |
|---|---|---|---|---|
| `/` | `GET` | None | HTML | Serves the enterprise dashboard UI |
| `/health` | `GET` | None | JSON | Service health status (`status: healthy`) |
| `/api/health` | `GET` | None | JSON | Service health status |
| `/api/fleet/summary` | `GET` | None | `FleetSummary` | Aggregated fleet KPIs and health index |
| `/api/assets` | `GET` | None | `List[AssetCard]` | Individual Solar and Wind asset overview cards |
| `/api/telemetry` | `GET` | `asset_id` (optional) | `List[TelemetryPoint]` | Time-series telemetry points for charts |
| `/api/anomalies` | `GET` | `asset_id`, `severity` | `List[AnomalyItem]` | Filtered list of detected operational anomalies |
| `/api/sops` | `GET` | None | JSON | Greenko SOP database and industry standards |
| `/api/demo` | `GET` | None | JSON | Complete end-to-end evaluation payload |
| `/api/run` | `GET` / `POST` | `Query` (text, asset_id, ts) | `PipelineRunResponse` | Runs copilot case and returns diagnosis |
| `/api/predict` | `POST` | `PredictionRequest` | JSON | Evaluates custom telemetry input point |
| `/api/metrics` | `GET` | None | JSON | Model validation and evaluation metrics |

Interactive Swagger documentation is available at **`http://127.0.0.1:8000/docs`**.

---

## 16. Frontend Description
The dashboard in [frontend/index.html](frontend/index.html) is an industrial single-page application built with enterprise dark navy / charcoal styling:
1. **Sidebar Navigation:** 7 views: Overview, Solar Assets, Wind Assets, Anomaly Detection, AI Copilot, Model Performance, and Settings/API.
2. **Top Navigation Bar:** Project branding, system operational status badge, last updated timestamp, and global asset filter.
3. **Fleet Overview Tab:** 7 SCADA KPI cards, dual-series Actual vs Forecast power line chart, and a dynamically calculated anomaly severity doughnut chart.
4. **Solar & Wind Asset Tabs:** Power conversion curves, temperature envelopes, and drivetrain vibration tracking against ISO 10816 thresholds.
5. **Anomaly Table:** Interactive table with severity badges (Critical, Medium), percentage variances, anomaly scores, and "Diagnose" action buttons.
6. **Copilot Workbench:** Displays headline, physical hypothesis, telemetry evidence array, and maintenance action items.
7. **Model Performance Tab:** Side-by-side regression metrics and anomaly detection statistics.

---

## 17. Installation Instructions

```bash
# 1. Clone repository
git clone <your-repo-url>
cd renewable_asset_fault_anomaly_copilot

# 2. Create and activate virtual environment (Python 3.11 recommended)
# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 18. Backend Startup
From the repository root directory:

```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

---

## 19. Frontend Startup
The frontend is embedded and served directly by FastAPI:
1. Ensure the backend server is running on `127.0.0.1:8000`.
2. Open your web browser and navigate to:
   **`http://127.0.0.1:8000/`**
3. The dashboard will automatically initialize, query `/api/fleet/summary`, `/api/anomalies`, and `/api/telemetry`, and render all interactive charts.

---

## 20. Testing Instructions
Run the automated test suite using pytest from the repository root:

```bash
# Run automated test suite (12 tests)
python -m pytest -q

# Run standalone evaluation & reproducibility verification
python backend/train_and_evaluate.py
```

All 12 automated unit and integration tests pass consistently (`12 passed in 2.73s`).

---

## 21. Known Limitations
1. **Synthetic Benchmark Telemetry:** The bundled dataset contains 240 synthetic hourly SCADA observations with injected fault patterns. Performance reflects benchmark validation rather than multi-year operational plant history.
2. **In-Sample Anomaly Evaluation Limitation:** In `asset_telemetry.csv`, all predefined synthetic fault events (Solar TS #67-69, Wind TS #173-174) are located exclusively in the first 70% of chronological records. The 30% chronological test split contains zero labeled faults, precluding independent out-of-sample supervised recall calculation without fabricating labels.
3. **Vibration Frequency Resolution:** Telemetry provides hourly RMS velocity ($mm/s$). In an operational facility, diagnosing specific bearing defect frequencies (BPFI/BPFO) requires high-frequency accelerometer waveforms ($10\text{ kHz}-50\text{ kHz}$).
4. **Curtailment Ingestion:** Grid curtailment orders from load dispatch centers are currently modeled via generation deficit rather than direct digital dispatch signals.

---

## 22. Railway Deployment Instructions
The application is pre-configured for deployment on [Railway](https://railway.app):
- **`Procfile`:** `web: uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
- **`railway.json`:** Build and restart configuration using NIXPACKS.
- **`Dockerfile`:** Containerized Python 3.11 build with automated dependency installation.

### Deployment Steps:
1. Log in to [railway.app](https://railway.app) and create a **New Project**.
2. Select **Deploy from GitHub repo** and select this repository.
3. Railway will automatically detect the root `Procfile` / `Dockerfile` and deploy the service.
4. Verify deployment by visiting `https://<your-railway-domain>/health`.

---

## 23. Screenshots & Demonstration Evidence
The `screenshots/` directory contains documentation and instructions for capturing all 7 dashboard views:
- `01_fleet_overview.png`: Real-time KPI cards, dual-series generation chart, and severity doughnut.
- `02_solar_monitoring.png`: Solar inverter power vs irradiance curve and cabinet temperature tracking.
- `03_wind_monitoring.png`: Turbine power curve and vibration tracking against ISO 10816 Zone D boundary.
- `04_anomaly_detection.png`: Operational anomaly table with severity filters and diagnose actions.
- `05_ai_copilot_workbench.png`: Operator triage workbench showing fault hypothesis, sensor evidence, and retrieved SOP.
- `06_model_performance.png`: Validation table comparing Random Forest Regressors vs Linear baselines.
- `07_api_documentation.png`: Interactive Swagger UI (`/docs`).

Refer to [screenshots/README.md](screenshots/README.md) for capture procedures.
