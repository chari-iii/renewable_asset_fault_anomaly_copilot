# Greenko Renewable Asset Fault & Anomaly Detection Copilot

**Industrial AI Solution for Predictive Maintenance, Anomaly Triage & Decision Support in Solar and Wind Fleet Assets**

---

## 1. Project Overview
Greenko operates utility-scale renewable energy portfolios comprising central solar photovoltaic (PV) power plants and wind turbine generators (WTGs). In large-scale renewable generation, operational anomalies—such as inverter cooling blower failures or wind turbine drivetrain bearing degradation—often manifest early through subtle multi-sensor telemetry signatures before escalating into catastrophic plant equipment failure.

The **Renewable Asset Fault & Anomaly Detection Copilot** is an enterprise-grade AI system that ingests hourly SCADA telemetry, computes expected generation baselines to perform actual-versus-forecast variance analysis, flags anomalous multi-sensor patterns using Isolation Forest and ISO 10816 standards, and provides plant operators with grounded natural language diagnostic triage and standard operating procedure (SOP) maintenance checklists.

---

## 2. Problem Statement
In renewable energy plant control rooms and field substations:
1. **Unplanned Downtime & Catastrophic Damage:** Bearing failures, lubrication breakdown, and inverter cooling fan seizures cause prolonged outages, expensive crane mobilizations, and substantial commercial revenue loss.
2. **Alert Fatigue & Ambiguous Root Causes:** Conventional fixed SCADA alarm thresholds trigger frequent nuisance alarms without isolating the underlying physical root cause (e.g. failing to connect an abnormal nighttime temperature surge with cooling fan bearing rub).
3. **Forecast Variance Uncertainty:** When generation deviates from scheduled day-ahead forecasts, operators need instant clarity on whether the shortfall is due to environmental resource fluctuations, utility grid curtailment orders, or impending mechanical distress.

---

## 3. Objectives
- **Automated Anomaly Detection:** Detect multi-sensor operational anomalies across solar and wind assets with high precision and zero missed critical failure events (100% recall on verified failure modes).
- **Physical Variance Analysis:** Provide continuous expected generation forecasts to quantify active power variances ($\Delta P = P_{\text{actual}} - P_{\text{expected}}$) in both megawatts ($MW$) and percentage terms.
- **Operator Decision Support:** Deliver grounded, deterministic diagnostic explanations and field troubleshooting checklists retrieved from Greenko plant SOPs, ISO 10816-21, and IEC standards.
- **Interactive Fleet Visibility:** Offer an industrial dashboard for real-time fleet overview, individual asset telemetry curves, anomaly event logs, and model validation inspection.

---

## 4. Technology Stack
- **Backend Framework:** FastAPI (0.115.0), Starlette (0.38.6), Uvicorn (0.30.6).
- **Data Validation & Schemas:** Pydantic v2 (2.8.2).
- **Machine Learning & Analytics:** Scikit-Learn (1.5.1), Pandas (2.2.2), NumPy (1.26.4), SciPy (1.17.1), Joblib (1.6.0).
- **Frontend Dashboard:** HTML5, CSS3 (Enterprise Dark Navy/Charcoal Design System), Vanilla JavaScript (ES6+), Chart.js (4.4.4).
- **Testing & Quality Assurance:** Pytest (9.2.0), HTTPX (0.28.1).
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
|  * Strict chronological sorting (No future-data leakage)                          |
|  * Cyclical diurnal time encoding (sin_hour, cos_hour)                             |
|  * Physics expectations: Solar PR (0.0019 MW/W/m²) | Wind Power Curve              |
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
|  * Deterministic Physics Engine grounded in verified SCADA telemetry             |
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
  - `timestamp_id`: Sequential hourly observation index.
  - `asset_id`: Asset identifier string.
  - `hour`: Diurnal hour (0 to 23).
  - `irradiance_wm2`: Plane-of-array solar irradiance ($W/m^2$).
  - `wind_speed_ms`: Nacelle anemometer wind speed ($m/s$).
  - `temperature_c`: Inverter cabinet / generator bearing temperature ($^\circ C$).
  - `vibration_mm_s`: RMS vibration velocity ($mm/s$).
  - `power_mw`: Active power generation injected into the plant bus ($MW$).

---

## 7. Data Preprocessing
- **Chronological Time Ordering:** Data points are ordered strictly by time. No random shuffling is performed to eliminate temporal lookahead leakage.
- **Chronological Train/Test Split:**
  - **Training Set:** First 70% of chronological observations (Days 1–3.5, 84 observations per asset).
  - **Testing Set:** Final 30% of chronological observations (Days 3.5–5, 36 observations per asset).
- **Feature Transformations:**
  - Diurnal circularity preserved using $\sin(2\pi \cdot \text{hour} / 24)$ and $\cos(2\pi \cdot \text{hour} / 24)$.
  - Missing and null value checks: Zero nulls or NaNs in the SCADA feed.

---

## 8. ML Models and Anomaly Detection Methodology

### 8.1 Expected Generation Models
- **Chosen Architecture:** Random Forest Regressor (`n_estimators=100`, `max_depth=6`, `min_samples_split=3`).
- **Baseline Comparison:** First-principles Linear Physics model ($P_{\text{solar}} = \eta \cdot G$; $P_{\text{wind}} = a \cdot v + b$).
- **Features Used:**
  - Solar: `irradiance_wm2`, `temperature_c`, `hour`, `hour_sin`, `hour_cos`.
  - Wind: `wind_speed_ms`, `temperature_c`, `hour`, `hour_sin`, `hour_cos`.

### 8.2 Anomaly Detection Engine
- **Unsupervised Model:** Isolation Forest (`contamination=0.035`, `n_estimators=100`, `max_samples='auto'`).
- **Domain Severity Classification:**
  - `CRITICAL`: Drivetrain vibration $\ge 3.5\text{ mm/s}$ (ISO 10816 Zone D) OR (temperature $\ge 45.0^\circ\text{C}$ with Isolation Forest flag).
  - `HIGH`: Vibration $\ge 2.0\text{ mm/s}$ OR temperature $\ge 42.0^\circ\text{C}$.
  - `MEDIUM`: Isolation Forest statistical outlier (`iso_pred == -1`).
  - `LOW`: Mild sensor drift.
  - `NORMAL`: Nominal operational state.

---

## 9. Solar and Wind Monitoring

### 9.1 Solar Asset Monitoring (`SOLAR_INV_01`)
- Monitors photovoltaic power conversion efficiency against solar irradiance.
- Analyzes nighttime cooling decay: under healthy conditions, cabinet temperature decays toward ambient ($\approx 25^\circ\text{C}$) when irradiance is 0.
- **Detected Fault:** At Timestamps 67, 68, 69 (Hours 19:00 - 21:00), cabinet temperature spiked to $53.48^\circ\text{C}$ and blower vibration rose to $3.98\text{ mm/s}$, diagnosing a cooling blower bearing rub / motor failure.

### 9.2 Wind Turbine Monitoring (`WTG_07`)
- Tracks active power against nacelle wind velocity and vibration against ISO 10816-21 thresholds.
- **Detected Fault:** At Timestamp 174 (Hour 06:00), WTG_07 exhibited $3.82\text{ mm/s RMS}$ vibration (exceeding the $3.5\text{ mm/s}$ ISO Zone D danger threshold) and $46.88^\circ\text{C}$ bearing temperature, resulting in a $-0.566\text{ MW}$ ($-44.3\%$) generation deficit due to bearing distress.

---

## 10. API Endpoint Documentation

| Endpoint | Method | Input Parameters | Output Response | Description |
|---|---|---|---|---|
| `/` | `GET` | None | HTML | Serves the enterprise dashboard UI |
| `/health` | `GET` | None | JSON | Health check endpoint for monitoring |
| `/api/fleet/summary` | `GET` | None | JSON (`FleetSummary`) | Aggregated fleet KPIs and health score |
| `/api/assets` | `GET` | None | JSON (`List[AssetCard]`) | Overview metrics for Solar and Wind assets |
| `/api/telemetry` | `GET` | `asset_id` (optional) | JSON (`List[TelemetrySeriesPoint]`) | Time-series data for interactive charts |
| `/api/anomalies` | `GET` | `asset_id`, `severity` (optional) | JSON (`List[AnomalyItem]`) | Filtered list of detected anomalies |
| `/api/sops` | `GET` | None | JSON | Greenko SOP database and standards |
| `/api/demo` | `GET` | None | JSON | Complete end-to-end demo payload |
| `/api/run` | `GET`/`POST` | `Query` (text, asset_id, ts) | JSON (`PipelineRunResponse`) | Executes copilot pipeline and explanation |
| `/api/predict` | `POST` | `PredictionRequest` | JSON | Point prediction and anomaly scoring |
| `/api/metrics` | `GET` | None | JSON | Chronological model validation metrics |

Interactive Swagger documentation is available at **`http://127.0.0.1:8000/docs`**.

---

## 11. Copilot and SOP Retrieval Explanation
The Copilot operates as a **grounded deterministic diagnostic engine**:
- **Knowledge Base:** Contains structured plant Standard Operating Procedures:
  - `SOP-PV-042` (IEC 62109-1): Central Inverter Thermal Runaway & Cooling Blower Seizure.
  - `SOP-WTG-089` (ISO 10816-21): Wind Turbine Drivetrain Vibration Alert & Bearing Distress.
  - `SOP-PV-018` (IEC 61724): Solar Generation Deficit & DC String Triage.
  - `SOP-WTG-012` (IEC 61400): Wind Power Curve Aerodynamic Deficit Investigation.
- **Zero Hallucination Guarantee:** Diagnoses are synthesized directly from verified numerical SCADA metrics, ISO/IEC thresholds, and retrieved SOP checklists without ungrounded guessing.

---

## 12. Model Evaluation Results

### 12.1 Generation Forecasting Metrics (Chronological Test Split)

| Target Asset | Model Architecture | MAE ($MW$) | RMSE ($MW$) | $R^2$ Score |
|---|---|---:|---:|---:|
| **`SOLAR_INV_01`** | **Random Forest Regressor (n=100, depth=6)** | **0.0012 MW** | **0.0024 MW** | **1.0000** |
| `SOLAR_INV_01` | Linear Physics Baseline ($P = \eta \cdot G$) | 0.0001 MW | 0.0002 MW | 1.0000 |
| **`WTG_07`** | **Random Forest Regressor (n=100, depth=6)** | **0.0178 MW** | **0.0258 MW** | **0.9986** |
| `WTG_07` | Linear Wind Speed Baseline ($P \approx 0.45\cdot v - 1.36$) | 0.0010 MW | 0.0012 MW | 1.0000 |

### 12.2 Anomaly Detection Ground-Truth Recall

| Asset Target | Algorithm | Predefined Benchmark Faults | Detected Recall | Precision | F1 Score |
|---|---|---|---:|---:|---:|
| **`SOLAR_INV_01`** | Isolation Forest + ISO Rules | Nighttime blower failure (TS #67, 68, 69) | **3 / 3 (100.0%)** | 60.0% | 0.750 |
| **`WTG_07`** | Isolation Forest + ISO Rules | Drivetrain bearing distress (TS #173, 174) | **2 / 2 (100.0%)** | 40.0% | 0.571 |
| **Fleet Critical Filter** | **Hybrid ML + ISO 10816 Zone D** | **All 5 critical failure events** | **5 / 5 (100.0%)** | **100.0%** | **1.000** |

### 12.3 Engineering Investigation Note (Linear vs Random Forest MAE)
On this benchmark dataset, Linear Regression achieves lower MAE than Random Forest because the synthetic benchmark dataset was constructed with affine linear generation formulas without measurement noise. However, in physical operating utility plants, real solar inverter MPPT clipping, temperature derating coefficients, and aerodynamic cubic wind power curves are non-linear, making Random Forest theoretically necessary for real-world plant operations.

---

## 13. Limitations
1. **Synthetic Benchmark Telemetry:** The bundled dataset contains synthetic hourly SCADA observations with injected fault patterns. Performance reflects benchmark validation rather than multi-year operational plant history.
2. **Vibration Frequency Resolution:** Telemetry provides hourly RMS velocity ($mm/s$). In an operational facility, diagnosing specific bearing defect frequencies (BPFI/BPFO) requires high-frequency accelerometer waveforms ($10\text{ kHz}-50\text{ kHz}$).
3. **Curtailment Ingestion:** Grid curtailment orders from load dispatch centers are currently modeled via generation deficit rather than direct digital dispatch signals.

---

## 14. Installation Instructions

```bash
# 1. Clone repository
git clone <your-repo-link>
cd renewable_asset_fault_anomaly_copilot/02_renewable_asset_fault_anomaly_copilot

# 2. Create and activate virtual environment (Python 3.11 recommended)
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## 15. Backend Startup Instructions

```bash
# Start FastAPI backend server
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --app-dir 02_renewable_asset_fault_anomaly_copilot --reload
```

---

## 16. Frontend Startup Instructions
The frontend is served directly by FastAPI at the root URL:
1. Ensure the backend server is running on `127.0.0.1:8000`.
2. Open your web browser and navigate to **`http://127.0.0.1:8000/`**.
3. The dashboard will automatically initialize, load live telemetry and KPIs, and render all interactive charts.

---

## 17. Railway Deployment Instructions

The repository is configured for deployment to [Railway](https://railway.app):
- **`Procfile`:** Declares web server command: `web: uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
- **`railway.json`:** Configures build settings and restart policies.
- **`Dockerfile`:** Provides containerized build with dependency installation and model preloading.

### Steps to Deploy on Railway:
1. Log in to your Railway dashboard at [railway.app](https://railway.app).
2. Click **New Project** $\to$ **Deploy from GitHub repo**.
3. Select this repository.
4. Set the Root Directory to `02_renewable_asset_fault_anomaly_copilot` (or leave default if deploying from subfolder).
5. Railway will automatically detect the `Procfile` / `Dockerfile` and deploy the service.
6. Once deployed, test the `/health` endpoint to verify service availability.

*(Note: Live deployment requires a Railway account and repository link).*

---

## 18. Screenshots & Dashboard Walkthrough Section

The enterprise dashboard provides 7 specialized views:

1. **Fleet Overview Tab:**
   - Real-time KPI cards for Monitored Assets, SCADA Records, Total Anomalies, Critical Alarms, Fleet Average Power, and Fleet Health Index.
   - Dual-series line chart tracking Actual vs Forecast baseline power generation.
   - Severity distribution doughnut chart dynamically computed from backend telemetry.
2. **Solar Assets Tab:**
   - Scatter plot of measured AC power versus plane-of-array solar irradiance ($W/m^2$).
   - Dual-axis line chart tracking the inverter cabinet temperature and cooling fan vibration envelope.
3. **Wind Assets Tab:**
   - Turbine power curve tracking measured power output against wind velocity.
   - RMS vibration tracking against the ISO 10816 Zone D ($3.5\text{ mm/s}$) danger boundary.
4. **Anomaly Detection Tab:**
   - Searchable, filterable operational event table with instant asset and severity filtering.
   - Direct "Diagnose" action buttons linking events to the Copilot workbench.
5. **AI Copilot Tab:**
   - Dedicated operator assistance workbench displaying problem headlines, physical fault hypotheses, verified telemetry evidence, actual-vs-expected variances, and retrieved SOP maintenance checklists.
6. **Model Performance Tab:**
   - Side-by-side comparison tables of Random Forest Regressors vs Linear baselines with full documentation of the benchmark data characteristics.
7. **Settings & API Tab:**
   - Backward-compatible raw JSON API inspector for quick verification of all endpoints.
