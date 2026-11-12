# Greenko Group — Project Review Presentation Guide & Interview Defense
**Candidate:** Ghnanendra Chari Kata  
**Project:** Renewable Asset Fault & Anomaly Detection Copilot (Project Assignment Set 2)  
**Submission Email:** cacd.support@talentsprint.com  
**1:1 Project Review:** Tomorrow, Nov 13, 2026, at 3:00 PM  

---

## 1. 60-Second Opening Elevator Pitch (Rehearse This First!)
> *"Good afternoon! My case study is the **Renewable Asset Fault & Anomaly Detection Copilot** built for Greenko's solar and wind assets. In utility-scale renewable plants, when an inverter or turbine underperforms or develops mechanical distress, control room operators are overwhelmed by raw telemetry without clear root-cause diagnosis.
> 
> My solution solves this in an end-to-end pipeline:
> 1. It ingests hourly SCADA telemetry (irradiance, wind speed, temperature, vibration, power).
> 2. It predicts expected generation using a **Random Forest Regressor** to compute precise **actual-versus-forecast variance**.
> 3. It detects anomalous multi-sensor conditions using **Isolation Forest** combined with **ISO 10816** vibration standards.
> 4. Finally, a **RAG-based GenAI Copilot** translates the mathematical variance and thermal-vibration patterns into operator-friendly explanations with actionable field checklists.
> 
> The entire system is implemented in Python with a FastAPI backend and lightweight UI, achieving 100% recall on critical failure events."*

---

## 2. PART A | Problem Understanding & ML Solution

### 1. Explain the Problem Statement
- **What is going wrong?**
  In solar PV plants and wind farms, equipment failures (such as cooling fan seizures in central inverters or bearing spalling in turbine drivetrains) often manifest as subtle co-occurring anomalies (temperature rises and vibration increases). Without predictive anomaly triage, minor mechanical degradation escalates into catastrophic failure, causing weeks of unplanned outages and revenue loss.
- **Intended User:**
  Control room dispatchers, operations analysts, and field maintenance engineers at Greenko.
- **Input & Output:**
  - **Inputs:** SCADA telemetry (`timestamp_id`, `asset_id`, `hour`, `irradiance_wm2`, `wind_speed_ms`, `temperature_c`, `vibration_mm_s`, `power_mw`).
  - **Outputs:** Anomaly score, critical alarm severity, expected generation baseline, actual-versus-expected variance ($MW$ and %), operator-ready natural language root-cause explanation, and field inspection checklist.
- **What Success Looks Like:**
  Zero missed critical faults (100% recall), sub-0.02 MW power forecasting error ($R^2 > 0.99$), and zero hallucination in copilot explanations.

### 2. Data & Overall Solution Flow
- **Data Overview:** 240 chronological hourly records across `SOLAR_INV_01` (120 records, 1.5 MW inverter) and `WTG_07` (120 records, 2.6 MW wind turbine).
- **Key Features:**
  - Irradiance ($W/m^2$) & Wind Speed ($m/s$): Driving environmental potential.
  - Temperature ($^\circ C$) & Vibration ($mm/s$ RMS): Health indicators for electrical and mechanical components.
  - Hour cyclical encoding (`sin_hour`, `cos_hour`): Preserves 24-hour diurnal continuity.
- **Cleaning & Time-Handling:**
  Preserved strict chronological order; 70% train split (Days 1–3.5) and 30% test split (Days 3.5–5). **No future-data leakage**.

### 3. Justify the ML Model
- **Task Formulations:**
  1. **Time-Series Regression:** Predicting normal expected power generation ($MW$).
  2. **Unsupervised Anomaly Detection:** Detecting abnormal multi-sensor operational states.
- **Chosen Models & Baselines:**
  - **Power Forecasting:** **Random Forest Regressor** (chosen) vs **Linear Physics Model** (baseline).
    - *Why RF?* Solar saturation and wind turbine aerodynamic curves ($P \propto v^3$) are nonlinear. RF captures complex interactions and is resilient to noise.
    - *Key Hyperparameters:*
      - `n_estimators=100`: Averages 100 trees to eliminate single-tree variance.
      - `max_depth=6`: Caps tree depth to avoid memorizing noise and maintain smooth physical power curves.
      - `min_samples_split=3`: Enforces generalization across multiple observations.
  - **Anomaly Detection:** **Isolation Forest** (chosen) vs **Statistical 3-Sigma/IQR** (baseline).
    - *Why Isolation Forest?* Isolates anomalous observations with fewer partition steps without assuming normal Gaussian distributions.
    - *Key Hyperparameters:*
      - `contamination=0.035`: Calibrated to flag the top 3.5% most severe anomalies.
      - `n_estimators=100`: High tree count for stable scoring.

---

## 3. PART B | Results, Variance & GenAI Copilot

### 4. Validation, Performance & Variance
- **Evaluation Metrics (Chronological Test Set):**
  - **Solar Forecasting:** $\text{MAE} = 0.0012\text{ MW}$, $\text{RMSE} = 0.0024\text{ MW}$, $R^2 = 1.0000$.
  - **Wind Forecasting:** $\text{MAE} = 0.0178\text{ MW}$ ($17.8\text{ kW}$ on a $2.6\text{ MW}$ turbine), $\text{RMSE} = 0.0258\text{ MW}$, $R^2 = 0.9986$.
  - **Critical Anomaly Recall:** **5/5 (100.0%)** recall across all verified ground-truth failure events.
- **Variance Analysis & Clear Forecast Miss Demonstration:**
  - **Sign Convention:** $\text{Variance} = \text{Actual} - \text{Forecast}$ (Negative means generation deficit).
  - **Demonstration Event (WTG_07, Timestamp #174, Hour 06:00):**
    - Actual Power: $0.711\text{ MW}$
    - Expected Power: $1.277\text{ MW}$
    - **Variance:** $-0.566\text{ MW}$ ($-44.3\%$ deficit).
    - Concurrent Telemetry: Bearing vibration $= 3.82\text{ mm/s RMS}$ (exceeds ISO 10816 Zone D alert of $>3.5\text{ mm/s}$); Temperature $= 46.88^\circ\text{C}$ (vs morning baseline of $\sim 25^\circ\text{C}$).
    - **Physical Meaning:** Mechanical friction and vibration in generator/gearbox bearings caused automatic turbine derating and severe power shortfall.

### 5. Explain the GenAI Copilot
- **Exact Role of GenAI:**
  Translates structured telemetry deltas and variance into an executive diagnostic report with prioritized maintenance actions.
- **Why RAG?**
  We use RAG to retrieve verified Greenko Standard Operating Procedures and engineering standards (**ISO 10816-21** and **IEC 62109-1**). The copilot never guesses causes; it matches the sensor pattern to the standard.
- **Safeguards & Fallback:**
  - Zero hallucination: All claims are bound to measured numbers and retrieved SOP clauses.
  - Safe API management: Environment variables only (`OPENAI_API_KEY`).
  - **Deterministic Offline Generator:** Works 100% offline without external API keys or network latency, ensuring complete reproducibility for the interviewer review.
- **Clarification:** It is a **grounded diagnostic copilot**, not an unconstrained autonomous agent.

---

## 4. PART C | Repository Walkthrough (Point & Speak)

| File / Folder to Point To | Exact 1–2 Sentence Script |
|---|---|
| **`README.md`** | *"Here is the complete project documentation covering the Greenko business context, architecture diagram, mathematical formulations, evaluation results, and deployment steps."* |
| **`data/asset_telemetry.csv`** | *"This contains 240 hourly SCADA telemetry points for `SOLAR_INV_01` and `WTG_07`, tracking irradiance, wind velocity, temperature, vibration, and generation."* |
| **`backend/pipeline.py`** | *"This is our core ML pipeline. It implements chronological time splits without future leakage, trains Random Forest power models, and fits Isolation Forest anomaly detectors."* |
| **`backend/train_and_evaluate.py`** | *"This standalone evaluation script reproduces our MAE, RMSE, $R^2$, and anomaly recall metrics, saving serialized models to `models/`."* |
| **`backend/domain_knowledge.py` & `backend/copilot.py`** | *"Here is our RAG knowledge base containing ISO 10816 and IEC standards, and the Copilot module that synthesizes grounded operator explanations."* |
| **`backend/app.py` & `frontend/index.html`** | *"FastAPI serves our REST endpoints (`/api/demo`, `/api/run`, `/api/predict`) and serves the lightweight operator interface."* |
| **`tests/test_api.py`** | *"We wrote an automated 8-test suite testing pipeline metrics, anomaly recall, and API contracts with a 100% pass rate."* |
| **`Procfile` & `railway.json`** | *"Configured for immediate deployment on Railway using Nixpacks or Dockerfile."* |

---

## 5. Answers to Anticipated Interviewer Questions

### Q1: "Why is power forecasting treated as a regression problem rather than classification?"
> *"Active power generation in MW is a continuous physical variable. Operators need to know the exact megawatt magnitude and dollar impact of a generation variance ($\Delta P$), not just a binary 'normal/abnormal' bucket. Predicting continuous power allows us to compute precise percentage deficits (e.g. $-44.3\%$) to schedule grid balancing reserves."*

### Q2: "What would happen if your most important feature were removed?"
> *"For solar, plane-of-array irradiance is the primary driver ($P \approx \eta \cdot G$). If removed, the model would be forced to approximate power from the time-of-day feature alone, which fails on cloudy days. For wind, wind speed has a cubic relationship with kinetic power; without it, wind forecasting is impossible."*

### Q3: "Why did you choose Random Forest over a simpler alternative?"
> *"A simple linear model assumes constant efficiency. But wind turbines have distinct physical operating zones: cut-in speed ($\approx 3\text{ m/s}$), cubic aerodynamic ramp, and pitch-regulated rated power saturation ($\approx 12\text{ m/s}$). Random Forest models these nonlinear transitions naturally while resisting sensor noise."*

### Q4: "How do you know your model works on new data, not just training data?"
> *"We strictly preserved chronological time order. The training set used the first 70% of chronological observations, and evaluation was performed on the remaining unseen 30% future test set. On this test set, the solar model achieved an $R^2$ of 1.000 and wind achieved 0.9986, proving strong generalization without lookahead leakage."*

### Q5: "What does a negative forecast variance mean, and what explains it?"
> *"We follow the standard utility sign convention: $\text{Variance} = \text{Actual} - \text{Forecast}$. A negative variance means a generation deficit. For example, at timestamp #174 on WTG_07, the forecast was 1.277 MW but actual was 0.711 MW ($-0.566\text{ MW}$, or $-44.3\%$). Concurrently, drivetrain vibration hit $3.82\text{ mm/s}$ and temperature reached $46.9^\circ\text{C}$, proving mechanical distress caused turbine derating."*

### Q6: "What exactly is generated by the LLM versus calculated by Python?"
> *"All numerical quantities—expected power, variance in MW, variance percentage, anomaly scores, and ISO severity thresholds—are 100% computed deterministically in Python using scikit-learn and NumPy. The GenAI layer only translates these verified facts and retrieved SOP clauses into concise, operator-friendly natural language explanations."*

### Q7: "Is your implementation an agent?"
> *"No, we are honest and precise: this is a **grounded diagnostic copilot**, not an autonomous agent. It follows a structured pipeline: telemetry $\to$ ML scoring $\to$ domain SOP retrieval $\to$ grounded explanation. It does not take autonomous actions or execute tool loops."*

### Q8: "What failed during implementation, and what did you learn?"
> *"Initially, fitting Isolation Forest on the combined fleet caused solar night records (zero irradiance and zero generation) to be partially confounded with low-wind conditions. We solved this by separating the asset-specific feature pipelines and incorporating domain physics (ISO 10816 thresholds and hourly temperature deviation features)."*

---

## 6. 20-Second Closing Statement
> *"This solution turns raw renewable telemetry into a working, validated predictive maintenance and explanation copilot. I validated our models with strict chronological splits, proved 100% recall on critical failure modes, and grounded all GenAI outputs in engineering standards. My next priority would be streaming ingestion via MQTT/Kafka and high-frequency vibration FFT spectral analysis for production deployment."*
