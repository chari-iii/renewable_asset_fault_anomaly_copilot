# Dashboard Demonstration Screenshots Guide

This directory is designated for authentic screenshots of the running **Renewable Asset Fault & Anomaly Copilot** dashboard.

> **Integrity Note:** In accordance with evaluation guidelines, no synthetic or fabricated image mockups have been generated. Follow the instructions below to capture authentic high-resolution screenshots directly from your running web interface.

---

## 📸 Required Screenshot Checklist

| # | Screenshot Filename | View to Capture | Key Elements to Verify in Frame |
|---|---|---|---|
| 1 | `01_fleet_overview.png` | **Fleet Overview Tab** | 7 SCADA KPI cards (Monitored Assets, 240 Records, 10 Anomalies, 5 Critical Alarms, 0.449 MW Average Power, 95.8% Health Index), Actual vs Forecast generation chart, and dynamic severity distribution doughnut. |
| 2 | `02_solar_monitoring.png` | **Solar Assets Tab** | Asset `SOLAR_INV_01` (1.5 MW), power vs irradiance scatter curve, inverter cabinet temperature decay, and blower fan vibration envelope. |
| 3 | `03_wind_monitoring.png` | **Wind Assets Tab** | Asset `WTG_07` (2.6 MW), cubic power vs wind speed curve, and drivetrain vibration tracking showing the ISO 10816 Zone D (3.5 mm/s) threshold breach. |
| 4 | `04_anomaly_detection.png` | **Anomaly Detection Tab** | Filterable operational anomaly table with severity badges (Critical, Medium), percentage variances, anomaly scores, and "Diagnose" action buttons. |
| 5 | `05_ai_copilot_workbench.png` | **AI Copilot Tab** | Operator triage workbench for Solar TS #67 or Wind TS #174 showing headline diagnosis, physical fault hypothesis, verified sensor evidence checklist, and retrieved Greenko SOP standard (`SOP-PV-042` or `SOP-WTG-089`). |
| 6 | `06_model_performance.png` | **Model Performance Tab** | Interactive validation table comparing Random Forest Regressors vs Linear Physics Baselines on 36-point chronological splits, plus Isolation Forest benchmark recall metrics. |
| 7 | `07_api_documentation.png` | **Swagger API Docs (`/docs`)** | FastAPI interactive documentation showing `/api/fleet/summary`, `/api/anomalies`, `/api/run`, `/api/predict`, and `/health` endpoints. |

---

## 🚀 How to Capture the Screenshots

1. **Start the application:**
   ```bash
   cd 02_renewable_asset_fault_anomaly_copilot
   uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
   ```
2. **Open browser:** Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000).
3. **Capture each tab:** Navigate through the sidebar navigation items (Overview, Solar Assets, Wind Assets, Anomaly Detection, AI Copilot, Model Performance).
4. **Capture API docs:** Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
5. **Save images:** Save each capture to this `02_renewable_asset_fault_anomaly_copilot/screenshots/` directory using the filenames listed in the checklist above.
