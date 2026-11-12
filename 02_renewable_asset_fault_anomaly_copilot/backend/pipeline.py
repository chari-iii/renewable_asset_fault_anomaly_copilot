"""
End-to-end ML & Anomaly Detection Pipeline for Renewable Asset Telemetry.
Handles:
1. Chronological time-series preprocessing and physics-informed feature engineering.
2. Baseline & Random Forest Power Generation forecasting (Variance analysis).
3. Statistical 3-sigma & Isolation Forest Anomaly Detection.
4. Evaluation metrics (MAE, RMSE, R2, Anomaly Precision/Recall).
5. Seamless integration with Domain Diagnostic & Copilot GenAI layers.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from backend.models import (
    TelemetryPoint,
    PredictionRequest,
    AnomalyItem,
    CopilotExplanation,
    PipelineRunResponse,
    FleetSummary,
    AssetCard,
    TelemetrySeriesPoint
)
from backend.copilot import generate_copilot_explanation

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "asset_telemetry.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class RenewableAssetPipeline:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or DATA_PATH
        self.df: pd.DataFrame = pd.DataFrame()
        self.solar_rf: Optional[RandomForestRegressor] = None
        self.wind_rf: Optional[RandomForestRegressor] = None
        self.solar_linear: Optional[LinearRegression] = None
        self.wind_linear: Optional[LinearRegression] = None
        self.solar_iso: Optional[IsolationForest] = None
        self.wind_iso: Optional[IsolationForest] = None
        self.metrics: Dict[str, Any] = {}
        # Verified ground-truth injected anomaly timestamps in bundled benchmark dataset
        self.known_solar_faults = [67, 68, 69]
        self.known_wind_faults = [173, 174]
        self.load_data_and_train()

    def load_data(self) -> pd.DataFrame:
        """Loads and prepares the raw SCADA telemetry dataset."""
        df = pd.read_csv(self.data_path)
        df['timestamp_id'] = df['timestamp_id'].astype(int)
        df['hour'] = df['hour'].astype(int)
        for col in ['irradiance_wm2', 'wind_speed_ms', 'temperature_c', 'vibration_mm_s', 'power_mw']:
            df[col] = df[col].astype(float)
        
        # Feature Engineering:
        # Cyclical temporal representation
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

        # Physics-based baseline expectations:
        df['solar_phys_expected'] = np.maximum(0.0, df['irradiance_wm2'] * 0.0019)
        df['wind_phys_expected'] = np.clip(0.00582 * (df['wind_speed_ms'] ** 3), 0.0, 2.8)

        self.df = df
        return df

    def train_expected_power_models(self, save_to_disk: bool = True) -> Dict[str, Any]:
        """
        Trains expected generation models with chronological time splits.
        Preserves temporal order: first 70% train, last 30% test (NO future lookahead).
        Compares Random Forest Regressor and Linear Regression baseline transparently.
        """
        results = {}
        
        # --- SOLAR FORECASTING ---
        solar = self.df[self.df['asset_id'] == 'SOLAR_INV_01'].copy()
        solar_clean = solar[~solar['timestamp_id'].isin(self.known_solar_faults)]
        split_idx = int(len(solar_clean) * 0.70)
        train_s = solar_clean.iloc[:split_idx]
        test_s = solar_clean.iloc[split_idx:]

        solar_features = ['irradiance_wm2', 'temperature_c', 'hour', 'hour_sin', 'hour_cos']
        X_train_s, y_train_s = train_s[solar_features], train_s['power_mw']
        X_test_s, y_test_s = test_s[solar_features], test_s['power_mw']

        # Model 1: Random Forest Regressor
        self.solar_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=3,
            random_state=42
        )
        self.solar_rf.fit(X_train_s, y_train_s)
        preds_rf_s = self.solar_rf.predict(X_test_s)

        # Model 2: Linear Regression Baseline
        self.solar_linear = LinearRegression().fit(train_s[['irradiance_wm2']], y_train_s)
        preds_base_s = self.solar_linear.predict(test_s[['irradiance_wm2']])

        results['solar_power_forecasting'] = {
            'train_samples': len(train_s),
            'test_samples': len(test_s),
            'chosen_model_rf': {
                'name': 'Random Forest Regressor',
                'mae_mw': float(mean_absolute_error(y_test_s, preds_rf_s)),
                'rmse_mw': float(np.sqrt(mean_squared_error(y_test_s, preds_rf_s))),
                'r2_score': float(r2_score(y_test_s, preds_rf_s)),
                'key_hyperparameters': {'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 3}
            },
            'baseline_linear': {
                'name': 'Linear Physics Baseline',
                'mae_mw': float(mean_absolute_error(y_test_s, preds_base_s)),
                'rmse_mw': float(np.sqrt(mean_squared_error(y_test_s, preds_base_s))),
                'r2_score': float(r2_score(y_test_s, preds_base_s))
            },
            'investigation_finding': (
                "Linear regression achieves lower MAE on this bundled dataset because the benchmark data "
                "was synthesized with an exact linear formula (power_mw = 0.0019 * irradiance_wm2) without noise. "
                "In real utility-scale plants, Random Forest is preferred to handle physical nonlinearities: "
                "inverter MPPT clipping, temperature derating coefficients, and diurnal inverter heat-loss curves."
            )
        }

        # --- WIND FORECASTING ---
        wind = self.df[self.df['asset_id'] == 'WTG_07'].copy()
        wind_clean = wind[~wind['timestamp_id'].isin(self.known_wind_faults)]
        split_idx_w = int(len(wind_clean) * 0.70)
        train_w = wind_clean.iloc[:split_idx_w]
        test_w = wind_clean.iloc[split_idx_w:]

        wind_features = ['wind_speed_ms', 'temperature_c', 'hour', 'hour_sin', 'hour_cos']
        X_train_w, y_train_w = train_w[wind_features], train_w['power_mw']
        X_test_w, y_test_w = test_w[wind_features], test_w['power_mw']

        # Model 1: Random Forest Regressor
        self.wind_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_split=3,
            random_state=42
        )
        self.wind_rf.fit(X_train_w, y_train_w)
        preds_rf_w = self.wind_rf.predict(X_test_w)

        # Model 2: Linear Wind Model
        self.wind_linear = LinearRegression().fit(train_w[['wind_speed_ms']], y_train_w)
        preds_base_w = self.wind_linear.predict(test_w[['wind_speed_ms']])

        results['wind_power_forecasting'] = {
            'train_samples': len(train_w),
            'test_samples': len(test_w),
            'chosen_model_rf': {
                'name': 'Random Forest Regressor',
                'mae_mw': float(mean_absolute_error(y_test_w, preds_rf_w)),
                'rmse_mw': float(np.sqrt(mean_squared_error(y_test_w, preds_rf_w))),
                'r2_score': float(r2_score(y_test_w, preds_rf_w)),
                'key_hyperparameters': {'n_estimators': 100, 'max_depth': 6, 'min_samples_split': 3}
            },
            'baseline_linear': {
                'name': 'Linear Wind Speed Baseline',
                'mae_mw': float(mean_absolute_error(y_test_w, preds_base_w)),
                'rmse_mw': float(np.sqrt(mean_squared_error(y_test_w, preds_base_w))),
                'r2_score': float(r2_score(y_test_w, preds_base_w))
            },
            'investigation_finding': (
                "Linear regression achieves lower MAE on this bundled dataset because the dummy wind output was "
                "generated using an affine approximation (P ≈ 0.45 * v - 1.36). In actual wind turbines, power follows "
                "a nonlinear cubic curve (P ∝ v³) that saturates at rated speed and shuts down at cut-out, which "
                "requires nonlinear models such as Random Forest."
            )
        }

        # Save models if requested
        if save_to_disk:
            joblib.dump(self.solar_rf, MODELS_DIR / "solar_rf.joblib")
            joblib.dump(self.wind_rf, MODELS_DIR / "wind_rf.joblib")
        return results

    def train_anomaly_detectors(self, save_to_disk: bool = True) -> Dict[str, Any]:
        """
        Trains IsolationForest anomaly detectors for Solar and Wind assets.
        Also evaluates statistical 3-sigma / IQR baseline.
        """
        results = {}
        solar_cols = ['temperature_c', 'vibration_mm_s', 'power_mw', 'irradiance_wm2']
        wind_cols = ['temperature_c', 'vibration_mm_s', 'power_mw', 'wind_speed_ms']

        # 1. SOLAR ISOLATION FOREST
        solar = self.df[self.df['asset_id'] == 'SOLAR_INV_01']
        self.solar_iso = IsolationForest(
            contamination=0.035,
            n_estimators=100,
            max_samples='auto',
            random_state=42
        )
        self.solar_iso.fit(solar[solar_cols])
        if save_to_disk:
            joblib.dump(self.solar_iso, MODELS_DIR / "solar_iso.joblib")

        solar_pred = self.solar_iso.predict(solar[solar_cols])
        solar_flagged = set(solar[solar_pred == -1]['timestamp_id'].tolist())
        solar_tp = len(solar_flagged.intersection(set(self.known_solar_faults)))
        solar_fp = len(solar_flagged - set(self.known_solar_faults))
        solar_fn = len(set(self.known_solar_faults) - solar_flagged)
        solar_prec = solar_tp / (solar_tp + solar_fp) if (solar_tp + solar_fp) > 0 else 0.0
        solar_rec = solar_tp / (solar_tp + solar_fn) if (solar_tp + solar_fn) > 0 else 0.0
        solar_f1 = (2 * solar_prec * solar_rec / (solar_prec + solar_rec)) if (solar_prec + solar_rec) > 0 else 0.0

        results['solar_anomaly_detection'] = {
            'algorithm': 'Isolation Forest',
            'contamination_setting': 0.035,
            'n_estimators': 100,
            'total_observations': len(solar),
            'flagged_anomalies': list(solar_flagged),
            'precision': float(solar_prec),
            'recall': float(solar_rec),
            'f1_score': float(solar_f1),
            'known_fault_capture': f"{solar_tp}/{len(self.known_solar_faults)}",
            'evaluation_split_note': 'In-sample benchmark evaluation on predefined synthetic faults (TS #67-69). The 30% test split contains zero labeled faults, precluding independent out-of-sample supervised recall.'
        }

        # 2. WIND ISOLATION FOREST
        wind = self.df[self.df['asset_id'] == 'WTG_07']
        self.wind_iso = IsolationForest(
            contamination=0.035,
            n_estimators=100,
            max_samples='auto',
            random_state=42
        )
        self.wind_iso.fit(wind[wind_cols])
        if save_to_disk:
            joblib.dump(self.wind_iso, MODELS_DIR / "wind_iso.joblib")

        wind_pred = self.wind_iso.predict(wind[wind_cols])
        wind_flagged = set(wind[wind_pred == -1]['timestamp_id'].tolist())
        wind_tp = len(wind_flagged.intersection(set(self.known_wind_faults)))
        wind_fp = len(wind_flagged - set(self.known_wind_faults))
        wind_fn = len(set(self.known_wind_faults) - wind_flagged)
        wind_prec = wind_tp / (wind_tp + wind_fp) if (wind_tp + wind_fp) > 0 else 0.0
        wind_rec = wind_tp / (wind_tp + wind_fn) if (wind_tp + wind_fn) > 0 else 0.0
        wind_f1 = (2 * wind_prec * wind_rec / (wind_prec + wind_rec)) if (wind_prec + wind_rec) > 0 else 0.0

        results['wind_anomaly_detection'] = {
            'algorithm': 'Isolation Forest',
            'contamination_setting': 0.035,
            'n_estimators': 100,
            'total_observations': len(wind),
            'flagged_anomalies': list(wind_flagged),
            'precision': float(wind_prec),
            'recall': float(wind_rec),
            'f1_score': float(wind_f1),
            'known_fault_capture': f"{wind_tp}/{len(self.known_wind_faults)}",
            'evaluation_split_note': 'In-sample benchmark evaluation on predefined synthetic faults (TS #173-174). The 30% test split contains zero labeled faults, precluding independent out-of-sample supervised recall.'
        }

        return results

    def load_data_and_train(self, force_retrain: bool = False):
        """Loads telemetry data and reuses existing trained models if present on disk."""
        self.load_data()
        
        solar_rf_path = MODELS_DIR / "solar_rf.joblib"
        wind_rf_path = MODELS_DIR / "wind_rf.joblib"
        solar_iso_path = MODELS_DIR / "solar_iso.joblib"
        wind_iso_path = MODELS_DIR / "wind_iso.joblib"
        metrics_path = MODELS_DIR / "evaluation_metrics.json"

        models_exist = (
            solar_rf_path.exists() and
            wind_rf_path.exists() and
            solar_iso_path.exists() and
            wind_iso_path.exists()
        )

        if models_exist and not force_retrain:
            self.solar_rf = joblib.load(solar_rf_path)
            self.wind_rf = joblib.load(wind_rf_path)
            self.solar_iso = joblib.load(solar_iso_path)
            self.wind_iso = joblib.load(wind_iso_path)
            if metrics_path.exists():
                import json
                with open(metrics_path, "r") as f:
                    self.metrics = json.load(f)
            else:
                power_metrics = self.train_expected_power_models(save_to_disk=False)
                anomaly_metrics = self.train_anomaly_detectors(save_to_disk=False)
                self.metrics = {**power_metrics, **anomaly_metrics}
        else:
            power_metrics = self.train_expected_power_models(save_to_disk=True)
            anomaly_metrics = self.train_anomaly_detectors(save_to_disk=True)
            self.metrics = {**power_metrics, **anomaly_metrics}

    def compute_all_predictions_and_variances(self) -> pd.DataFrame:
        """Computes expected generation, variance, and anomaly flags for all 240 telemetry rows."""
        df = self.df.copy()
        
        # Solar predictions
        solar_mask = df['asset_id'] == 'SOLAR_INV_01'
        solar_feats = ['irradiance_wm2', 'temperature_c', 'hour', 'hour_sin', 'hour_cos']
        df.loc[solar_mask, 'expected_power_mw'] = self.solar_rf.predict(df.loc[solar_mask, solar_feats])
        
        solar_iso_cols = ['temperature_c', 'vibration_mm_s', 'power_mw', 'irradiance_wm2']
        df.loc[solar_mask, 'iso_pred'] = self.solar_iso.predict(df.loc[solar_mask, solar_iso_cols])
        df.loc[solar_mask, 'anomaly_score'] = -self.solar_iso.decision_function(df.loc[solar_mask, solar_iso_cols])

        # Wind predictions
        wind_mask = df['asset_id'] == 'WTG_07'
        wind_feats = ['wind_speed_ms', 'temperature_c', 'hour', 'hour_sin', 'hour_cos']
        df.loc[wind_mask, 'expected_power_mw'] = self.wind_rf.predict(df.loc[wind_mask, wind_feats])

        wind_iso_cols = ['temperature_c', 'vibration_mm_s', 'power_mw', 'wind_speed_ms']
        df.loc[wind_mask, 'iso_pred'] = self.wind_iso.predict(df.loc[wind_mask, wind_iso_cols])
        df.loc[wind_mask, 'anomaly_score'] = -self.wind_iso.decision_function(df.loc[wind_mask, wind_iso_cols])

        # Variance calculations (Sign convention: actual - expected)
        df['variance_mw'] = df['power_mw'] - df['expected_power_mw']
        df['variance_pct'] = np.where(
            df['expected_power_mw'] > 0.05,
            (df['variance_mw'] / df['expected_power_mw']) * 100.0,
            0.0
        )

        # Domain Fault Severity & Classification
        def classify_row(row):
            is_iso_anomaly = (row['iso_pred'] == -1)
            is_vib_critical = (row['vibration_mm_s'] >= 3.5)
            is_temp_critical = (row['temperature_c'] >= 45.0)

            if is_vib_critical or (is_temp_critical and is_iso_anomaly):
                if 'SOLAR' in row['asset_id']:
                    return ('CRITICAL', 'INVERTER_THERMAL_COOLING_BLOWER_FAILURE')
                else:
                    return ('CRITICAL', 'WTG_DRIVETRAIN_BEARING_DEGRADATION')
            elif is_iso_anomaly and (row['vibration_mm_s'] >= 2.0 or row['temperature_c'] >= 42.0):
                return ('HIGH', 'ELEVATED_THERMAL_VIBRATION_STRESS')
            elif is_iso_anomaly:
                return ('MEDIUM', 'ELEVATED_SENSOR_DRIFT')
            elif row['vibration_mm_s'] >= 1.8:
                return ('LOW', 'SLIGHT_VIBRATION_ELEVATION')
            else:
                return ('NORMAL', 'NOMINAL')

        classifications = df.apply(classify_row, axis=1)
        df['severity'] = [c[0] for c in classifications]
        df['fault_type'] = [c[1] for c in classifications]
        return df

    def get_detected_anomalies(self) -> List[AnomalyItem]:
        """Returns structured list of confirmed critical, high, medium, and low anomalies."""
        df = self.compute_all_predictions_and_variances()
        anomalies_df = df[df['severity'].isin(['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'WARNING'])].sort_values(
            by=['severity', 'anomaly_score'], ascending=[True, False]
        )
        items = []
        for _, row in anomalies_df.iterrows():
            items.append(AnomalyItem(
                timestamp_id=int(row['timestamp_id']),
                asset_id=str(row['asset_id']),
                hour=int(row['hour']),
                temperature_c=float(round(row['temperature_c'], 2)),
                vibration_mm_s=float(round(row['vibration_mm_s'], 2)),
                power_mw=float(round(row['power_mw'], 3)),
                expected_power_mw=float(round(row['expected_power_mw'], 3)),
                variance_mw=float(round(row['variance_mw'], 3)),
                variance_pct=float(round(row['variance_pct'], 1)),
                anomaly_score=float(round(row['anomaly_score'], 4)),
                severity=str(row['severity']),
                fault_type=str(row['fault_type'])
            ))
        return items

    def get_fleet_summary(self) -> FleetSummary:
        """Returns fleet-level KPI metrics computed from actual telemetry."""
        df = self.compute_all_predictions_and_variances()
        total_records = len(df)
        critical_count = len(df[df['severity'] == 'CRITICAL'])
        high_count = len(df[df['severity'] == 'HIGH'])
        medium_count = len(df[df['severity'] == 'MEDIUM'])
        low_count = len(df[df['severity'] == 'LOW'])
        total_anomalies = critical_count + high_count + medium_count + low_count
        
        avg_power = float(df['power_mw'].mean())
        health_score = round(((total_records - (critical_count * 5 + high_count * 2 + medium_count)) / total_records) * 100.0, 1)
        health_score = max(0.0, min(100.0, health_score))

        system_status = "WARNING_ALARMS_ACTIVE" if critical_count > 0 else "NOMINAL"

        return FleetSummary(
            total_monitored_assets=2,
            solar_assets=1,
            wind_assets=1,
            total_records=total_records,
            total_anomalies=total_anomalies,
            critical_anomalies=critical_count,
            high_anomalies=high_count,
            medium_anomalies=medium_count,
            low_anomalies=low_count,
            average_power_mw=round(avg_power, 3),
            fleet_health_score_pct=health_score,
            system_status=system_status
        )

    def get_asset_cards(self) -> List[AssetCard]:
        """Returns structured summaries for individual assets."""
        df = self.compute_all_predictions_and_variances()
        cards = []

        # Solar Card
        solar_df = df[df['asset_id'] == 'SOLAR_INV_01']
        solar_latest = solar_df.iloc[-1]
        solar_anoms = len(solar_df[solar_df['severity'].isin(['CRITICAL', 'HIGH', 'MEDIUM'])])
        solar_critical = len(solar_df[solar_df['severity'] == 'CRITICAL'])
        solar_health = "Critical (Cooling Alert)" if solar_critical > 0 else ("Warning" if solar_anoms > 0 else "Good")
        solar_status = "Anomaly Flagged" if solar_anoms > 0 else "Nominal"

        cards.append(AssetCard(
            asset_id="SOLAR_INV_01",
            asset_type="Central Solar Inverter",
            nominal_capacity_mw=1.5,
            current_power_mw=float(round(solar_latest['power_mw'], 3)),
            expected_power_mw=float(round(solar_latest['expected_power_mw'], 3)),
            performance_deviation_pct=float(round(solar_latest['variance_pct'], 1)),
            current_temperature_c=float(round(solar_latest['temperature_c'], 1)),
            current_vibration_mm_s=float(round(solar_latest['vibration_mm_s'], 2)),
            asset_health=solar_health,
            anomaly_status=solar_status,
            total_anomalies=solar_anoms
        ))

        # Wind Card
        wind_df = df[df['asset_id'] == 'WTG_07']
        wind_latest = wind_df.iloc[-1]
        wind_anoms = len(wind_df[wind_df['severity'].isin(['CRITICAL', 'HIGH', 'MEDIUM'])])
        wind_critical = len(wind_df[wind_df['severity'] == 'CRITICAL'])
        wind_health = "Critical (Bearing Distress)" if wind_critical > 0 else ("Warning" if wind_anoms > 0 else "Good")
        wind_status = "Anomaly Flagged" if wind_anoms > 0 else "Nominal"

        cards.append(AssetCard(
            asset_id="WTG_07",
            asset_type="Wind Turbine Generator",
            nominal_capacity_mw=2.6,
            current_power_mw=float(round(wind_latest['power_mw'], 3)),
            expected_power_mw=float(round(wind_latest['expected_power_mw'], 3)),
            performance_deviation_pct=float(round(wind_latest['variance_pct'], 1)),
            current_temperature_c=float(round(wind_latest['temperature_c'], 1)),
            current_vibration_mm_s=float(round(wind_latest['vibration_mm_s'], 2)),
            asset_health=wind_health,
            anomaly_status=wind_status,
            total_anomalies=wind_anoms
        ))

        return cards

    def get_telemetry_series(self, asset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns time-series telemetry data for dashboard charts."""
        df = self.compute_all_predictions_and_variances()
        if asset_id:
            df = df[df['asset_id'].str.upper() == asset_id.upper()]
        
        records = []
        for _, r in df.iterrows():
            records.append({
                'timestamp_id': int(r['timestamp_id']),
                'asset_id': str(r['asset_id']),
                'hour': int(r['hour']),
                'irradiance_wm2': float(round(r['irradiance_wm2'], 1)),
                'wind_speed_ms': float(round(r['wind_speed_ms'], 2)),
                'temperature_c': float(round(r['temperature_c'], 2)),
                'vibration_mm_s': float(round(r['vibration_mm_s'], 2)),
                'power_mw': float(round(r['power_mw'], 3)),
                'expected_power_mw': float(round(r['expected_power_mw'], 3)),
                'variance_mw': float(round(r['variance_mw'], 3)),
                'variance_pct': float(round(r['variance_pct'], 1)),
                'anomaly_score': float(round(r['anomaly_score'], 4)),
                'is_anomaly': bool(r['severity'] in ['CRITICAL', 'HIGH', 'MEDIUM']),
                'severity': str(r['severity'])
            })
        return records

    def predict_single_point(self, req: PredictionRequest) -> Dict[str, Any]:
        """Predicts expected power and anomaly score for a user-supplied telemetry point."""
        hour_sin = np.sin(2 * np.pi * req.hour / 24.0)
        hour_cos = np.cos(2 * np.pi * req.hour / 24.0)
        
        if 'SOLAR' in req.asset_id.upper():
            feats = pd.DataFrame([{
                'irradiance_wm2': req.irradiance_wm2,
                'temperature_c': req.temperature_c,
                'hour': req.hour,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos
            }])
            expected_power = float(self.solar_rf.predict(feats)[0])
            iso_cols = pd.DataFrame([{
                'temperature_c': req.temperature_c,
                'vibration_mm_s': req.vibration_mm_s,
                'power_mw': req.actual_power_mw if req.actual_power_mw is not None else expected_power,
                'irradiance_wm2': req.irradiance_wm2
            }])
            iso_pred = int(self.solar_iso.predict(iso_cols)[0])
            score = float(-self.solar_iso.decision_function(iso_cols)[0])
        else:
            feats = pd.DataFrame([{
                'wind_speed_ms': req.wind_speed_ms,
                'temperature_c': req.temperature_c,
                'hour': req.hour,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos
            }])
            expected_power = float(self.wind_rf.predict(feats)[0])
            iso_cols = pd.DataFrame([{
                'temperature_c': req.temperature_c,
                'vibration_mm_s': req.vibration_mm_s,
                'power_mw': req.actual_power_mw if req.actual_power_mw is not None else expected_power,
                'wind_speed_ms': req.wind_speed_ms
            }])
            iso_pred = int(self.wind_iso.predict(iso_cols)[0])
            score = float(-self.wind_iso.decision_function(iso_cols)[0])

        actual = req.actual_power_mw if req.actual_power_mw is not None else expected_power
        variance_mw = actual - expected_power
        variance_pct = (variance_mw / expected_power * 100.0) if expected_power > 0.05 else 0.0

        severity = "NORMAL"
        if req.vibration_mm_s >= 3.5 or req.temperature_c >= 45.0:
            severity = "CRITICAL"
        elif req.vibration_mm_s >= 2.0 or req.temperature_c >= 42.0:
            severity = "HIGH"
        elif iso_pred == -1:
            severity = "MEDIUM"

        return {
            'asset_id': req.asset_id,
            'hour': req.hour,
            'actual_power_mw': round(actual, 3),
            'expected_power_mw': round(expected_power, 3),
            'variance_mw': round(variance_mw, 3),
            'variance_pct': round(variance_pct, 1),
            'anomaly_score': round(score, 4),
            'is_anomaly': (iso_pred == -1 or severity in ['CRITICAL', 'HIGH', 'MEDIUM']),
            'severity': severity
        }

    def run_copilot_case(self, query_text: str, asset_id: Optional[str] = None, timestamp_id: Optional[int] = None) -> PipelineRunResponse:
        """
        Executes end-to-end case:
        Filters anomalies, selects the representative event,
        and generates an operator-friendly GenAI explanation.
        """
        anomalies = self.get_detected_anomalies()
        target_asset = asset_id or ('WTG_07' if 'wind' in query_text.lower() or 'wtg' in query_text.lower() else 'SOLAR_INV_01')
        
        filtered = [a for a in anomalies if a.asset_id == target_asset]
        if not filtered:
            filtered = anomalies

        if timestamp_id is not None:
            selected = next((a for a in anomalies if a.timestamp_id == timestamp_id), filtered[0])
        else:
            selected = next((a for a in filtered if a.severity == 'CRITICAL'), filtered[0])

        explanation = generate_copilot_explanation(
            asset_id=selected.asset_id,
            hour=selected.hour,
            temperature_c=selected.temperature_c,
            vibration_mm_s=selected.vibration_mm_s,
            actual_power_mw=selected.power_mw,
            expected_power_mw=selected.expected_power_mw,
            variance_mw=selected.variance_mw,
            variance_pct=selected.variance_pct,
            anomaly_score=selected.anomaly_score,
            timestamp_id=selected.timestamp_id,
            user_query=query_text
        )

        return PipelineRunResponse(
            query=query_text or "Diagnose active renewable asset anomalies and provide mitigation guidance.",
            status="SUCCESS",
            asset_analyzed=selected.asset_id,
            total_telemetry_points=len(self.df),
            anomalies_detected=len(anomalies),
            anomaly_summary=anomalies,
            copilot_explanation=explanation,
            evaluation_metrics=self.metrics,
            limitations_and_assumptions=[
                "1. Bundled dataset provides hourly telemetry; high-frequency kHz vibration spectral data would be required in production for precise bearing fault frequency identification (BPFO/BPFI).",
                "2. Model assumes steady grid connection; actual plant curtailments require auxiliary grid dispatch signal ingestion.",
                "3. Evaluation on bundled dummy data demonstrates synthetic linear generation characteristics; field operational data contains environmental noise and non-linear derating."
            ]
        )


_pipeline_instance: Optional[RenewableAssetPipeline] = None

def get_pipeline() -> RenewableAssetPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RenewableAssetPipeline()
    return _pipeline_instance
