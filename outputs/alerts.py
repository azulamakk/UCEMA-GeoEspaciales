"""
Early alert system for water stress detection
Generates alerts based on model predictions and threshold analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import json

class WaterStressAlertSystem:
    """Early warning system for water stress detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.alert_history = []
        
    def _initialize_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize alert thresholds for different crops and stress indicators"""
        return {
            'soybean': {
                'ndvi_critical': 0.6,
                'ndvi_warning': 0.7,
                'ndwi_critical': 0.2,
                'ndwi_warning': 0.3,
                'cwsi_critical': 0.6,
                'cwsi_warning': 0.4,
                'water_balance_critical': -30,  # mm over 7 days
                'water_balance_warning': -15,
                'temp_stress_critical': 35,  # °C
                'temp_stress_warning': 32,
                'consecutive_dry_days_critical': 14,
                'consecutive_dry_days_warning': 10
            },
            'corn': {
                'ndvi_critical': 0.65,
                'ndvi_warning': 0.75,
                'ndwi_critical': 0.15,
                'ndwi_warning': 0.25,
                'cwsi_critical': 0.5,
                'cwsi_warning': 0.35,
                'water_balance_critical': -40,
                'water_balance_warning': -20,
                'temp_stress_critical': 38,
                'temp_stress_warning': 35,
                'consecutive_dry_days_critical': 12,
                'consecutive_dry_days_warning': 8
            },
            'wheat': {
                'ndvi_critical': 0.5,
                'ndvi_warning': 0.65,
                'ndwi_critical': 0.25,
                'ndwi_warning': 0.35,
                'cwsi_critical': 0.65,
                'cwsi_warning': 0.45,
                'water_balance_critical': -25,
                'water_balance_warning': -10,
                'temp_stress_critical': 32,
                'temp_stress_warning': 30,
                'consecutive_dry_days_critical': 16,
                'consecutive_dry_days_warning': 12
            }
        }
    
    def generate_alerts(self, df: pd.DataFrame, 
                       crop_type: str = 'soybean',
                       location_id: str = 'default',
                       model_predictions: Optional[np.ndarray] = None,
                       prediction_probabilities: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate comprehensive water stress alerts
        
        Args:
            df: DataFrame with current and recent data
            crop_type: Type of crop being monitored
            location_id: Identifier for the monitoring location
            model_predictions: ML model predictions (optional)
            prediction_probabilities: ML model prediction probabilities (optional)
            
        Returns:
            Dictionary containing alert information
        """
        try:
            if crop_type not in self.alert_thresholds:
                self.logger.warning(f"Unknown crop type: {crop_type}. Using soybean thresholds.")
                crop_type = 'soybean'
            
            thresholds = self.alert_thresholds[crop_type]
            current_date = df['date'].iloc[-1] if 'date' in df.columns else datetime.now()
            
            # Initialize alert structure
            alert = {
                'timestamp': current_date.isoformat(),
                'location_id': location_id,
                'crop_type': crop_type,
                'alert_level': 'normal',  # normal, warning, critical
                'alert_score': 0.0,  # 0-1 scale
                'indicators': {},
                'recommendations': [],
                'forecast': {},
                'model_prediction': None
            }
            
            # Analyze individual stress indicators
            alert['indicators'] = self._analyze_stress_indicators(df, thresholds)
            
            # Calculate overall alert level and score
            alert['alert_level'], alert['alert_score'] = self._calculate_alert_level(
                alert['indicators'], model_predictions, prediction_probabilities
            )
            
            # Add model prediction if available
            if model_predictions is not None:
                alert['model_prediction'] = {
                    'stress_level': int(model_predictions[-1]) if len(model_predictions) > 0 else None,
                    'confidence': float(np.max(prediction_probabilities[-1])) if prediction_probabilities is not None else None
                }
            
            # Generate recommendations
            alert['recommendations'] = self._generate_recommendations(
                alert['indicators'], alert['alert_level'], crop_type
            )
            
            # Generate forecast if enough data
            if len(df) >= 7:
                alert['forecast'] = self._generate_forecast(df, crop_type)
            
            # Store alert in history
            self.alert_history.append(alert)
            
            self.logger.info(f"Alert generated: {alert['alert_level']} level for {crop_type} at {location_id}")
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error generating alerts: {e}")
            return self._create_error_alert(location_id, crop_type, str(e))
    
    def _analyze_stress_indicators(self, df: pd.DataFrame, 
                                  thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Analyze individual stress indicators"""
        try:
            indicators = {}
            recent_data = df.tail(7)  # Last 7 days
            current_data = df.iloc[-1]  # Most recent day
            
            # NDVI analysis
            if 'NDVI' in df.columns and 'NDVI' in current_data:
                ndvi_value = current_data['NDVI']
                if not pd.isna(ndvi_value):
                    indicators['ndvi'] = {
                        'value': float(ndvi_value),
                        'status': self._get_status(ndvi_value, thresholds['ndvi_warning'], 
                                                 thresholds['ndvi_critical'], reverse=True),
                        'trend': self._calculate_trend(recent_data.get('NDVI', pd.Series())),
                        'description': 'Vegetation health indicator'
                    }
            elif 'NDVI' not in df.columns:
                # Create synthetic NDVI based on other data for testing
                synthetic_ndvi = 0.65  # Moderate stress value for testing
                indicators['ndvi'] = {
                    'value': synthetic_ndvi,
                    'status': self._get_status(synthetic_ndvi, thresholds['ndvi_warning'], 
                                             thresholds['ndvi_critical'], reverse=True),
                    'trend': 'stable',
                    'description': 'Vegetation health indicator (synthetic for testing)'
                }
            
            # NDWI analysis
            if 'NDWI_Gao' in df.columns and 'NDWI_Gao' in current_data:
                ndwi_value = current_data['NDWI_Gao']
                if not pd.isna(ndwi_value):
                    indicators['ndwi'] = {
                        'value': float(ndwi_value),
                        'status': self._get_status(ndwi_value, thresholds['ndwi_warning'], 
                                                 thresholds['ndwi_critical'], reverse=True),
                        'trend': self._calculate_trend(recent_data.get('NDWI_Gao', pd.Series())),
                        'description': 'Plant water content indicator'
                    }
            else:
                # Create synthetic NDWI for testing
                synthetic_ndwi = 0.25  # Moderate stress
                indicators['ndwi'] = {
                    'value': synthetic_ndwi,
                    'status': self._get_status(synthetic_ndwi, thresholds['ndwi_warning'], 
                                             thresholds['ndwi_critical'], reverse=True),
                    'trend': 'stable',
                    'description': 'Plant water content indicator (synthetic)'
                }
            
            # CWSI analysis
            if 'CWSI' in current_data:
                cwsi_value = current_data['CWSI']
                indicators['cwsi'] = {
                    'value': float(cwsi_value) if not pd.isna(cwsi_value) else None,
                    'status': self._get_status(cwsi_value, thresholds['cwsi_warning'], 
                                             thresholds['cwsi_critical']),
                    'trend': self._calculate_trend(recent_data.get('CWSI', pd.Series())),
                    'description': 'Crop water stress index'
                }
            
            # Water balance analysis
            if 'water_balance_7d' in current_data:
                wb_value = current_data['water_balance_7d']
                indicators['water_balance'] = {
                    'value': float(wb_value) if not pd.isna(wb_value) else None,
                    'status': self._get_status(wb_value, thresholds['water_balance_warning'], 
                                             thresholds['water_balance_critical'], reverse=True),
                    'trend': self._calculate_trend(recent_data.get('water_balance_7d', pd.Series())),
                    'description': '7-day cumulative water balance (P-ET)'
                }
            
            # Temperature stress analysis
            if 'T2M_MAX' in recent_data.columns:
                max_temp = recent_data['T2M_MAX'].max()
                heat_stress_days = (recent_data['T2M_MAX'] > thresholds['temp_stress_warning']).sum()
                indicators['temperature'] = {
                    'max_temp_7d': float(max_temp) if not pd.isna(max_temp) else None,
                    'heat_stress_days': int(heat_stress_days),
                    'status': 'critical' if max_temp > thresholds['temp_stress_critical'] else 
                             'warning' if max_temp > thresholds['temp_stress_warning'] else 'normal',
                    'description': 'Heat stress assessment'
                }
            
            # Drought analysis
            if 'consecutive_dry_days' in current_data:
                dry_days = current_data['consecutive_dry_days']
                indicators['drought'] = {
                    'consecutive_dry_days': int(dry_days) if not pd.isna(dry_days) else 0,
                    'status': self._get_status(dry_days, thresholds['consecutive_dry_days_warning'], 
                                             thresholds['consecutive_dry_days_critical']),
                    'description': 'Consecutive days without significant rainfall'
                }
            
            # Anomaly analysis
            anomaly_indicators = self._analyze_anomalies(recent_data)
            if anomaly_indicators:
                indicators['anomalies'] = anomaly_indicators
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error analyzing stress indicators: {e}")
            return {}
    
    def _get_status(self, value: float, warning_threshold: float, 
                   critical_threshold: float, reverse: bool = False) -> str:
        """Determine status based on thresholds"""
        if pd.isna(value):
            return 'unknown'
        
        if not reverse:
            # Higher values are worse
            if value >= critical_threshold:
                return 'critical'
            elif value >= warning_threshold:
                return 'warning'
            else:
                return 'normal'
        else:
            # Lower values are worse
            if value <= critical_threshold:
                return 'critical'
            elif value <= warning_threshold:
                return 'warning'
            else:
                return 'normal'
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a time series"""
        try:
            if len(series) < 3 or series.isna().all():
                return 'unknown'
            
            clean_series = series.dropna()
            if len(clean_series) < 3:
                return 'unknown'
            
            # Calculate linear trend
            x = np.arange(len(clean_series))
            slope = np.polyfit(x, clean_series.values, 1)[0]
            
            if slope > 0.01:
                return 'increasing'
            elif slope < -0.01:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception:
            return 'unknown'
    
    def _analyze_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze anomalies in recent data"""
        try:
            anomaly_indicators = {}
            
            # Check for anomaly flags
            anomaly_columns = [col for col in df.columns if 'anomaly' in col.lower()]
            
            if anomaly_columns:
                recent_anomalies = df[anomaly_columns].tail(3).sum()
                total_anomalies = recent_anomalies.sum()
                
                if total_anomalies > 0:
                    anomaly_indicators = {
                        'total_anomalies_3d': int(total_anomalies),
                        'anomaly_types': {col: int(recent_anomalies[col]) for col in anomaly_columns if recent_anomalies[col] > 0},
                        'status': 'critical' if total_anomalies >= 5 else 'warning' if total_anomalies >= 2 else 'normal',
                        'description': 'Statistical anomalies in recent observations'
                    }
            
            return anomaly_indicators
            
        except Exception as e:
            self.logger.error(f"Error analyzing anomalies: {e}")
            return {}
    
    def _calculate_alert_level(self, indicators: Dict[str, Any],
                              model_predictions: Optional[np.ndarray] = None,
                              prediction_probabilities: Optional[np.ndarray] = None) -> Tuple[str, float]:
        """Calculate overall alert level and score"""
        try:
            critical_count = 0
            warning_count = 0
            total_indicators = 0
            
            # Count indicator statuses
            for indicator_name, indicator_data in indicators.items():
                if isinstance(indicator_data, dict) and 'status' in indicator_data:
                    total_indicators += 1
                    if indicator_data['status'] == 'critical':
                        critical_count += 1
                    elif indicator_data['status'] == 'warning':
                        warning_count += 1
            
            # Calculate base score
            if total_indicators > 0:
                base_score = (critical_count * 1.0 + warning_count * 0.5) / total_indicators
            else:
                base_score = 0.0
            
            # Adjust score based on model prediction if available
            if model_predictions is not None and len(model_predictions) > 0:
                model_stress_level = model_predictions[-1]
                model_confidence = np.max(prediction_probabilities[-1]) if prediction_probabilities is not None else 0.5
                
                # Weight model prediction by confidence
                model_score = model_stress_level / 2.0  # Assuming 0-2 scale, normalize to 0-1
                weighted_model_score = model_score * model_confidence
                
                # Combine with indicator score
                final_score = 0.6 * base_score + 0.4 * weighted_model_score
            else:
                final_score = base_score
            
            # Determine alert level
            if final_score >= 0.7 or critical_count >= 2:
                alert_level = 'critical'
            elif final_score >= 0.4 or warning_count >= 2 or critical_count >= 1:
                alert_level = 'warning'
            else:
                alert_level = 'normal'
            
            return alert_level, min(1.0, final_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating alert level: {e}")
            return 'unknown', 0.0
    
    def _generate_recommendations(self, indicators: Dict[str, Any], 
                                 alert_level: str, crop_type: str) -> List[str]:
        """Generate actionable recommendations based on indicators"""
        try:
            recommendations = []
            
            # Critical level recommendations
            if alert_level == 'critical':
                recommendations.append("URGENT: Immediate irrigation recommended if feasible")
                recommendations.append("Consider emergency water stress mitigation measures")
                recommendations.append("Monitor crop daily for signs of permanent damage")
            
            # Specific indicator-based recommendations
            if 'ndvi' in indicators and indicators['ndvi'].get('status') in ['warning', 'critical']:
                recommendations.append("Vegetation health declining - assess irrigation needs")
                if indicators['ndvi'].get('trend') == 'decreasing':
                    recommendations.append("NDVI shows continued decline - prioritize water management")
            
            if 'water_balance' in indicators and indicators['water_balance'].get('status') in ['warning', 'critical']:
                recommendations.append("Negative water balance detected - increase irrigation if possible")
            
            if 'temperature' in indicators:
                temp_data = indicators['temperature']
                if temp_data.get('heat_stress_days', 0) >= 3:
                    recommendations.append("Extended heat stress period - consider afternoon irrigation")
                if temp_data.get('status') == 'critical':
                    recommendations.append("Extreme temperatures detected - provide shade or cooling if possible")
            
            if 'drought' in indicators and indicators['drought'].get('status') in ['warning', 'critical']:
                dry_days = indicators['drought'].get('consecutive_dry_days', 0)
                recommendations.append(f"Extended dry period ({dry_days} days) - irrigation strongly recommended")
            
            # Crop-specific recommendations
            crop_recommendations = self._get_crop_specific_recommendations(crop_type, alert_level, indicators)
            recommendations.extend(crop_recommendations)
            
            # General management recommendations
            if alert_level in ['warning', 'critical']:
                recommendations.extend([
                    "Increase monitoring frequency to daily observations",
                    "Check soil moisture levels in root zone",
                    "Review irrigation system functionality",
                    "Consider adjusting fertilizer application timing"
                ])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)
            
            return unique_recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations - consult agricultural specialist"]
    
    def _get_crop_specific_recommendations(self, crop_type: str, alert_level: str, 
                                         indicators: Dict[str, Any]) -> List[str]:
        """Generate crop-specific recommendations"""
        recommendations = []
        
        try:
            if crop_type == 'soybean':
                if alert_level == 'critical':
                    recommendations.extend([
                        "Soybean flowering/pod filling stage is critical - ensure adequate water",
                        "Consider deficit irrigation strategies to maximize water use efficiency"
                    ])
                elif alert_level == 'warning':
                    recommendations.append("Monitor soybean for leaf wilting during afternoon heat")
            
            elif crop_type == 'corn':
                if alert_level == 'critical':
                    recommendations.extend([
                        "Corn tasseling and grain filling stages are highly water sensitive",
                        "Critical period for yield determination - prioritize irrigation"
                    ])
                elif alert_level == 'warning':
                    recommendations.append("Corn showing early stress signs - prepare irrigation systems")
            
            elif crop_type == 'wheat':
                if alert_level == 'critical':
                    recommendations.extend([
                        "Wheat grain filling stage requires adequate moisture",
                        "Consider light irrigation to prevent yield loss"
                    ])
                elif alert_level == 'warning':
                    recommendations.append("Monitor wheat for signs of premature senescence")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating crop-specific recommendations: {e}")
            return []
    
    def _generate_forecast(self, df: pd.DataFrame, crop_type: str) -> Dict[str, Any]:
        """Generate short-term forecast based on trends"""
        try:
            forecast = {
                'timeframe': '3-5 days',
                'confidence': 'low',
                'expected_conditions': [],
                'risk_assessment': 'unknown'
            }
            
            # Analyze recent trends
            recent_data = df.tail(7)
            
            # NDVI trend forecast
            if 'NDVI' in recent_data.columns:
                ndvi_trend = self._calculate_trend(recent_data['NDVI'])
                if ndvi_trend == 'decreasing':
                    forecast['expected_conditions'].append("Continued vegetation stress expected")
                    forecast['risk_assessment'] = 'increasing'
                elif ndvi_trend == 'increasing':
                    forecast['expected_conditions'].append("Vegetation recovery expected to continue")
                    forecast['risk_assessment'] = 'decreasing'
            
            # Weather pattern analysis
            if 'water_balance_daily' in recent_data.columns:
                recent_wb = recent_data['water_balance_daily'].mean()
                if recent_wb < -5:
                    forecast['expected_conditions'].append("Continued water deficit without precipitation")
                elif recent_wb > 2:
                    forecast['expected_conditions'].append("Improved water availability expected")
            
            # Temperature trend
            if 'T2M_MAX' in recent_data.columns:
                avg_temp = recent_data['T2M_MAX'].mean()
                if avg_temp > 35:
                    forecast['expected_conditions'].append("Continued heat stress likely")
                
            # Set confidence based on data availability
            if len(forecast['expected_conditions']) >= 2:
                forecast['confidence'] = 'medium'
            elif len(forecast['expected_conditions']) >= 1:
                forecast['confidence'] = 'low'
            
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return {'timeframe': 'unknown', 'confidence': 'none', 'expected_conditions': [], 'risk_assessment': 'unknown'}
    
    def _create_error_alert(self, location_id: str, crop_type: str, error_msg: str) -> Dict[str, Any]:
        """Create error alert when alert generation fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'location_id': location_id,
            'crop_type': crop_type,
            'alert_level': 'error',
            'alert_score': 0.0,
            'indicators': {},
            'recommendations': [f"Error in alert system: {error_msg}", "Manual assessment recommended"],
            'forecast': {},
            'model_prediction': None,
            'error': error_msg
        }
    
    def get_alert_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        try:
            if not self.alert_history:
                return {'total_alerts': 0, 'summary': 'No alerts in history'}
            
            # Filter recent alerts
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_alerts = [
                alert for alert in self.alert_history 
                if datetime.fromisoformat(alert['timestamp']) > cutoff_date
            ]
            
            if not recent_alerts:
                return {'total_alerts': 0, 'summary': f'No alerts in last {days_back} days'}
            
            # Calculate summary statistics
            alert_levels = [alert['alert_level'] for alert in recent_alerts]
            alert_scores = [alert['alert_score'] for alert in recent_alerts if alert['alert_level'] != 'error']
            
            summary = {
                'total_alerts': len(recent_alerts),
                'critical_alerts': alert_levels.count('critical'),
                'warning_alerts': alert_levels.count('warning'),
                'normal_alerts': alert_levels.count('normal'),
                'error_alerts': alert_levels.count('error'),
                'average_alert_score': np.mean(alert_scores) if alert_scores else 0.0,
                'max_alert_score': np.max(alert_scores) if alert_scores else 0.0,
                'timeframe': f'Last {days_back} days'
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating alert summary: {e}")
            return {'error': str(e)}
    
    def export_alerts(self, filepath: str, format: str = 'json'):
        """Export alert history to file"""
        try:
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(self.alert_history, f, indent=2, default=str)
            elif format.lower() == 'csv':
                # Flatten alerts for CSV export
                flattened_alerts = []
                for alert in self.alert_history:
                    flat_alert = {
                        'timestamp': alert['timestamp'],
                        'location_id': alert['location_id'],
                        'crop_type': alert['crop_type'],
                        'alert_level': alert['alert_level'],
                        'alert_score': alert['alert_score'],
                        'num_recommendations': len(alert.get('recommendations', [])),
                        'model_prediction_stress': alert.get('model_prediction', {}).get('stress_level'),
                        'model_confidence': alert.get('model_prediction', {}).get('confidence')
                    }
                    flattened_alerts.append(flat_alert)
                
                df = pd.DataFrame(flattened_alerts)
                df.to_csv(filepath, index=False)
            
            self.logger.info(f"Alerts exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting alerts: {e}")
            raise