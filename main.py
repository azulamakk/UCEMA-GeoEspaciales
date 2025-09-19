"""
Main orchestration script for water stress detection system
Coordinates all components for end-to-end water stress monitoring and prediction
"""
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import system components
from config.api_config import api_config
from config.crop_parameters import crop_config
from data_acquisition.satellite_data import SatelliteDataAcquisition
from data_acquisition.weather_data import WeatherDataAcquisition
from data_acquisition.soil_data import SoilDataAcquisition
from processing.vegetation_indices import VegetationIndices
from processing.time_series_analysis import TimeSeriesAnalysis
from modeling.ml_models import WaterStressMLPipeline
from modeling.feature_engineering import FeatureEngineering
from outputs.alerts import WaterStressAlertSystem
from outputs.prescription_maps import PrescriptionMapGenerator
from utils.data_management import DataMinimizationManager, DataQualityController

class WaterStressDetectionSystem:
    """Main system orchestrator for water stress detection"""

    def __init__(self, config_file: Optional[str] = None):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize system components
        self.satellite_data = SatelliteDataAcquisition()
        self.weather_data = WeatherDataAcquisition()
        self.soil_data = SoilDataAcquisition()
        self.vegetation_indices = VegetationIndices()
        self.time_series_analysis = TimeSeriesAnalysis()
        self.ml_pipeline = WaterStressMLPipeline()
        self.feature_engineering = FeatureEngineering()
        self.alert_system = WaterStressAlertSystem()
        self.prescription_generator = PrescriptionMapGenerator()
        self.data_manager = DataMinimizationManager()
        self.quality_controller = DataQualityController()

        # Load configuration
        self.config = self.load_configuration(config_file)

        self.logger.info("Water Stress Detection System initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('water_stress_detection.log'),
                logging.StreamHandler()
            ]
        )

    def load_configuration(self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'default_crop_type': 'soybean',
            'analysis_period_days': 30,
            'prediction_horizon_days': 7,
            'alert_threshold': 0.7,
            'data_retention_days': 30,
            'model_retrain_frequency_days': 7,
            'output_directory': './outputs',
            'study_areas': api_config.get_study_area_argentina()
        }

        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                self.logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                self.logger.warning(f"Error loading config file: {e}. Using default configuration.")

        return default_config

    async def run_full_analysis(self, study_area: str = 'test_area_small',
                               crop_type: str = None,
                               start_date: str = None,
                               end_date: str = None) -> Dict[str, Any]:
        """
        Run complete water stress analysis for a study area

        Args:
            study_area: Name of study area from configuration
            crop_type: Type of crop to analyze
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)

        Returns:
            Comprehensive analysis results
        """
        try:
            crop_type = crop_type or self.config['default_crop_type']

            # Set default date range if not provided
            # Use a safer date range to avoid NASA POWER API issues with recent dates
            if not end_date:
                # Use a date 7 days ago to ensure data availability
                end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            if not start_date:
                # Use 37 days ago (30 days analysis + 7 days buffer)
                start_date = (datetime.now() - timedelta(days=self.config['analysis_period_days'] + 7)).strftime('%Y-%m-%d')

            self.logger.info(f"Starting analysis for {study_area}, {crop_type} from {start_date} to {end_date}")

            # Get study area geometry
            if study_area not in self.config['study_areas']:
                raise ValueError(f"Unknown study area: {study_area}")

            area_config = self.config['study_areas'][study_area]
            geometry = area_config['geometry']

            # Step 1: Data acquisition
            self.logger.info("Step 1: Acquiring satellite, weather, and soil data...")
            data_results = await self.acquire_all_data(geometry, start_date, end_date)

            # Step 2: Data quality validation
            self.logger.info("Step 2: Validating data quality...")
            quality_results = self.validate_data_quality(data_results)

            # Step 3: Data processing and feature engineering
            self.logger.info("Step 3: Processing data and engineering features...")
            processed_data = self.process_and_engineer_features(data_results, crop_type)

            # Step 4: Time series analysis
            self.logger.info("Step 4: Performing time series analysis...")
            ts_results = self.perform_time_series_analysis(processed_data)

            # Step 5: Machine learning prediction
            self.logger.info("Step 5: Running machine learning predictions...")
            ml_results = self.run_ml_predictions(processed_data, crop_type)

            # Step 6: Generate alerts
            self.logger.info("Step 6: Generating water stress alerts...")
            alert_results = self.generate_alerts(processed_data, ml_results, crop_type)

            # Step 7: Create prescription maps
            self.logger.info("Step 7: Creating prescription maps...")
            prescription_results = self.create_prescription_maps(processed_data, geometry, crop_type)

            # Step 8: Data minimization and cleanup
            self.logger.info("Step 8: Applying data minimization...")
            cleanup_results = self.apply_data_minimization()

            # Compile final results
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'study_area': study_area,
                'crop_type': crop_type,
                'analysis_period': {'start': start_date, 'end': end_date},
                'data_acquisition': data_results,
                'data_quality': quality_results,
                'time_series_analysis': ts_results,
                'ml_predictions': ml_results,
                'alerts': alert_results,
                'prescription_maps': prescription_results,
                'data_management': cleanup_results
            }

            # Save results
            output_path = self.save_analysis_results(analysis_results)
            analysis_results['output_path'] = output_path

            self.logger.info(f"Analysis completed successfully. Results saved to {output_path}")

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in full analysis: {e}")
            raise

    async def acquire_all_data(self, geometry: List[List[float]],
                              start_date: str, end_date: str) -> Dict[str, Any]:
        """Acquire data from all sources"""
        try:
            # Calculate centroid for point-based data
            # Handle both polygon (list of coordinates) and coordinate pair formats
            if isinstance(geometry[0], list):
                # Polygon format: [[lon, lat], [lon, lat], ...]
                coords = geometry
            else:
                # Already a list of coordinate pairs
                coords = [geometry]
            
            # Calculate centroid from all coordinates
            all_lons = [coord[0] for coord in coords]
            all_lats = [coord[1] for coord in coords]
            centroid_lon = sum(all_lons) / len(all_lons)
            centroid_lat = sum(all_lats) / len(all_lats)

            # Acquire satellite data
            satellite_data = self.satellite_data.get_sentinel2_data(geometry, start_date, end_date)

            # Acquire weather data
            weather_data = self.weather_data.get_weather_data(centroid_lon, centroid_lat, start_date, end_date)

            # Acquire soil data
            soil_data = self.soil_data.get_soil_data(centroid_lon, centroid_lat)

            # Store data with minimization
            self.data_manager.store_data(satellite_data, 'satellite_data', f'satellite_{start_date}_{end_date}')
            self.data_manager.store_data(weather_data, 'weather_data', f'weather_{start_date}_{end_date}')
            self.data_manager.store_data(soil_data, 'soil_data', f'soil_{centroid_lon}_{centroid_lat}')

            return {
                'satellite': satellite_data,
                'weather': weather_data,
                'soil': soil_data,
                'acquisition_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error acquiring data: {e}")
            raise

    def validate_data_quality(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality of acquired data"""
        try:
            validation_results = []

            # Validate satellite data
            if 'satellite' in data_results and 'time_series' in data_results['satellite']:
                sat_validation = self.quality_controller.validate_satellite_data(
                    data_results['satellite']['time_series']
                )
                validation_results.append(sat_validation)

            # Validate weather data
            if 'weather' in data_results:
                weather_validation = self.quality_controller.validate_weather_data(
                    data_results['weather']
                )
                validation_results.append(weather_validation)

            # Generate quality report
            quality_report = self.quality_controller.generate_quality_report(validation_results)

            return {
                'individual_validations': validation_results,
                'quality_report': quality_report,
                'validation_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return {'error': str(e)}

    def process_and_engineer_features(self, data_results: Dict[str, Any],
                                    crop_type: str) -> Dict[str, Any]:
        """Process raw data and engineer features"""
        try:
            processed_data = {}

            # Process satellite data
            if 'satellite' in data_results and 'time_series' in data_results['satellite']:
                sat_df = data_results['satellite']['time_series']

                # Calculate vegetation indices
                sat_df = self.vegetation_indices.calculate_stress_indicators(sat_df, crop_type)
                sat_df = self.vegetation_indices.calculate_temporal_derivatives(sat_df)
                sat_df = self.vegetation_indices.calculate_anomalies(sat_df)

                processed_data['satellite'] = sat_df

            # Process weather data
            if 'weather' in data_results:
                weather_df = data_results['weather']
                processed_data['weather'] = weather_df

            # Combine datasets
            if 'satellite' in processed_data and 'weather' in processed_data:
                combined_df = self.combine_datasets(processed_data['satellite'], processed_data['weather'])

                # Engineer features
                combined_df = self.feature_engineering.create_temporal_features(combined_df)
                combined_df = self.feature_engineering.create_vegetation_features(combined_df)
                combined_df = self.feature_engineering.create_weather_features(combined_df)

                processed_data['combined'] = combined_df

            # Store processed data
            self.data_manager.store_data(processed_data, 'processed_data', f'processed_{datetime.now().strftime("%Y%m%d")}')

            return processed_data

        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            raise

    def combine_datasets(self, satellite_df, weather_df):
        """Combine satellite and weather datasets"""
        try:
            # Merge on date
            if 'date' in satellite_df.columns and 'date' in weather_df.columns:
                combined_df = satellite_df.merge(weather_df, on='date', how='outer')
                combined_df = combined_df.sort_values('date').reset_index(drop=True)
                return combined_df
            else:
                self.logger.warning("No date column found for merging datasets")
                return satellite_df

        except Exception as e:
            self.logger.error(f"Error combining datasets: {e}")
            return satellite_df

    def perform_time_series_analysis(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive time series analysis"""
        try:
            if 'combined' not in processed_data:
                return {'error': 'No combined dataset available for time series analysis'}

            df = processed_data['combined']

            # Define columns for analysis
            vegetation_columns = ['NDVI', 'GNDVI', 'NDWI_Gao', 'EVI', 'SAVI']
            weather_columns = ['T2M', 'PRECTOTCORR', 'EVPTRNS', 'VPD']

            # Trend detection
            df = self.time_series_analysis.detect_trends(df, vegetation_columns + weather_columns)

            # Anomaly detection
            df = self.time_series_analysis.detect_anomalies(df, vegetation_columns)

            # Water balance analysis
            df = self.time_series_analysis.water_balance_analysis(df)

            # Extreme event analysis
            df = self.time_series_analysis.calculate_extremes(df, vegetation_columns + weather_columns)

            # Generate summary statistics
            summary_stats = self.time_series_analysis.generate_summary_statistics(df, vegetation_columns + weather_columns)

            return {
                'processed_timeseries': df,
                'summary_statistics': summary_stats,
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in time series analysis: {e}")
            return {'error': str(e)}

    def run_ml_predictions(self, processed_data: Dict[str, Any], crop_type: str) -> Dict[str, Any]:
        """Run machine learning predictions"""
        try:
            if 'combined' not in processed_data:
                return {'error': 'No combined dataset available for ML predictions'}

            df = processed_data['combined'].copy()

            # Prepare features
            X, y = self.ml_pipeline.prepare_features(df)

            # Validate model inputs
            validation_result = self.quality_controller.validate_model_inputs(X, y)

            if not validation_result['is_valid']:
                return {'error': 'Invalid model inputs', 'validation': validation_result}

            # Train models
            rf_results = self.ml_pipeline.train_random_forest(X, y, task_type='classification')
            xgb_results = self.ml_pipeline.train_xgboost(X, y, task_type='classification')

            # Make predictions on recent data
            recent_data = df.tail(7)  # Last 7 days
            if len(recent_data) > 0:
                X_recent, _ = self.ml_pipeline.prepare_features(recent_data)

                rf_predictions = self.ml_pipeline.predict_water_stress(X_recent, 'random_forest')
                xgb_predictions = self.ml_pipeline.predict_water_stress(X_recent, 'xgboost')

                predictions = {
                    'random_forest': rf_predictions,
                    'xgboost': xgb_predictions,
                    'dates': recent_data['date'].tolist() if 'date' in recent_data.columns else None
                }
            else:
                predictions = {}

            # Store models
            model_path = f"./models/water_stress_models_{datetime.now().strftime('%Y%m%d')}.joblib"
            self.ml_pipeline.save_models(model_path)

            return {
                'model_performance': {
                    'random_forest': rf_results['performance'],
                    'xgboost': xgb_results['performance']
                },
                'feature_importance': {
                    'random_forest': rf_results['feature_importance'].to_dict(),
                    'xgboost': xgb_results['feature_importance'].to_dict()
                },
                'predictions': predictions,
                'model_path': model_path,
                'training_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in ML predictions: {e}")
            return {'error': str(e)}

    def generate_alerts(self, processed_data: Dict[str, Any],
                       ml_results: Dict[str, Any], crop_type: str) -> Dict[str, Any]:
        """Generate water stress alerts"""
        try:
            if 'combined' not in processed_data:
                return {'error': 'No processed data available for alert generation'}

            df = processed_data['combined']

            # Extract predictions if available
            predictions = None
            probabilities = None

            if 'predictions' in ml_results and 'random_forest' in ml_results['predictions']:
                rf_pred = ml_results['predictions']['random_forest']
                if isinstance(rf_pred, tuple):
                    predictions, probabilities = rf_pred
                else:
                    predictions = rf_pred

            # Generate alert
            alert = self.alert_system.generate_alerts(
                df, crop_type=crop_type,
                model_predictions=predictions,
                prediction_probabilities=probabilities
            )

            # Store alert
            self.data_manager.store_data(alert, 'alert_history', f'alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

            return {
                'current_alert': alert,
                'alert_summary': self.alert_system.get_alert_summary(),
                'generation_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error generating alerts: {e}")
            return {'error': str(e)}

    def create_prescription_maps(self, processed_data: Dict[str, Any],
                               geometry: List[List[float]], crop_type: str) -> Dict[str, Any]:
        """Create prescription maps for irrigation"""
        try:
            if 'combined' not in processed_data:
                return {'error': 'No processed data available for prescription maps'}

            df = processed_data['combined']

            # Create stress data for mapping
            stress_data = df.tail(1).copy()  # Most recent data

            # Add coordinates (simplified - in practice would use spatial data)
            # Handle both polygon (list of coordinates) and coordinate pair formats
            if isinstance(geometry[0], list):
                # Polygon format: [[lon, lat], [lon, lat], ...]
                coords = geometry
            else:
                # Already a list of coordinate pairs
                coords = [geometry]
            
            # Calculate centroid from all coordinates
            all_lons = [coord[0] for coord in coords]
            all_lats = [coord[1] for coord in coords]
            centroid_lon = sum(all_lons) / len(all_lons)
            centroid_lat = sum(all_lats) / len(all_lats)
            stress_data['longitude'] = centroid_lon
            stress_data['latitude'] = centroid_lat

            # Create prescription map
            prescription_map = self.prescription_generator.create_prescription_map(
                stress_data, geometry, crop_type
            )

            # Save maps
            output_dir = f"{self.config['output_directory']}/prescription_maps_{datetime.now().strftime('%Y%m%d')}"
            self.prescription_generator.save_prescription_maps(prescription_map, output_dir)

            # Store prescription data
            self.data_manager.store_data(prescription_map, 'prescription_maps', f'prescription_{datetime.now().strftime("%Y%m%d")}')

            return {
                'prescription_map': prescription_map,
                'output_directory': output_dir,
                'creation_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error creating prescription maps: {e}")
            return {'error': str(e)}

    def apply_data_minimization(self) -> Dict[str, Any]:
        """Apply data minimization and cleanup policies"""
        try:
            # Cleanup expired data
            cleanup_stats = self.data_manager.cleanup_expired_data()

            # Get storage summary
            storage_summary = self.data_manager.get_storage_summary()

            return {
                'cleanup_statistics': cleanup_stats,
                'storage_summary': storage_summary,
                'cleanup_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error in data minimization: {e}")
            return {'error': str(e)}

    def save_analysis_results(self, results: Dict[str, Any]) -> str:
        """Save complete analysis results"""
        try:
            output_dir = Path(self.config['output_directory'])
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"water_stress_analysis_{timestamp}.json"
            filepath = output_dir / filename

            # Remove large objects for JSON serialization
            results_copy = results.copy()

            # Serialize complex objects
            for key in ['data_acquisition', 'time_series_analysis', 'prescription_maps']:
                if key in results_copy and isinstance(results_copy[key], dict):
                    results_copy[key] = self._serialize_for_json(results_copy[key])

            with open(filepath, 'w') as f:
                json.dump(results_copy, f, indent=2, default=str)

            return str(filepath)

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return ""

    def _serialize_for_json(self, obj: Any) -> Any:
        """Serialize objects for JSON output"""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Water Stress Detection System')
    parser.add_argument('--study-area', default='test_area_small', help='Study area name')
    parser.add_argument('--crop-type', default='soybean', help='Crop type')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()

    # Initialize system
    system = WaterStressDetectionSystem(args.config)

    # Run analysis
    try:
        results = asyncio.run(system.run_full_analysis(
            study_area=args.study_area,
            crop_type=args.crop_type,
            start_date=args.start_date,
            end_date=args.end_date
        ))

        print(f"Analysis completed successfully!")
        print(f"Results saved to: {results.get('output_path', 'Unknown')}")

        # Print summary
        if 'alerts' in results and 'current_alert' in results['alerts']:
            alert = results['alerts']['current_alert']
            print(f"Current alert level: {alert['alert_level']}")
            print(f"Alert score: {alert['alert_score']:.2f}")

    except Exception as e:
        print(f"Error running analysis: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
