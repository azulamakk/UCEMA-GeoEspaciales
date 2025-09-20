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
import pandas as pd

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
from outputs.statistical_reports import StatisticalReportGenerator
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
        self.report_generator = StatisticalReportGenerator()
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
            'analysis_period_days': 120,  # Extended to 4 months for full growing season
            'prediction_horizon_days': 14,  # Extended prediction horizon
            'alert_threshold': 0.7,
            'data_retention_days': 365,  # Extended retention for seasonal analysis
            'model_retrain_frequency_days': 30,  # Monthly model updates
            'output_directory': './outputs',
            'study_areas': api_config.get_study_area_argentina(),
            'default_study_area': 'buenos_aires_01_northwest',  # Default to manageable agricultural region
            'enable_multi_region': True,  # Enable analysis of multiple regions
            'priority_regions': ['buenos_aires_01_northwest', 'buenos_aires_02_northeast', 'cordoba_01_north', 'santa_fe_01_northwest']
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

    async def run_full_analysis(self, study_area: str = None,
                               crop_type: str = None,
                               start_date: str = None,
                               end_date: str = None,
                               multi_region: bool = None) -> Dict[str, Any]:
        """
        Run complete water stress analysis for study area(s)

        Args:
            study_area: Name of study area from configuration (default: pampas_central)
            crop_type: Type of crop to analyze (default: soybean)
            start_date: Analysis start date (YYYY-MM-DD)
            end_date: Analysis end date (YYYY-MM-DD)
            multi_region: Enable multi-region analysis (default: True)

        Returns:
            Comprehensive analysis results
        """
        try:
            # Set defaults
            study_area = study_area or self.config['default_study_area']
            crop_type = crop_type or self.config['default_crop_type']
            multi_region = multi_region if multi_region is not None else self.config['enable_multi_region']

            # Set default date range for agricultural growing season analysis
            if not end_date:
                # Use a date 14 days ago to ensure data availability
                end_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
            if not start_date:
                # Use full growing season analysis (120 days + 14 days buffer)
                start_date = (datetime.now() - timedelta(days=self.config['analysis_period_days'] + 14)).strftime('%Y-%m-%d')

            # Determine regions to analyze
            if multi_region and study_area == 'all_argentina':
                # Get all 38 Argentina agricultural regions
                all_regions = list(self.config['study_areas'].keys())
                regions_to_analyze = all_regions
                self.logger.info(f"Starting COMPLETE ARGENTINA analysis for ALL {len(regions_to_analyze)} agricultural regions, {crop_type} from {start_date} to {end_date}")
            elif multi_region and study_area in ['buenos_aires_01_northwest', self.config['default_study_area']]:
                regions_to_analyze = self.config['priority_regions']
                self.logger.info(f"Starting multi-region analysis for {len(regions_to_analyze)} priority agricultural regions, {crop_type} from {start_date} to {end_date}")
            else:
                regions_to_analyze = [study_area]
                self.logger.info(f"Starting single-region analysis for {study_area}, {crop_type} from {start_date} to {end_date}")

            # Validate study areas (skip validation for 'all_argentina' as it's a special case)
            if study_area != 'all_argentina':
                for region in regions_to_analyze:
                    if region not in self.config['study_areas']:
                        raise ValueError(f"Unknown study area: {region}")

            # Run analysis for each region
            if len(regions_to_analyze) == 1:
                # Single region analysis
                return await self._analyze_single_region(regions_to_analyze[0], crop_type, start_date, end_date)
            else:
                # Multi-region analysis
                return await self._analyze_multiple_regions(regions_to_analyze, crop_type, start_date, end_date)

        except Exception as e:
            self.logger.error(f"Error in full analysis: {e}")
            raise

    async def _analyze_single_region(self, study_area: str, crop_type: str, 
                                   start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze a single region"""
        try:
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
                'region_info': area_config,
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

            self.logger.info(f"Single region analysis completed successfully. Results saved to {output_path}")

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in single region analysis: {e}")
            raise

    async def _analyze_single_region_no_save(self, study_area: str, crop_type: str, 
                                           start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze a single region without saving individual files (for multi-region analysis)"""
        try:
            area_config = self.config['study_areas'][study_area]
            geometry = area_config['geometry']

            # Step 1: Data acquisition
            data_results = await self.acquire_all_data(geometry, start_date, end_date)

            # Step 2: Data quality validation
            quality_results = self.validate_data_quality(data_results)

            # Step 3: Data processing and feature engineering
            processed_data = self.process_and_engineer_features(data_results, crop_type)

            # Step 4: Time series analysis
            ts_results = self.perform_time_series_analysis(processed_data)

            # Step 5: Machine learning prediction
            ml_results = self.run_ml_predictions(processed_data, crop_type)

            # Step 6: Generate alerts
            alert_results = self.generate_alerts(processed_data, ml_results, crop_type)

            # Step 7: Create prescription maps (in memory only)
            prescription_results = self.create_prescription_maps_no_save(processed_data, geometry, crop_type)

            # Step 8: Data minimization and cleanup
            cleanup_results = self.apply_data_minimization()

            # Compile final results (no file saving)
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'study_area': study_area,
                'crop_type': crop_type,
                'analysis_period': {'start': start_date, 'end': end_date},
                'region_info': area_config,
                'data_acquisition': data_results,
                'data_quality': quality_results,
                'time_series_analysis': ts_results,
                'ml_predictions': ml_results,
                'alerts': alert_results,
                'prescription_maps': prescription_results,
                'data_management': cleanup_results
            }

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error in single region analysis for {study_area}: {e}")
            return {'error': str(e), 'study_area': study_area, 'timestamp': datetime.now().isoformat()}

    async def _analyze_multiple_regions(self, regions: List[str], crop_type: str, 
                                      start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze multiple regions for comprehensive coverage"""
        try:
            region_results = {}
            combined_alerts = []
            combined_prescriptions = {}

            self.logger.info(f"Analyzing {len(regions)} regions: {', '.join(regions)}")

            # Analyze each region
            for i, region in enumerate(regions):
                self.logger.info(f"Analyzing region {i+1}/{len(regions)}: {region}")
                
                try:
                    region_result = await self._analyze_single_region_no_save(region, crop_type, start_date, end_date)
                    region_results[region] = region_result
                    
                    # Collect alerts and prescriptions
                    if 'alerts' in region_result and 'current_alert' in region_result['alerts']:
                        alert = region_result['alerts']['current_alert'].copy()
                        alert['region'] = region
                        alert['region_name'] = self.config['study_areas'][region]['name']
                        combined_alerts.append(alert)
                    
                    if 'prescription_maps' in region_result:
                        combined_prescriptions[region] = region_result['prescription_maps']
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing region {region}: {e}")
                    region_results[region] = {'error': str(e)}

            # Create national summary
            national_summary = self._create_national_summary(region_results, combined_alerts)

            # Create comprehensive maps
            comprehensive_maps = self._create_comprehensive_maps(region_results, regions)
            
            # Create comprehensive prescription map
            comprehensive_prescription = self.prescription_generator.create_comprehensive_prescription_map(
                region_results, crop_type
            )

            # Compile multi-region results
            multi_region_results = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'multi_region',
                'crop_type': crop_type,
                'analysis_period': {'start': start_date, 'end': end_date},
                'regions_analyzed': regions,
                'region_results': region_results,
                'national_summary': national_summary,
                'combined_alerts': combined_alerts,
                'comprehensive_maps': comprehensive_maps,
                'combined_prescriptions': combined_prescriptions,
                'comprehensive_prescription_map': comprehensive_prescription
            }

            # Save consolidated results using new system
            output_paths = self.save_consolidated_analysis_results(multi_region_results)
            multi_region_results['output_paths'] = output_paths

            self.logger.info(f"Multi-region analysis completed successfully. Consolidated results saved to {len(output_paths)} files")

            return multi_region_results

        except Exception as e:
            self.logger.error(f"Error in multi-region analysis: {e}")
            raise

    def _create_national_summary(self, region_results: Dict, combined_alerts: List) -> Dict:
        """Create a national summary from all regional results"""
        try:
            total_regions = len(region_results)
            successful_regions = len([r for r in region_results.values() if 'error' not in r])
            
            # Alert level distribution
            alert_levels = [alert['alert_level'] for alert in combined_alerts]
            alert_distribution = {
                'critical': alert_levels.count('critical'),
                'warning': alert_levels.count('warning'), 
                'normal': alert_levels.count('normal')
            }
            
            # Overall stress level
            if alert_distribution['critical'] > 0:
                overall_status = 'critical'
            elif alert_distribution['warning'] > 0:
                overall_status = 'warning'
            else:
                overall_status = 'normal'
                
            return {
                'total_regions_analyzed': total_regions,
                'successful_analyses': successful_regions,
                'overall_status': overall_status,
                'alert_distribution': alert_distribution,
                'high_priority_regions': [alert['region'] for alert in combined_alerts if alert['alert_level'] == 'critical'],
                'summary_generated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating national summary: {e}")
            return {'error': str(e)}

    def _create_comprehensive_maps(self, region_results: Dict, regions: List) -> Dict:
        """Create comprehensive maps combining all regions"""
        try:
            # This would create Argentina-wide maps
            # For now, return structure for individual region maps
            comprehensive_maps = {
                'argentina_wide_stress_map': f"./outputs/comprehensive_maps/argentina_stress_{datetime.now().strftime('%Y%m%d')}.tif",
                'argentina_wide_prescription_map': f"./outputs/comprehensive_maps/argentina_prescription_{datetime.now().strftime('%Y%m%d')}.tif",
                'regional_maps': {}
            }
            
            for region in regions:
                if region in region_results and 'prescription_maps' in region_results[region]:
                    comprehensive_maps['regional_maps'][region] = region_results[region]['prescription_maps']
                    
            return comprehensive_maps
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive maps: {e}")
            return {'error': str(e)}

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

            # Use default soil data (skip API that's timing out)
            self.logger.info("Using default soil properties (skipping unreliable SoilGrids API)")
            soil_data = self._create_default_soil_data(centroid_lon, centroid_lat)

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

    def _create_default_soil_data(self, longitude: float, latitude: float) -> pd.DataFrame:
        """Create default soil data when API is unavailable"""
        try:
            # Default soil properties for Argentina agricultural regions
            default_properties = {
                'location_id': f"lon_{longitude}_lat_{latitude}",
                'property': ['bdod', 'clay', 'sand', 'silt', 'phh2o', 'soc'],
                'depth_label': ['0-5cm', '5-15cm', '15-30cm'],
                'depth_top_cm': [0, 5, 15],
                'depth_bottom_cm': [5, 15, 30],
                'value': [1400, 25, 45, 30, 6.5, 2.0],  # Typical values for Argentine soils
                'unit': ['g/dm3', '%', '%', '%', 'pH', '%']
            }
            
            # Create DataFrame with repeated values for each depth
            records = []
            for i, prop in enumerate(default_properties['property']):
                for j, depth in enumerate(default_properties['depth_label']):
                    records.append({
                        'location_id': default_properties['location_id'],
                        'property': prop,
                        'depth_label': depth,
                        'depth_top_cm': default_properties['depth_top_cm'][j],
                        'depth_bottom_cm': default_properties['depth_bottom_cm'][j],
                        'value': default_properties['value'][i],
                        'unit': default_properties['unit'][i]
                    })
            
            return pd.DataFrame(records)
            
        except Exception as e:
            self.logger.error(f"Error creating default soil data: {e}")
            return pd.DataFrame()

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
    
    def create_prescription_maps_no_save(self, processed_data: Dict[str, Any],
                                       geometry: List[List[float]], crop_type: str) -> Dict[str, Any]:
        """Create prescription maps for irrigation without saving files (for multi-region analysis)"""
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

            # Create prescription map (in memory only)
            prescription_map = self.prescription_generator.create_prescription_map(
                stress_data, geometry, crop_type
            )

            return {
                'prescription_map': prescription_map,
                'creation_timestamp': datetime.now().isoformat(),
                'coordinates': {
                    'centroid_lon': centroid_lon,
                    'centroid_lat': centroid_lat,
                    'geometry': geometry
                }
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

    def save_analysis_results(self, results: Dict[str, Any], prefix: str = "water_stress_analysis") -> str:
        """Save complete analysis results"""
        try:
            output_dir = Path(self.config['output_directory'])
            output_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{prefix}_{timestamp}.json"
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

    def save_consolidated_analysis_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Save consolidated national analysis results in organized structure"""
        try:
            output_dir = Path(self.config['output_directory'])
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            date_str = datetime.now().strftime('%Y%m%d')
            
            output_paths = {}
            
            # 1. National Analysis File (complete results)
            national_file = output_dir / f"argentina_national_analysis_{timestamp}.json"
            with open(national_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            output_paths['national_analysis'] = str(national_file)
            
            # 2. Consolidated Alerts File
            alerts_data = {
                'timestamp': timestamp,
                'total_regions': len(results.get('regions_analyzed', [])),
                'national_summary': results.get('national_summary', {}),
                'regional_alerts': results.get('combined_alerts', []),
                'alert_distribution': results.get('national_summary', {}).get('alert_distribution', {}),
                'overall_status': results.get('national_summary', {}).get('overall_status', 'unknown')
            }
            alerts_file = output_dir / f"argentina_alerts_consolidated_{date_str}.json"
            with open(alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2, default=str)
            output_paths['alerts_consolidated'] = str(alerts_file)
            
            # 3. Satellite Data Consolidation
            satellite_data = self._consolidate_satellite_data(results.get('region_results', {}))
            satellite_file = output_dir / f"argentina_satellite_data_{date_str}.json"
            with open(satellite_file, 'w') as f:
                json.dump(satellite_data, f, indent=2, default=str)
            output_paths['satellite_data'] = str(satellite_file)
            
            # 4. Weather and Soil Data Consolidation
            weather_soil_data = self._consolidate_weather_soil_data(results.get('region_results', {}))
            weather_soil_file = output_dir / f"argentina_weather_soil_data_{date_str}.json"
            with open(weather_soil_file, 'w') as f:
                json.dump(weather_soil_data, f, indent=2, default=str)
            output_paths['weather_soil_data'] = str(weather_soil_file)
            
            # 5. Prescription Maps Consolidation
            prescriptions_data = self._consolidate_prescription_data(results.get('combined_prescriptions', {}))
            prescriptions_file = output_dir / f"argentina_prescription_maps_{date_str}.json"
            with open(prescriptions_file, 'w') as f:
                json.dump(prescriptions_data, f, indent=2, default=str)
            output_paths['prescription_maps'] = str(prescriptions_file)
            
            # Create organized directory structure
            reports_dir = self._create_organized_directories(output_dir, date_str)
            
            # Generate statistical reports
            national_report = self.report_generator.generate_national_summary_report(results, reports_dir)
            regional_report = self.report_generator.generate_regional_comparison_report(results, reports_dir)
            
            if national_report:
                output_paths['national_summary_report'] = national_report
            if regional_report:
                output_paths['regional_comparison_report'] = regional_report
            
            self.logger.info(f"Consolidated analysis results saved to {len(output_paths)} files including statistical reports")
            return output_paths
            
        except Exception as e:
            self.logger.error(f"Error saving consolidated results: {e}")
            return {}

    def _consolidate_satellite_data(self, region_results: Dict) -> Dict:
        """Consolidate satellite data from all regions"""
        consolidated = {
            'timestamp': datetime.now().isoformat(),
            'total_regions': len(region_results),
            'regions': {},
            'national_statistics': {},
            'vegetation_indices_summary': {}
        }
        
        ndvi_values = []
        ndwi_values = []
        
        for region, data in region_results.items():
            if 'error' not in data and 'data_acquisition' in data:
                sat_data = data['data_acquisition'].get('satellite', {})
                if 'time_series' in sat_data:
                    ts = sat_data['time_series']
                    region_summary = {
                        'region_name': data.get('region_info', {}).get('name', region),
                        'data_points': len(ts) if isinstance(ts, list) else 0,
                        'latest_ndvi': ts[-1].get('NDVI', 0) if ts and isinstance(ts, list) else 0,
                        'latest_ndwi': ts[-1].get('NDWI', 0) if ts and isinstance(ts, list) else 0,
                        'avg_ndvi': sum([d.get('NDVI', 0) for d in ts]) / len(ts) if ts else 0,
                        'avg_ndwi': sum([d.get('NDWI', 0) for d in ts]) / len(ts) if ts else 0
                    }
                    consolidated['regions'][region] = region_summary
                    ndvi_values.append(region_summary['latest_ndvi'])
                    ndwi_values.append(region_summary['latest_ndwi'])
        
        # National statistics
        if ndvi_values:
            consolidated['national_statistics'] = {
                'avg_ndvi': sum(ndvi_values) / len(ndvi_values),
                'min_ndvi': min(ndvi_values),
                'max_ndvi': max(ndvi_values),
                'avg_ndwi': sum(ndwi_values) / len(ndwi_values),
                'min_ndwi': min(ndwi_values),
                'max_ndwi': max(ndwi_values)
            }
        
        return consolidated

    def _consolidate_weather_soil_data(self, region_results: Dict) -> Dict:
        """Consolidate weather and soil data from all regions"""
        consolidated = {
            'timestamp': datetime.now().isoformat(),
            'total_regions': len(region_results),
            'weather_data': {},
            'soil_data': {},
            'climate_summary': {}
        }
        
        temperatures = []
        precipitations = []
        
        for region, data in region_results.items():
            if 'error' not in data and 'data_acquisition' in data:
                # Weather data
                weather_data = data['data_acquisition'].get('weather', {})
                if weather_data:
                    latest_temp = weather_data.get('T2M', 0) if isinstance(weather_data.get('T2M'), (int, float)) else 0
                    latest_precip = weather_data.get('PRECTOTCORR', 0) if isinstance(weather_data.get('PRECTOTCORR'), (int, float)) else 0
                    
                    consolidated['weather_data'][region] = {
                        'region_name': data.get('region_info', {}).get('name', region),
                        'temperature': latest_temp,
                        'precipitation': latest_precip,
                        'evapotranspiration': weather_data.get('EVPTRNS', 0)
                    }
                    temperatures.append(latest_temp)
                    precipitations.append(latest_precip)
                
                # Soil data
                soil_data = data['data_acquisition'].get('soil', {})
                if soil_data:
                    consolidated['soil_data'][region] = {
                        'region_name': data.get('region_info', {}).get('name', region),
                        'properties': soil_data
                    }
        
        # Climate summary
        if temperatures:
            consolidated['climate_summary'] = {
                'avg_temperature': sum(temperatures) / len(temperatures),
                'min_temperature': min(temperatures),
                'max_temperature': max(temperatures),
                'avg_precipitation': sum(precipitations) / len(precipitations),
                'total_precipitation': sum(precipitations)
            }
        
        return consolidated

    def _consolidate_prescription_data(self, combined_prescriptions: Dict) -> Dict:
        """Consolidate prescription map data from all regions"""
        consolidated = {
            'timestamp': datetime.now().isoformat(),
            'total_regions': len(combined_prescriptions),
            'regional_prescriptions': {},
            'national_irrigation_summary': {},
            'water_usage_optimization': {}
        }
        
        total_irrigation_rates = []
        
        for region, prescription_data in combined_prescriptions.items():
            if 'error' not in prescription_data and 'prescription_map' in prescription_data:
                pmap = prescription_data['prescription_map']
                
                # Extract irrigation recommendations
                irrigation_rate = 0
                if 'irrigation_recommendations' in pmap and 'total_irrigation_mm_day' in pmap['irrigation_recommendations']:
                    irrigation_rate = pmap['irrigation_recommendations']['total_irrigation_mm_day']
                
                consolidated['regional_prescriptions'][region] = {
                    'irrigation_rate_mm_day': irrigation_rate,
                    'prescription_zones': pmap.get('zones', {}),
                    'efficiency_metrics': pmap.get('efficiency_metrics', {}),
                    'coordinates': prescription_data.get('coordinates', {})
                }
                
                if irrigation_rate > 0:
                    total_irrigation_rates.append(irrigation_rate)
        
        # National irrigation summary
        if total_irrigation_rates:
            consolidated['national_irrigation_summary'] = {
                'avg_irrigation_rate': sum(total_irrigation_rates) / len(total_irrigation_rates),
                'total_regions_requiring_irrigation': len(total_irrigation_rates),
                'max_irrigation_rate': max(total_irrigation_rates),
                'min_irrigation_rate': min(total_irrigation_rates),
                'total_water_requirement_estimate': sum(total_irrigation_rates) * 1624  # 1624 km² per region
            }
        
        return consolidated

    def _create_organized_directories(self, output_dir: Path, date_str: str) -> Path:
        """Create organized directory structure for reports and maps"""
        # Create directories for future reports and maps
        reports_dir = output_dir / "statistical_reports"
        reports_dir.mkdir(exist_ok=True)
        
        maps_dir = output_dir / "interactive_maps"
        maps_dir.mkdir(exist_ok=True)
        
        return reports_dir

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Water Stress Detection System')
    parser.add_argument('--study-area', default='all_argentina', help='Study area name (default: all_argentina for complete coverage)')
    parser.add_argument('--multi-region', action='store_true', default=True, help='Enable multi-region analysis for comprehensive coverage (default: True)')
    parser.add_argument('--single-region', action='store_true', help='Run single region analysis instead of all Argentina')
    parser.add_argument('--crop-type', default='soybean', help='Crop type')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()
    
    # Override multi-region if single-region is explicitly requested
    if args.single_region:
        args.multi_region = False
        if args.study_area == 'all_argentina':
            args.study_area = 'buenos_aires_01_northwest'  # Default single region

    # Initialize system
    system = WaterStressDetectionSystem(args.config)

    # Run analysis
    try:
        results = asyncio.run(system.run_full_analysis(
            study_area=args.study_area,
            crop_type=args.crop_type,
            start_date=args.start_date,
            end_date=args.end_date,
            multi_region=args.multi_region
        ))

        print(f"Analysis completed successfully!")
        
        # Print summary based on analysis type
        if 'analysis_type' in results and results['analysis_type'] == 'multi_region':
            # Multi-region consolidated output summary
            output_paths = results.get('output_paths', {})
            if output_paths:
                print(f"\n🇦🇷 === CONSOLIDATED ARGENTINA ANALYSIS COMPLETE ===")
                print(f"📊 Generated {len(output_paths)} consolidated files:")
                print(f"   🌍 National Analysis: {output_paths.get('national_analysis', 'N/A')}")
                print(f"   🚨 Alerts Consolidated: {output_paths.get('alerts_consolidated', 'N/A')}")
                print(f"   🛰️  Satellite Data: {output_paths.get('satellite_data', 'N/A')}")
                print(f"   🌦️  Weather/Soil Data: {output_paths.get('weather_soil_data', 'N/A')}")
                print(f"   💧 Prescription Maps: {output_paths.get('prescription_maps', 'N/A')}")
                print(f"\n📈 === STATISTICAL REPORTS GENERATED ===")
                print(f"   📋 National Summary: {output_paths.get('national_summary_report', 'N/A')}")
                print(f"   📊 Regional Comparison: {output_paths.get('regional_comparison_report', 'N/A')}")
            
            # National summary
            if 'national_summary' in results:
                summary = results['national_summary']
                print(f"\n=== NATIONAL SUMMARY ===")
                print(f"Regions analyzed: {summary.get('total_regions_analyzed', 0)}")
                print(f"Successful analyses: {summary.get('successful_analyses', 0)}")
                print(f"Overall status: {summary.get('overall_status', 'unknown').upper()}")
                
                if 'alert_distribution' in summary:
                    dist = summary['alert_distribution']
                    print(f"Alert distribution: Critical: {dist.get('critical', 0)}, Warning: {dist.get('warning', 0)}, Normal: {dist.get('normal', 0)}")
                
                if summary.get('high_priority_regions'):
                    print(f"High priority regions: {', '.join(summary['high_priority_regions'])}")
            
            if 'regions_analyzed' in results:
                print(f"\nRegions analyzed: {len(results['regions_analyzed'])} total")
                print(f"Regional coverage: Buenos Aires (10), Córdoba (8), Santa Fe (8), Entre Ríos (3), La Pampa (3), Northern (5), Test (1)")
                
            print(f"\n📁 Output Structure:")
            print(f"   ├── statistical_reports/     # Future analytical reports")
            print(f"   └── interactive_maps/        # National visualization maps")
        else:
            # Single region summary
            if 'alerts' in results and 'current_alert' in results['alerts']:
                alert = results['alerts']['current_alert']
                print(f"Current alert level: {alert['alert_level']}")
                print(f"Alert score: {alert['alert_score']:.2f}")
                
            if 'study_area' in results:
                print(f"Study area: {results['study_area']}")
                if 'region_info' in results and 'name' in results['region_info']:
                    print(f"Region: {results['region_info']['name']}")

    except Exception as e:
        print(f"Error running analysis: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
