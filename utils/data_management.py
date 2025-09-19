"""
Data minimization and quality control utilities
Ensures compliance with data minimization principles and maintains data quality
"""
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import hashlib
import json
from pathlib import Path

class DataMinimizationManager:
    """Manages data storage with minimization principles"""
    
    def __init__(self, max_storage_days: int = 30, storage_path: str = './data'):
        self.logger = logging.getLogger(__name__)
        self.max_storage_days = max_storage_days
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Define essential vs. temporary data categories
        self.essential_data_types = {
            'model_parameters',
            'prescription_maps',
            'alert_history',
            'crop_parameters',
            'validation_results'
        }
        
        self.temporary_data_types = {
            'raw_satellite_data',
            'intermediate_calculations',
            'temp_weather_data',
            'debug_outputs'
        }
    
    def store_data(self, data: Any, data_type: str, 
                   identifier: str, metadata: Optional[Dict] = None) -> str:
        """
        Store data with automatic categorization and lifecycle management
        
        Args:
            data: Data to store
            data_type: Type of data (affects retention policy)
            identifier: Unique identifier for the data
            metadata: Optional metadata about the data
            
        Returns:
            Storage path or identifier
        """
        try:
            # Create storage record
            storage_record = {
                'identifier': identifier,
                'data_type': data_type,
                'timestamp': datetime.now().isoformat(),
                'is_essential': data_type in self.essential_data_types,
                'retention_days': None if data_type in self.essential_data_types else self.max_storage_days,
                'metadata': metadata or {},
                'data_hash': self._calculate_data_hash(data)
            }
            
            # Determine storage approach based on data type
            if data_type in self.essential_data_types:
                storage_path = self._store_essential_data(data, identifier, storage_record)
            else:
                storage_path = self._store_temporary_data(data, identifier, storage_record)
            
            # Log storage action
            self.logger.info(f"Stored {data_type} data: {identifier}")
            
            # Update storage index
            self._update_storage_index(storage_record, storage_path)
            
            return storage_path
            
        except Exception as e:
            self.logger.error(f"Error storing data: {e}")
            raise
    
    def _store_essential_data(self, data: Any, identifier: str, 
                             record: Dict) -> str:
        """Store essential data with permanent retention"""
        essential_dir = self.storage_path / 'essential'
        essential_dir.mkdir(exist_ok=True)
        
        # Store with compression and metadata
        file_path = essential_dir / f"{identifier}.npz"
        
        if isinstance(data, pd.DataFrame):
            # Store DataFrame efficiently
            data_dict = {
                'data': data.to_records(index=False),
                'columns': data.columns.tolist(),
                'dtypes': data.dtypes.to_dict()
            }
            np.savez_compressed(file_path, **data_dict, metadata=record)
        elif isinstance(data, dict):
            # Store dictionary data
            np.savez_compressed(file_path, data=data, metadata=record)
        else:
            # Store array-like data
            np.savez_compressed(file_path, data=data, metadata=record)
        
        return str(file_path)
    
    def _store_temporary_data(self, data: Any, identifier: str, 
                             record: Dict) -> str:
        """Store temporary data with automatic cleanup"""
        temp_dir = self.storage_path / 'temporary'
        temp_dir.mkdir(exist_ok=True)
        
        # Add timestamp to filename for cleanup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = temp_dir / f"{identifier}_{timestamp}.npz"
        
        # Store with minimal metadata
        if isinstance(data, pd.DataFrame):
            data.to_pickle(file_path.with_suffix('.pkl'))
        else:
            np.savez_compressed(file_path, data=data, metadata=record)
        
        return str(file_path)
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash for data integrity checking"""
        try:
            if isinstance(data, pd.DataFrame):
                data_str = data.to_string()
            elif isinstance(data, np.ndarray):
                data_str = str(data.tobytes())
            elif isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True, default=str)
            else:
                data_str = str(data)
            
            return hashlib.md5(data_str.encode()).hexdigest()
            
        except Exception:
            return 'hash_unavailable'
    
    def _update_storage_index(self, record: Dict, storage_path: str):
        """Update storage index for tracking"""
        index_file = self.storage_path / 'storage_index.json'
        
        # Load existing index
        if index_file.exists():
            with open(index_file, 'r') as f:
                index = json.load(f)
        else:
            index = {'records': []}
        
        # Add new record
        record['storage_path'] = storage_path
        index['records'].append(record)
        
        # Save updated index
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2, default=str)
    
    def cleanup_expired_data(self) -> Dict[str, Any]:
        """Remove expired temporary data"""
        try:
            cleanup_stats = {
                'files_removed': 0,
                'space_freed_mb': 0,
                'errors': []
            }
            
            index_file = self.storage_path / 'storage_index.json'
            if not index_file.exists():
                return cleanup_stats
            
            # Load storage index
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            current_time = datetime.now()
            updated_records = []
            
            for record in index['records']:
                should_remove = False
                
                # Check if data has expired
                if record.get('retention_days') is not None:
                    storage_time = datetime.fromisoformat(record['timestamp'])
                    expiry_time = storage_time + timedelta(days=record['retention_days'])
                    
                    if current_time > expiry_time:
                        should_remove = True
                
                if should_remove:
                    try:
                        file_path = Path(record['storage_path'])
                        if file_path.exists():
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            cleanup_stats['files_removed'] += 1
                            cleanup_stats['space_freed_mb'] += file_size / (1024 * 1024)
                            
                            self.logger.info(f"Removed expired data: {record['identifier']}")
                    except Exception as e:
                        cleanup_stats['errors'].append(f"Error removing {record['identifier']}: {str(e)}")
                else:
                    updated_records.append(record)
            
            # Update index with remaining records
            index['records'] = updated_records
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2, default=str)
            
            self.logger.info(f"Cleanup completed: {cleanup_stats['files_removed']} files removed, "
                           f"{cleanup_stats['space_freed_mb']:.2f} MB freed")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {'error': str(e)}
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of current storage usage"""
        try:
            index_file = self.storage_path / 'storage_index.json'
            if not index_file.exists():
                return {'total_records': 0, 'storage_empty': True}
            
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            records = index['records']
            
            # Calculate statistics
            total_size_mb = 0
            data_type_counts = {}
            essential_count = 0
            temporary_count = 0
            
            for record in records:
                try:
                    file_path = Path(record['storage_path'])
                    if file_path.exists():
                        total_size_mb += file_path.stat().st_size / (1024 * 1024)
                    
                    data_type = record['data_type']
                    data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
                    
                    if record.get('is_essential', False):
                        essential_count += 1
                    else:
                        temporary_count += 1
                        
                except Exception:
                    continue
            
            summary = {
                'total_records': len(records),
                'total_size_mb': round(total_size_mb, 2),
                'essential_data_count': essential_count,
                'temporary_data_count': temporary_count,
                'data_type_breakdown': data_type_counts,
                'last_cleanup': self._get_last_cleanup_time()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating storage summary: {e}")
            return {'error': str(e)}
    
    def _get_last_cleanup_time(self) -> Optional[str]:
        """Get timestamp of last cleanup operation"""
        try:
            log_file = self.storage_path / 'cleanup.log'
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        return lines[-1].strip()
            return None
        except:
            return None

class DataQualityController:
    """Controls data quality and validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.quality_metrics = {}
        
    def validate_satellite_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate satellite data quality"""
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'quality_score': 1.0,
                'metrics': {}
            }
            
            # Check for required columns
            required_columns = ['NDVI', 'NDWI_Gao', 'date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                validation_results['errors'].append(f"Missing required columns: {missing_columns}")
                validation_results['is_valid'] = False
                return validation_results
            
            # Check data ranges
            if 'NDVI' in df.columns:
                ndvi_out_of_range = ((df['NDVI'] < -1) | (df['NDVI'] > 1)).sum()
                if ndvi_out_of_range > 0:
                    validation_results['warnings'].append(f"{ndvi_out_of_range} NDVI values out of range [-1, 1]")
                    validation_results['quality_score'] *= (1 - ndvi_out_of_range / len(df) * 0.5)
            
            if 'NDWI_Gao' in df.columns:
                ndwi_out_of_range = ((df['NDWI_Gao'] < -1) | (df['NDWI_Gao'] > 1)).sum()
                if ndwi_out_of_range > 0:
                    validation_results['warnings'].append(f"{ndwi_out_of_range} NDWI values out of range [-1, 1]")
                    validation_results['quality_score'] *= (1 - ndwi_out_of_range / len(df) * 0.5)
            
            # Check for missing data
            missing_data_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            validation_results['metrics']['missing_data_percentage'] = missing_data_pct
            
            if missing_data_pct > 50:
                validation_results['errors'].append(f"Excessive missing data: {missing_data_pct:.1f}%")
                validation_results['is_valid'] = False
            elif missing_data_pct > 20:
                validation_results['warnings'].append(f"High missing data: {missing_data_pct:.1f}%")
                validation_results['quality_score'] *= (1 - missing_data_pct / 100 * 0.3)
            
            # Check temporal continuity
            if 'date' in df.columns and len(df) > 1:
                df_sorted = df.sort_values('date')
                date_gaps = df_sorted['date'].diff().dt.days
                max_gap = date_gaps.max()
                
                validation_results['metrics']['max_temporal_gap_days'] = max_gap
                
                if max_gap > 30:
                    validation_results['warnings'].append(f"Large temporal gap: {max_gap} days")
                    validation_results['quality_score'] *= 0.9
            
            # Check for outliers
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            outlier_counts = {}
            
            for col in numeric_columns:
                if col in ['NDVI', 'NDWI_Gao', 'EVI', 'SAVI']:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    outliers = ((df[col] < Q1 - 3 * IQR) | (df[col] > Q3 + 3 * IQR)).sum()
                    outlier_counts[col] = outliers
                    
                    if outliers > len(df) * 0.1:  # More than 10% outliers
                        validation_results['warnings'].append(f"High outlier count in {col}: {outliers}")
                        validation_results['quality_score'] *= 0.95
            
            validation_results['metrics']['outlier_counts'] = outlier_counts
            
            # Final quality assessment
            if validation_results['quality_score'] < 0.5:
                validation_results['errors'].append("Overall data quality too low")
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating satellite data: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'quality_score': 0.0,
                'metrics': {}
            }
    
    def validate_weather_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate weather data quality"""
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'quality_score': 1.0,
                'metrics': {}
            }
            
            # Check temperature reasonableness
            if 'T2M' in df.columns:
                temp_min, temp_max = df['T2M'].min(), df['T2M'].max()
                validation_results['metrics']['temperature_range'] = (temp_min, temp_max)
                
                if temp_min < -40 or temp_max > 60:
                    validation_results['warnings'].append(f"Extreme temperatures: {temp_min}°C to {temp_max}°C")
                    validation_results['quality_score'] *= 0.9
            
            # Check precipitation reasonableness
            if 'PRECTOTCORR' in df.columns:
                precip_max = df['PRECTOTCORR'].max()
                validation_results['metrics']['max_daily_precipitation'] = precip_max
                
                if precip_max > 200:  # 200mm in a day is extreme
                    validation_results['warnings'].append(f"Extreme precipitation: {precip_max} mm/day")
                    validation_results['quality_score'] *= 0.95
                
                # Check for negative precipitation
                negative_precip = (df['PRECTOTCORR'] < 0).sum()
                if negative_precip > 0:
                    validation_results['errors'].append(f"{negative_precip} negative precipitation values")
                    validation_results['is_valid'] = False
            
            # Check data completeness
            required_weather_vars = ['T2M', 'PRECTOTCORR']
            missing_vars = [var for var in required_weather_vars if var not in df.columns]
            
            if missing_vars:
                validation_results['warnings'].append(f"Missing weather variables: {missing_vars}")
                validation_results['quality_score'] *= 0.8
            
            # Check temporal consistency
            if 'date' in df.columns and len(df) > 1:
                temporal_gaps = self._check_temporal_gaps(df['date'])
                validation_results['metrics']['temporal_gaps'] = temporal_gaps
                
                if temporal_gaps['max_gap_days'] > 5:
                    validation_results['warnings'].append(f"Weather data gaps up to {temporal_gaps['max_gap_days']} days")
                    validation_results['quality_score'] *= 0.9
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating weather data: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'quality_score': 0.0,
                'metrics': {}
            }
    
    def validate_model_inputs(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate machine learning model inputs"""
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'quality_score': 1.0,
                'metrics': {}
            }
            
            # Check feature matrix
            if X.empty:
                validation_results['errors'].append("Empty feature matrix")
                validation_results['is_valid'] = False
                return validation_results
            
            # Check for infinite values
            inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                validation_results['errors'].append(f"{inf_count} infinite values in features")
                validation_results['is_valid'] = False
            
            # Check for excessive missing values
            missing_pct = X.isnull().sum().sum() / (len(X) * len(X.columns)) * 100
            validation_results['metrics']['missing_data_percentage'] = missing_pct
            
            if missing_pct > 30:
                validation_results['errors'].append(f"Excessive missing data: {missing_pct:.1f}%")
                validation_results['is_valid'] = False
            elif missing_pct > 10:
                validation_results['warnings'].append(f"High missing data: {missing_pct:.1f}%")
                validation_results['quality_score'] *= 0.9
            
            # Check target variable
            if len(y) != len(X):
                validation_results['errors'].append("Feature and target length mismatch")
                validation_results['is_valid'] = False
            
            # Check target distribution
            if pd.api.types.is_numeric_dtype(y):
                target_stats = {
                    'mean': y.mean(),
                    'std': y.std(),
                    'min': y.min(),
                    'max': y.max(),
                    'unique_values': len(y.unique())
                }
                validation_results['metrics']['target_statistics'] = target_stats
                
                if target_stats['unique_values'] == 1:
                    validation_results['errors'].append("Target variable has no variation")
                    validation_results['is_valid'] = False
            
            # Check feature correlations
            numeric_features = X.select_dtypes(include=[np.number])
            if len(numeric_features.columns) > 1:
                corr_matrix = numeric_features.corr()
                high_corr = (corr_matrix.abs() > 0.95).sum().sum() - len(corr_matrix)  # Exclude diagonal
                
                validation_results['metrics']['highly_correlated_pairs'] = high_corr
                
                if high_corr > len(numeric_features.columns) * 0.5:
                    validation_results['warnings'].append(f"Many highly correlated features: {high_corr}")
                    validation_results['quality_score'] *= 0.8
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating model inputs: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'quality_score': 0.0,
                'metrics': {}
            }
    
    def _check_temporal_gaps(self, dates: pd.Series) -> Dict[str, Any]:
        """Check for gaps in temporal data"""
        try:
            sorted_dates = dates.sort_values()
            gaps = sorted_dates.diff().dt.days
            
            return {
                'max_gap_days': gaps.max(),
                'avg_gap_days': gaps.mean(),
                'gap_count': (gaps > 1).sum(),
                'total_span_days': (sorted_dates.max() - sorted_dates.min()).days
            }
            
        except Exception:
            return {'max_gap_days': 0, 'avg_gap_days': 0, 'gap_count': 0, 'total_span_days': 0}
    
    def generate_quality_report(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_validations': len(validation_results),
                'passed_validations': sum(1 for r in validation_results if r.get('is_valid', False)),
                'failed_validations': sum(1 for r in validation_results if not r.get('is_valid', True)),
                'average_quality_score': np.mean([r.get('quality_score', 0) for r in validation_results]),
                'common_issues': {},
                'recommendations': []
            }
            
            # Identify common issues
            all_warnings = []
            all_errors = []
            
            for result in validation_results:
                all_warnings.extend(result.get('warnings', []))
                all_errors.extend(result.get('errors', []))
            
            # Count issue frequencies
            warning_counts = {}
            error_counts = {}
            
            for warning in all_warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1
            
            for error in all_errors:
                error_counts[error] = error_counts.get(error, 0) + 1
            
            report['common_issues'] = {
                'warnings': dict(sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                'errors': dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
            # Generate recommendations
            if report['average_quality_score'] < 0.7:
                report['recommendations'].append("Overall data quality is low - review data collection processes")
            
            if 'missing data' in str(all_warnings + all_errors).lower():
                report['recommendations'].append("Address missing data through improved collection or interpolation")
            
            if 'outlier' in str(all_warnings + all_errors).lower():
                report['recommendations'].append("Implement outlier detection and correction procedures")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {'error': str(e)}

class DataPrivacyManager:
    """Manages data privacy and anonymization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def anonymize_location_data(self, df: pd.DataFrame, 
                               precision: float = 0.01) -> pd.DataFrame:
        """Anonymize location data by reducing precision"""
        try:
            df_anon = df.copy()
            
            location_columns = ['longitude', 'latitude', 'lon', 'lat']
            
            for col in location_columns:
                if col in df_anon.columns:
                    # Round to specified precision
                    df_anon[col] = np.round(df_anon[col] / precision) * precision
            
            self.logger.info(f"Anonymized location data with {precision} degree precision")
            return df_anon
            
        except Exception as e:
            self.logger.error(f"Error anonymizing location data: {e}")
            return df
    
    def remove_identifying_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove potentially identifying metadata"""
        try:
            sensitive_keys = [
                'user_id', 'farm_id', 'owner_name', 'contact_info',
                'exact_coordinates', 'property_address', 'field_id'
            ]
            
            cleaned_data = {}
            
            for key, value in data.items():
                if key.lower() not in [s.lower() for s in sensitive_keys]:
                    if isinstance(value, dict):
                        cleaned_data[key] = self.remove_identifying_metadata(value)
                    else:
                        cleaned_data[key] = value
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error removing identifying metadata: {e}")
            return data