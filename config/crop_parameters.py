"""
Crop-specific parameters for water stress detection
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import datetime

@dataclass
class CropParameters:
    """Parameters specific to each crop type"""
    name: str
    growing_season_months: Tuple[int, int]  # Start and end month
    critical_growth_stages: List[str]
    ndvi_healthy_range: Tuple[float, float]
    ndwi_stress_threshold: float
    cwsi_stress_threshold: float
    water_requirement_mm_day: float
    rooting_depth_cm: int
    stress_sensitivity_score: float  # 1-10 scale

class CropConfig:
    """Configuration for different crop types in Argentina"""
    
    def __init__(self):
        self.crops = {
            'soybean': CropParameters(
                name='Soybean',
                growing_season_months=(10, 4),  # October to April
                critical_growth_stages=['emergence', 'flowering', 'pod_filling'],
                ndvi_healthy_range=(0.7, 0.9),
                ndwi_stress_threshold=0.3,
                cwsi_stress_threshold=0.4,
                water_requirement_mm_day=5.5,
                rooting_depth_cm=120,
                stress_sensitivity_score=7.5
            ),
            'corn': CropParameters(
                name='Corn',
                growing_season_months=(9, 3),  # September to March
                critical_growth_stages=['emergence', 'tasseling', 'grain_filling'],
                ndvi_healthy_range=(0.75, 0.95),
                ndwi_stress_threshold=0.25,
                cwsi_stress_threshold=0.35,
                water_requirement_mm_day=6.0,
                rooting_depth_cm=150,
                stress_sensitivity_score=8.0
            ),
            'wheat': CropParameters(
                name='Wheat',
                growing_season_months=(5, 12),  # May to December
                critical_growth_stages=['tillering', 'flowering', 'grain_filling'],
                ndvi_healthy_range=(0.6, 0.85),
                ndwi_stress_threshold=0.35,
                cwsi_stress_threshold=0.45,
                water_requirement_mm_day=4.0,
                rooting_depth_cm=100,
                stress_sensitivity_score=6.5
            )
        }
        
        self.growth_stage_timing = {
            'soybean': {
                'emergence': (10, 11),      # October-November
                'flowering': (12, 1),       # December-January
                'pod_filling': (1, 3)       # January-March
            },
            'corn': {
                'emergence': (9, 10),       # September-October
                'tasseling': (12, 1),       # December-January
                'grain_filling': (1, 3)     # January-March
            },
            'wheat': {
                'tillering': (6, 8),        # June-August
                'flowering': (9, 10),       # September-October
                'grain_filling': (10, 12)   # October-December
            }
        }
    
    def get_crop_parameters(self, crop_type: str) -> CropParameters:
        """Get parameters for a specific crop"""
        return self.crops.get(crop_type.lower())
    
    def is_growing_season(self, crop_type: str, date: datetime.date) -> bool:
        """Check if given date is within crop growing season"""
        crop = self.crops.get(crop_type.lower())
        if not crop:
            return False
        
        start_month, end_month = crop.growing_season_months
        current_month = date.month
        
        if start_month <= end_month:
            return start_month <= current_month <= end_month
        else:  # Season spans year boundary
            return current_month >= start_month or current_month <= end_month
    
    def get_current_growth_stage(self, crop_type: str, date: datetime.date) -> str:
        """Determine current growth stage based on date"""
        stages = self.growth_stage_timing.get(crop_type.lower())
        if not stages:
            return 'unknown'
        
        current_month = date.month
        
        for stage, (start_month, end_month) in stages.items():
            if start_month <= end_month:
                if start_month <= current_month <= end_month:
                    return stage
            else:  # Stage spans year boundary
                if current_month >= start_month or current_month <= end_month:
                    return stage
        
        return 'vegetative'
    
    def get_stress_thresholds(self, crop_type: str) -> Dict[str, float]:
        """Get all stress thresholds for a crop"""
        crop = self.crops.get(crop_type.lower())
        if not crop:
            return {}
        
        return {
            'ndvi_min': crop.ndvi_healthy_range[0],
            'ndvi_max': crop.ndvi_healthy_range[1],
            'ndwi_stress': crop.ndwi_stress_threshold,
            'cwsi_stress': crop.cwsi_stress_threshold
        }

# Global crop configuration instance
crop_config = CropConfig()