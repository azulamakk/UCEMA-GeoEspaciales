#!/usr/bin/env python3
"""
Comprehensive Argentina Regional Analysis
Analyzes each region separately, then combines all results into one comprehensive analysis
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main import WaterStressDetectionSystem
from outputs.prescription_maps import PrescriptionMapGenerator
import folium
import pandas as pd

class ComprehensiveArgentinAnalysis:
    """Manages comprehensive analysis of all Argentina regions"""
    
    def __init__(self):
        self.system = WaterStressDetectionSystem()
        self.prescription_generator = PrescriptionMapGenerator()
        self.results = {}
        
    async def analyze_all_regions(self, start_date: str, end_date: str, crop_type: str = 'soybean'):
        """Analyze all regions separately"""
        print("🇦🇷 COMPREHENSIVE ARGENTINA WATER STRESS ANALYSIS")
        print("=" * 60)
        
        # Get all high-priority regions
        all_regions = self.system.config['study_areas']
        high_priority_regions = [
            name for name, config in all_regions.items() 
            if config.get('priority') == 'high'
        ]
        
        print(f"📊 Analyzing {len(high_priority_regions)} high-priority agricultural regions")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🌾 Crop: {crop_type}")
        print()
        
        successful_analyses = []
        failed_analyses = []
        
        for i, region_name in enumerate(high_priority_regions, 1):
            region_config = all_regions[region_name]
            print(f"[{i}/{len(high_priority_regions)}] 🗺️  Analyzing {region_config['name']}...")
            
            try:
                # Run individual region analysis
                result = await self.system._analyze_single_region(
                    region_name, crop_type, start_date, end_date
                )
                
                # Save individual region results
                self._save_individual_region_result(region_name, result)
                
                self.results[region_name] = result
                successful_analyses.append(region_name)
                
                # Print region summary
                alert = result.get('alerts', {}).get('current_alert', {})
                alert_level = alert.get('alert_level', 'unknown')
                alert_score = alert.get('alert_score', 0)
                
                status_emoji = "🟢" if alert_level == 'normal' else "🟡" if alert_level == 'warning' else "🔴"
                print(f"   {status_emoji} Alert: {alert_level.upper()} (score: {alert_score:.2f})")
                
            except Exception as e:
                print(f"   ❌ FAILED: {str(e)[:100]}...")
                failed_analyses.append(region_name)
                self.results[region_name] = {'error': str(e)}
        
        print(f"\n✅ Completed: {len(successful_analyses)}/{len(high_priority_regions)} regions")
        print(f"❌ Failed: {len(failed_analyses)} regions")
        
        # Create comprehensive analysis
        await self._create_comprehensive_analysis(successful_analyses, start_date, end_date, crop_type)
        
        return self.results
    
    def _save_individual_region_result(self, region_name: str, result: dict):
        """Save individual region analysis"""
        output_dir = Path("outputs") / "regional_analyses"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = output_dir / f"{region_name}_analysis_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"   💾 Saved: {filename}")
    
    async def _create_comprehensive_analysis(self, successful_regions: list, 
                                           start_date: str, end_date: str, crop_type: str):
        """Create comprehensive analysis combining all regions"""
        print("\n🔄 Creating comprehensive national analysis...")
        
        # Collect all regional data
        all_alerts = []
        regional_summaries = {}
        total_area = 0
        
        for region_name in successful_regions:
            result = self.results[region_name]
            region_config = self.system.config['study_areas'][region_name]
            
            # Extract alert data
            if 'alerts' in result and 'current_alert' in result['alerts']:
                alert = result['alerts']['current_alert'].copy()
                alert['region'] = region_name
                alert['region_name'] = region_config['name']
                all_alerts.append(alert)
            
            # Extract regional summary
            regional_summaries[region_name] = {
                'name': region_config['name'],
                'crops': region_config['crops'],
                'alert_level': result.get('alerts', {}).get('current_alert', {}).get('alert_level', 'unknown'),
                'alert_score': result.get('alerts', {}).get('current_alert', {}).get('alert_score', 0),
                'analysis_success': 'error' not in result
            }
        
        # Create national summary
        national_summary = self._create_national_summary(all_alerts, regional_summaries)
        
        # Create comprehensive interactive map
        comprehensive_map = self._create_comprehensive_interactive_map(successful_regions)
        
        # Save comprehensive results
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'comprehensive_argentina',
            'period': {'start': start_date, 'end': end_date},
            'crop_type': crop_type,
            'total_regions_analyzed': len(successful_regions),
            'national_summary': national_summary,
            'regional_summaries': regional_summaries,
            'comprehensive_map_path': comprehensive_map,
            'individual_results': {region: self.results[region] for region in successful_regions}
        }
        
        # Save comprehensive analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comprehensive_file = f"outputs/comprehensive_argentina_analysis_{timestamp}.json"
        
        with open(comprehensive_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"📋 Comprehensive analysis saved: {comprehensive_file}")
        print(f"🗺️  Comprehensive map saved: {comprehensive_map}")
        
        return comprehensive_results
    
    def _create_national_summary(self, all_alerts: list, regional_summaries: dict) -> dict:
        """Create national-level summary statistics"""
        alert_levels = [alert['alert_level'] for alert in all_alerts]
        
        alert_distribution = {
            'critical': alert_levels.count('critical'),
            'warning': alert_levels.count('warning'),
            'normal': alert_levels.count('normal')
        }
        
        # Overall national status
        if alert_distribution['critical'] > 0:
            overall_status = 'critical'
        elif alert_distribution['warning'] > 0:
            overall_status = 'warning'
        else:
            overall_status = 'normal'
        
        # High-risk regions
        high_risk_regions = [
            alert['region_name'] for alert in all_alerts 
            if alert['alert_level'] in ['critical', 'warning']
        ]
        
        # Average alert score
        alert_scores = [alert.get('alert_score', 0) for alert in all_alerts]
        avg_alert_score = sum(alert_scores) / len(alert_scores) if alert_scores else 0
        
        return {
            'overall_status': overall_status,
            'alert_distribution': alert_distribution,
            'high_risk_regions': high_risk_regions,
            'average_alert_score': avg_alert_score,
            'total_successful_regions': len(all_alerts),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _create_comprehensive_interactive_map(self, successful_regions: list) -> str:
        """Create comprehensive interactive map combining all regions"""
        print("🗺️  Creating comprehensive interactive map...")
        
        # Create base map centered on Argentina
        argentina_center = [-34.6, -60.0]
        m = folium.Map(
            location=argentina_center,
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add each region to the map
        colors = {
            'normal': 'green',
            'warning': 'orange', 
            'critical': 'red',
            'unknown': 'gray'
        }
        
        for region_name in successful_regions:
            region_config = self.system.config['study_areas'][region_name]
            result = self.results[region_name]
            
            # Get alert info
            alert = result.get('alerts', {}).get('current_alert', {})
            alert_level = alert.get('alert_level', 'unknown')
            alert_score = alert.get('alert_score', 0)
            
            # Create polygon for region
            geometry = region_config['geometry']
            coords = [[lat, lon] for lon, lat in geometry]
            
            color = colors.get(alert_level, 'gray')
            
            # Create popup with region information
            popup_html = f"""
            <b>{region_config['name']}</b><br>
            Alert Level: <b style="color:{color}">{alert_level.upper()}</b><br>
            Alert Score: {alert_score:.2f}<br>
            Crops: {', '.join(region_config['crops'])}<br>
            Region ID: {region_name}
            """
            
            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fillColor=color,
                fillOpacity=0.3,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{region_config['name']}: {alert_level.upper()}"
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Water Stress Levels</h4>
        <p><span style="color:green;">●</span> Normal</p>
        <p><span style="color:orange;">●</span> Warning</p>
        <p><span style="color:red;">●</span> Critical</p>
        <p><span style="color:gray;">●</span> No Data</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save comprehensive map
        map_file = "outputs/comprehensive_argentina_map.html"
        m.save(map_file)
        
        return map_file

async def main():
    """Main execution function"""
    analyzer = ComprehensiveArgentinAnalysis()
    
    # Use shorter date range for faster processing
    end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=21)).strftime('%Y-%m-%d')  # 2 weeks
    
    try:
        results = await analyzer.analyze_all_regions(start_date, end_date, 'soybean')
        
        print("\n🏁 ANALYSIS COMPLETE!")
        print("📊 Check outputs/regional_analyses/ for individual region results")
        print("🗺️  Check outputs/comprehensive_argentina_map.html for the combined map")
        print("📋 Check outputs/comprehensive_argentina_analysis_*.json for full results")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))