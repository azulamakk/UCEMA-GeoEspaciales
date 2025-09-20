#!/usr/bin/env python3
"""
Complete Argentina Water Stress Analysis
Comprehensive analysis of ALL agricultural regions in Argentina
"""
import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import folium
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from main import WaterStressDetectionSystem

class CompleteArgentinaAnalysis:
    """Complete national analysis system"""
    
    def __init__(self):
        self.system = WaterStressDetectionSystem()
        self.results = {}
        
    async def run_complete_analysis(self):
        """Run complete national analysis"""
        print("🇦🇷 COMPLETE ARGENTINA AGRICULTURAL WATER STRESS ANALYSIS")
        print("=" * 70)
        
        # Get all regions organized by priority
        all_regions = self.system.config['study_areas']
        
        high_priority = [name for name, config in all_regions.items() if config.get('priority') == 'high']
        medium_priority = [name for name, config in all_regions.items() if config.get('priority') == 'medium']
        low_priority = [name for name, config in all_regions.items() if config.get('priority') == 'low']
        
        total_regions = len(high_priority) + len(medium_priority) + len(low_priority)
        
        print(f"📊 Total agricultural regions: {total_regions}")
        print(f"   🔴 High priority (Pampas core): {len(high_priority)} regions")
        print(f"   🟡 Medium priority (Extended): {len(medium_priority)} regions") 
        print(f"   🟢 Low priority (Northern): {len(low_priority)} regions")
        
        # Use short date range for speed
        end_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        
        print(f"📅 Analysis period: {start_date} to {end_date}")
        print()
        
        all_regions_list = high_priority + medium_priority + low_priority
        successful_count = 0
        failed_count = 0
        start_time = time.time()
        
        # Process each region
        for i, region_name in enumerate(all_regions_list, 1):
            region_config = all_regions[region_name]
            priority = region_config.get('priority', 'unknown')
            priority_emoji = "🔴" if priority == 'high' else "🟡" if priority == 'medium' else "🟢"
            
            print(f"[{i:2d}/{total_regions}] {priority_emoji} {region_config['name'][:35]:35s} ", end="", flush=True)
            
            region_start_time = time.time()
            
            try:
                # Run analysis for this region
                result = await self.system._analyze_single_region(
                    region_name, 'soybean', start_date, end_date
                )
                
                self.results[region_name] = result
                successful_count += 1
                
                # Extract key info
                alert = result.get('alerts', {}).get('current_alert', {})
                alert_level = alert.get('alert_level', 'unknown')
                alert_score = alert.get('alert_score', 0)
                
                # Display result
                status_emoji = "🟢" if alert_level == 'normal' else "🟡" if alert_level == 'warning' else "🔴"
                region_time = time.time() - region_start_time
                
                print(f"{status_emoji} {alert_level.upper():8s} ({alert_score:.2f}) [{region_time:4.1f}s]")
                
            except Exception as e:
                failed_count += 1
                print(f"❌ FAILED ({str(e)[:20]}...)")
                self.results[region_name] = {'error': str(e)}
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 70)
        print("🏁 COMPLETE ANALYSIS FINISHED!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"✅ Successful: {successful_count}/{total_regions} regions ({successful_count/total_regions*100:.1f}%)")
        print(f"❌ Failed: {failed_count} regions")
        
        # Create comprehensive outputs
        if successful_count > 0:
            print(f"\n📊 Creating comprehensive analysis outputs...")
            
            # Create national map
            map_file = self._create_national_map()
            
            # Create comprehensive summary
            summary_file = self._create_comprehensive_summary(start_date, end_date)
            
            # Create regional reports
            reports_dir = self._create_regional_reports()
            
            print(f"\n🎯 ANALYSIS COMPLETE!")
            print(f"🗺️  National map: {map_file}")
            print(f"📋 Summary report: {summary_file}")
            print(f"📁 Regional reports: {reports_dir}")
            
        return self.results
    
    def _create_national_map(self) -> str:
        """Create comprehensive national water stress map"""
        print("🗺️  Generating national map... ", end="", flush=True)
        
        # Create map centered on Argentina agricultural heartland
        m = folium.Map(
            location=[-32.5, -61.0], 
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Color scheme for alert levels
        colors = {
            'normal': '#2d5a27',      # Dark green
            'warning': '#f39c12',     # Orange
            'critical': '#c0392b',    # Red
            'unknown': '#7f8c8d'      # Gray
        }
        
        successful_regions = 0
        alert_summary = {'normal': 0, 'warning': 0, 'critical': 0, 'unknown': 0}
        
        # Add each region to map
        for region_name, result in self.results.items():
            if 'error' in result:
                continue
                
            region_config = self.system.config['study_areas'][region_name]
            
            # Get analysis results
            alert = result.get('alerts', {}).get('current_alert', {})
            alert_level = alert.get('alert_level', 'unknown')
            alert_score = alert.get('alert_score', 0)
            indicators = alert.get('indicators', {})
            
            alert_summary[alert_level] += 1
            
            # Create region polygon
            geometry = region_config['geometry']
            coords = [[lat, lon] for lon, lat in geometry]
            
            color = colors.get(alert_level, colors['unknown'])
            
            # Create detailed popup
            popup_html = self._create_region_popup(
                region_config, alert_level, alert_score, indicators
            )
            
            # Add polygon to map
            folium.Polygon(
                locations=coords,
                color=color,
                weight=2,
                fillColor=color,
                fillOpacity=0.5,
                popup=folium.Popup(popup_html, max_width=400),
                tooltip=f"{region_config['name']}: {alert_level.upper()} ({alert_score:.2f})"
            ).add_to(m)
            
            successful_regions += 1
        
        # Add comprehensive legend
        self._add_comprehensive_legend(m, alert_summary, successful_regions)
        
        # Add title
        self._add_map_title(m)
        
        # Save map
        map_file = "outputs/argentina_complete_water_stress_map.html"
        m.save(map_file)
        
        print(f"✅ ({successful_regions} regions)")
        return map_file
    
    def _create_region_popup(self, region_config: dict, alert_level: str, 
                           alert_score: float, indicators: dict) -> str:
        """Create detailed popup for region"""
        # Color for alert level
        color = '#2d5a27' if alert_level == 'normal' else '#f39c12' if alert_level == 'warning' else '#c0392b'
        
        popup_html = f"""
        <div style="width: 350px; font-family: Arial, sans-serif;">
            <h4 style="margin-top: 0; color: #2c3e50;">{region_config['name']}</h4>
            
            <div style="background: {color}; color: white; padding: 8px; border-radius: 4px; margin: 5px 0;">
                <b>Alert Level: {alert_level.upper()}</b><br>
                <b>Score: {alert_score:.3f}</b>
            </div>
            
            <p><b>Primary Crops:</b> {', '.join(region_config['crops'])}</p>
            <p><b>Priority:</b> {region_config.get('priority', 'unknown').title()}</p>
            
            <h5 style="margin-bottom: 5px;">Stress Indicators:</h5>
        """
        
        # Add indicator details
        for indicator_name, indicator_data in indicators.items():
            if isinstance(indicator_data, dict) and 'value' in indicator_data:
                value = indicator_data.get('value', 'N/A')
                status = indicator_data.get('status', 'unknown')
                description = indicator_data.get('description', '')
                
                indicator_color = '#2d5a27' if status == 'normal' else '#f39c12' if status == 'warning' else '#c0392b'
                
                popup_html += f"""
                <div style="margin: 3px 0; padding: 3px; background: #f8f9fa; border-radius: 3px;">
                    <b>{indicator_name.upper()}:</b> 
                    <span style="color: {indicator_color}; font-weight: bold;">
                        {value:.3f if isinstance(value, (int, float)) else value}
                    </span>
                    <small style="color: #6c757d;">({status})</small>
                    {f'<br><small style="color: #6c757d;">{description}</small>' if description else ''}
                </div>
                """
        
        popup_html += f"""
            <hr style="margin: 10px 0;">
            <small style="color: #6c757d;">
                Analysis generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </small>
        </div>
        """
        
        return popup_html
    
    def _add_comprehensive_legend(self, map_obj: folium.Map, alert_summary: dict, total_regions: int):
        """Add comprehensive legend to map"""
        legend_html = f'''
        <div style="position: fixed; 
                    bottom: 20px; left: 20px; width: 280px; height: 200px; 
                    background-color: white; border: 2px solid #2c3e50; z-index: 9999; 
                    font-size: 12px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
        
        <h4 style="margin-top: 0; color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px;">
            🇦🇷 Argentina Water Stress Status
        </h4>
        
        <div style="margin: 8px 0;">
            <span style="color: #2d5a27; font-size: 16px;">●</span> 
            <b>Normal:</b> {alert_summary['normal']} regions ({alert_summary['normal']/total_regions*100:.1f}%)
        </div>
        <div style="margin: 8px 0;">
            <span style="color: #f39c12; font-size: 16px;">●</span> 
            <b>Warning:</b> {alert_summary['warning']} regions ({alert_summary['warning']/total_regions*100:.1f}%)
        </div>
        <div style="margin: 8px 0;">
            <span style="color: #c0392b; font-size: 16px;">●</span> 
            <b>Critical:</b> {alert_summary['critical']} regions ({alert_summary['critical']/total_regions*100:.1f}%)
        </div>
        
        <hr style="margin: 10px 0; border: 1px solid #bdc3c7;">
        
        <div style="font-size: 10px; color: #7f8c8d;">
            <b>Total regions analyzed:</b> {total_regions}<br>
            <b>Generated:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}<br>
            <b>Coverage:</b> Complete Argentina agricultural zones
        </div>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_map_title(self, map_obj: folium.Map):
        """Add title to map"""
        title_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 50%; transform: translateX(-50%); width: 500px;
                    background-color: white; border: 2px solid #2c3e50; z-index: 9999; 
                    text-align: center; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
        <h2 style="margin: 0; color: #2c3e50;">🇦🇷 Argentina Agricultural Water Stress</h2>
        <h4 style="margin: 5px 0; color: #7f8c8d;">Complete National Analysis Dashboard</h4>
        <p style="margin: 5px 0; font-size: 12px; color: #95a5a6;">
            Real-time monitoring of water stress across all agricultural regions
        </p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(title_html))
    
    def _create_comprehensive_summary(self, start_date: str, end_date: str) -> str:
        """Create comprehensive analysis summary"""
        print("📋 Generating summary report... ", end="", flush=True)
        
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        failed_results = {k: v for k, v in self.results.items() if 'error' in v}
        
        # Calculate statistics
        alert_levels = []
        alert_scores = []
        province_stats = {}
        
        for region_name, result in successful_results.items():
            alert = result.get('alerts', {}).get('current_alert', {})
            alert_level = alert.get('alert_level', 'unknown')
            alert_score = alert.get('alert_score', 0)
            
            alert_levels.append(alert_level)
            alert_scores.append(alert_score)
            
            # Group by province
            province = region_name.split('_')[0]
            if province not in province_stats:
                province_stats[province] = {'normal': 0, 'warning': 0, 'critical': 0, 'total': 0}
            province_stats[province][alert_level] += 1
            province_stats[province]['total'] += 1
        
        # Create comprehensive summary
        summary = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'complete_argentina_agricultural',
                'period': {'start': start_date, 'end': end_date},
                'crop_focus': 'soybean',
                'total_regions_attempted': len(self.results),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(failed_results)
            },
            
            'national_overview': {
                'overall_status': self._determine_national_status(alert_levels),
                'total_agricultural_area_analyzed': len(successful_results),
                'alert_distribution': {
                    'normal': alert_levels.count('normal'),
                    'warning': alert_levels.count('warning'),
                    'critical': alert_levels.count('critical')
                },
                'statistics': {
                    'average_alert_score': np.mean(alert_scores) if alert_scores else 0,
                    'median_alert_score': np.median(alert_scores) if alert_scores else 0,
                    'max_alert_score': max(alert_scores) if alert_scores else 0,
                    'regions_requiring_attention': len([level for level in alert_levels if level in ['warning', 'critical']]),
                    'percentage_under_stress': len([level for level in alert_levels if level in ['warning', 'critical']]) / len(alert_levels) * 100 if alert_levels else 0
                }
            },
            
            'provincial_breakdown': province_stats,
            
            'high_risk_regions': [
                {
                    'region_id': region_name,
                    'region_name': self.system.config['study_areas'][region_name]['name'],
                    'alert_level': result.get('alerts', {}).get('current_alert', {}).get('alert_level', 'unknown'),
                    'alert_score': result.get('alerts', {}).get('current_alert', {}).get('alert_score', 0),
                    'primary_crops': self.system.config['study_areas'][region_name]['crops']
                }
                for region_name, result in successful_results.items()
                if result.get('alerts', {}).get('current_alert', {}).get('alert_level') in ['warning', 'critical']
            ],
            
            'analysis_quality': {
                'success_rate': len(successful_results) / len(self.results) * 100 if self.results else 0,
                'failed_regions': list(failed_results.keys()),
                'data_completeness': 'high' if len(successful_results) > len(self.results) * 0.8 else 'medium'
            }
        }
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = f"outputs/argentina_complete_analysis_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("✅")
        return summary_file
    
    def _determine_national_status(self, alert_levels: list) -> str:
        """Determine overall national water stress status"""
        if not alert_levels:
            return 'unknown'
        
        critical_pct = alert_levels.count('critical') / len(alert_levels) * 100
        warning_pct = alert_levels.count('warning') / len(alert_levels) * 100
        
        if critical_pct > 15:  # More than 15% critical
            return 'critical'
        elif critical_pct > 5 or warning_pct > 25:  # More than 5% critical or 25% warning
            return 'elevated'
        elif warning_pct > 10:  # More than 10% warning
            return 'moderate'
        else:
            return 'normal'
    
    def _create_regional_reports(self) -> str:
        """Create individual regional reports"""
        print("📁 Creating regional reports... ", end="", flush=True)
        
        reports_dir = Path("outputs/regional_reports")
        reports_dir.mkdir(exist_ok=True)
        
        successful_count = 0
        
        for region_name, result in self.results.items():
            if 'error' not in result:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_file = reports_dir / f"{region_name}_report_{timestamp}.json"
                
                with open(report_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
                successful_count += 1
        
        print(f"✅ ({successful_count} reports)")
        return str(reports_dir)

async def main():
    """Main execution"""
    analyzer = CompleteArgentinaAnalysis()
    
    try:
        results = await analyzer.run_complete_analysis()
        
        print(f"\n🎉 COMPLETE ARGENTINA ANALYSIS FINISHED!")
        print(f"🗺️  Open outputs/argentina_complete_water_stress_map.html for the interactive map")
        print(f"📊 Check outputs/argentina_complete_analysis_*.json for comprehensive results")
        print(f"📁 Individual region reports in outputs/regional_reports/")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))