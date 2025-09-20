#!/usr/bin/env python3
"""
Quick Argentina Water Stress Analysis Script
Runs analysis on all small regions sequentially with short timeframes for fast results
"""
import subprocess
import time
import sys
from datetime import datetime, timedelta

def run_region_analysis(region_name, start_date, end_date):
    """Run analysis for a single region"""
    print(f"\n🌾 Analyzing {region_name} ({start_date} to {end_date})...")
    
    cmd = [
        sys.executable, "main.py",
        "--study-area", region_name,
        "--start-date", start_date,
        "--end-date", end_date
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        if result.returncode == 0:
            print(f"✅ {region_name}: SUCCESS")
            # Extract key info from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "alert level:" in line.lower() or "study area:" in line.lower():
                    print(f"   {line.strip()}")
            return True
        else:
            print(f"❌ {region_name}: FAILED")
            print(f"   Error: {result.stderr.split(chr(10))[-2] if result.stderr else 'Unknown error'}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {region_name}: TIMEOUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ {region_name}: ERROR - {e}")
        return False

def main():
    """Run comprehensive Argentina analysis"""
    print("🇦🇷 ARGENTINA WATER STRESS DETECTION SYSTEM")
    print("=" * 50)
    
    # Define priority regions (small, fast-processing areas)
    priority_regions = [
        'test_area_micro',  # Start with smallest for testing
        'buenos_aires_northwest',
        'buenos_aires_northeast', 
        'cordoba_north',
        'santa_fe_north',
        'buenos_aires_central',
        'cordoba_central',
        'santa_fe_central',
        'entre_rios_north',
        'buenos_aires_southwest',
        'cordoba_south',
        'santa_fe_south',
        'la_pampa_east'
    ]
    
    # Use short date range for fast processing
    end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')  # 1 week period
    
    print(f"📅 Analysis period: {start_date} to {end_date}")
    print(f"🗺️  Total regions to analyze: {len(priority_regions)}")
    print()
    
    successful_regions = []
    failed_regions = []
    start_time = time.time()
    
    for i, region in enumerate(priority_regions, 1):
        print(f"[{i}/{len(priority_regions)}]", end=" ")
        
        region_start_time = time.time()
        success = run_region_analysis(region, start_date, end_date)
        region_time = time.time() - region_start_time
        
        print(f"   Time: {region_time:.1f}s")
        
        if success:
            successful_regions.append(region)
        else:
            failed_regions.append(region)
        
        # Small delay between regions to be nice to APIs
        time.sleep(2)
    
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "=" * 50)
    print("🏁 ANALYSIS COMPLETE")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {len(successful_regions)}/{len(priority_regions)} regions")
    print(f"❌ Failed: {len(failed_regions)} regions")
    
    if successful_regions:
        print(f"\n🎯 Successful regions:")
        for region in successful_regions:
            print(f"   • {region}")
    
    if failed_regions:
        print(f"\n⚠️  Failed regions:")
        for region in failed_regions:
            print(f"   • {region}")
    
    print("\n📊 Results are saved in the outputs/ directory")
    print("🗺️  Check outputs/ for interactive maps and analysis files")

if __name__ == "__main__":
    main()