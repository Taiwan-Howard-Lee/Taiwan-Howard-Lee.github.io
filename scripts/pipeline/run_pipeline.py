#!/usr/bin/env python3
"""
Pipeline Runner - Execute the complete Apple ML Trading data pipeline
Orchestrates data collection, validation, processing, and feature engineering
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data_pipeline.orchestrators.pipeline_orchestrator import DataPipelineOrchestrator

def main():
    """Main pipeline runner with command-line interface"""
    
    print("🚀 Apple ML Trading - Data Pipeline Runner")
    print("=" * 55)
    print("📊 Complete data pipeline orchestration")
    print("🔄 Collection → Validation → Processing → Features → Export")
    print()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apple ML Trading Data Pipeline')
    parser.add_argument('--config', type=str, help='Path to pipeline configuration file')
    parser.add_argument('--stages', nargs='+', help='Specific stages to run', 
                       choices=['data_collection', 'data_validation', 'data_processing', 
                               'feature_engineering', 'data_export'])
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--force', action='store_true', help='Force execution even if recent data exists')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    try:
        orchestrator = DataPipelineOrchestrator(config_path=args.config)
        
        # Show pipeline configuration
        print(f"📋 Pipeline Configuration:")
        print(f"   Name: {orchestrator.config['pipeline']['name']}")
        print(f"   Version: {orchestrator.config['pipeline']['version']}")
        print(f"   Config: {orchestrator.config_path}")
        
        stages_to_run = args.stages or orchestrator.config['pipeline']['stages']
        print(f"   Stages: {', '.join(stages_to_run)}")
        print()
        
        if args.dry_run:
            print("🧪 DRY RUN MODE - No actual execution")
            print(f"📊 Would execute {len(stages_to_run)} stages:")
            for i, stage in enumerate(stages_to_run, 1):
                print(f"   {i}. {stage}")
            print()
            return
        
        # Confirm execution
        if not args.force:
            confirm = input("🚀 Execute pipeline? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Pipeline execution cancelled")
                return
        
        print("🎯 Starting pipeline execution...")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run the pipeline
        result = orchestrator.run_pipeline(stages=stages_to_run)
        
        # Display results
        print("\n" + "=" * 55)
        print("🎉 PIPELINE EXECUTION COMPLETED")
        print("=" * 55)
        
        print(f"📊 Overall Status: {result['status'].upper()}")
        print(f"⏰ Duration: {result.get('metrics', {}).get('duration_seconds', 0):.2f} seconds")
        print(f"✅ Stages Completed: {len(result['stages_completed'])}/{len(stages_to_run)}")
        
        if result['stages_completed']:
            print(f"📋 Completed Stages:")
            for stage in result['stages_completed']:
                print(f"   ✅ {stage}")
        
        if result['errors']:
            print(f"❌ Errors ({len(result['errors'])}):")
            for error in result['errors']:
                print(f"   ❌ {error['stage']}: {error['error']}")
        
        if result.get('metrics'):
            metrics = result['metrics']
            print(f"\n📈 Performance Metrics:")
            print(f"   Success Rate: {metrics.get('success_rate', 0):.2%}")
            print(f"   Stages Failed: {metrics.get('stages_failed', 0)}")
        
        # Show data locations
        print(f"\n📁 Data Locations:")
        print(f"   Raw Data: data/raw/")
        print(f"   Processed Data: data/processed/")
        print(f"   Features: data/features/")
        print(f"   Exports: data/exports/")
        print(f"   Logs: logs/")
        
        # Recommendations
        if result['status'] == 'completed':
            print(f"\n💡 Next Steps:")
            print(f"   📊 View processed data: ls data/processed/")
            print(f"   🔧 Check features: ls data/features/")
            print(f"   📈 Run ML models: python scripts/models/train_model.py")
            print(f"   🌐 Update dashboard: python scripts/dashboard/update_dashboard.py")
        elif result['status'] == 'failed':
            print(f"\n🔧 Troubleshooting:")
            print(f"   📝 Check logs: tail -f logs/pipeline.log")
            print(f"   🔍 Validate data: python src/data_pipeline/validators/data_validator.py")
            print(f"   🧪 Test components: python -m pytest tests/")
        
        return result
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print(f"📝 Check logs for details: logs/pipeline.log")
        return None

def show_pipeline_status():
    """Show current pipeline status and recent runs"""
    print("📊 Pipeline Status Dashboard")
    print("=" * 30)
    
    # Check for recent pipeline states
    logs_dir = project_root / 'logs'
    if logs_dir.exists():
        state_files = list(logs_dir.glob('pipeline_state_*.json'))
        if state_files:
            latest_state = max(state_files, key=lambda x: x.stat().st_mtime)
            
            import json
            with open(latest_state, 'r') as f:
                state = json.load(f)
            
            print(f"🕐 Last Run: {state.get('start_time', 'Unknown')}")
            print(f"📊 Status: {state.get('status', 'Unknown')}")
            print(f"✅ Completed: {len(state.get('stages_completed', []))}")
            print(f"❌ Errors: {len(state.get('errors', []))}")
            
            if state.get('metrics'):
                print(f"⏱️ Duration: {state['metrics'].get('duration_seconds', 0):.2f}s")
                print(f"📈 Success Rate: {state['metrics'].get('success_rate', 0):.2%}")
        else:
            print("📭 No recent pipeline runs found")
    else:
        print("📁 No logs directory found")
    
    # Check data freshness
    data_root = project_root / 'data'
    if data_root.exists():
        print(f"\n📁 Data Status:")
        
        for subdir in ['raw', 'processed', 'features', 'exports']:
            subdir_path = data_root / subdir
            if subdir_path.exists():
                files = list(subdir_path.rglob('*'))
                file_count = len([f for f in files if f.is_file()])
                print(f"   {subdir}: {file_count} files")
            else:
                print(f"   {subdir}: Not found")

if __name__ == "__main__":
    # Check if user wants status instead of running pipeline
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        show_pipeline_status()
    else:
        main()
