#!/usr/bin/env python3
"""
Codebase Reorganization Script
Moves files to the new organized structure and cleans up duplicates
"""

import os
import shutil
from pathlib import Path
import json

def reorganize_codebase():
    """Reorganize the codebase into the new structure"""

    print("🏗️ Apple ML Trading - Codebase Reorganization")
    print("=" * 50)
    print("📁 Moving files to organized structure...")
    print()

    project_root = Path.cwd()

    # File movements
    movements = [
        # Move collection scripts
        {
            'files': [
                'run_continuous_collection.py',
                'run_enhanced_collection.py',
                'compare_collection_strategies.py',
                'test_continuous_collector.py',
                'test_polygon_collector.py',
                'test_polygon_api.py'
            ],
            'destination': 'scripts/data_collection/',
            'description': 'Data collection scripts'
        },

        # Move documentation
        {
            'files': [
                'README_backup.md',
                'CODEBASE_REORGANIZATION_PLAN.md',
                'DATA_DIVERSITY_STRATEGY.md',
                'QUICK_START.md'
            ],
            'destination': 'docs/user_guides/',
            'description': 'Documentation files'
        }
    ]
    
    # Create necessary directories
    directories_to_create = [
        'src/data_pipeline/collectors',
        'src/data_pipeline/processors', 
        'src/data_pipeline/validators',
        'src/data_pipeline/orchestrators',
        'config/environments',
        'config/pipelines',
        'config/models',
        'scripts/data_collection',
        'scripts/pipeline',
        'scripts/deployment',
        'tests/unit',
        'tests/integration',
        'tests/data',
        'docs/api',
        'docs/pipeline',
        'docs/user_guides',
        'data/raw/polygon',
        'data/raw/trading_economics',
        'data/raw/sessions',
        'data/processed',
        'data/features',
        'data/models',
        'data/exports',
        'data/validation_reports',
        'data/processing_reports',
        'logs',
        'notebooks'
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories_to_create:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print(f"\n📦 Moving files...")
    
    # Process each movement group
    for movement in movements:
        print(f"\n📋 {movement['description']}:")
        destination = project_root / movement['destination']
        destination.mkdir(parents=True, exist_ok=True)
        
        files_moved = 0
        
        # Handle specific files
        if 'source_files' in movement:
            for source_file in movement['source_files']:
                source_path = project_root / source_file
                if source_path.exists():
                    dest_path = destination / source_path.name
                    try:
                        shutil.move(str(source_path), str(dest_path))
                        print(f"   ✅ {source_file} → {movement['destination']}")
                        files_moved += 1
                    except Exception as e:
                        print(f"   ❌ Failed to move {source_file}: {e}")
        
        # Handle pattern-based files
        if 'source_patterns' in movement:
            for pattern in movement['source_patterns']:
                if '*' in pattern:
                    # Handle glob patterns
                    pattern_path = project_root / pattern.replace('*', '')
                    if pattern_path.parent.exists():
                        for file_path in pattern_path.parent.glob(pattern_path.name):
                            if file_path.is_file():
                                dest_path = destination / file_path.name
                                try:
                                    shutil.move(str(file_path), str(dest_path))
                                    print(f"   ✅ {file_path.name} → {movement['destination']}")
                                    files_moved += 1
                                except Exception as e:
                                    print(f"   ❌ Failed to move {file_path.name}: {e}")
                else:
                    # Handle specific files
                    source_path = project_root / pattern
                    if source_path.exists():
                        dest_path = destination / source_path.name
                        try:
                            shutil.move(str(source_path), str(dest_path))
                            print(f"   ✅ {pattern} → {movement['destination']}")
                            files_moved += 1
                        except Exception as e:
                            print(f"   ❌ Failed to move {pattern}: {e}")
        
        print(f"   📊 Moved {files_moved} files")
    
    # Clean up empty directories
    print(f"\n🧹 Cleaning up empty directories...")
    cleanup_dirs = ['test', 'tests']  # Old test directories
    
    for cleanup_dir in cleanup_dirs:
        cleanup_path = project_root / cleanup_dir
        if cleanup_path.exists() and cleanup_path.is_dir():
            try:
                # Only remove if empty
                if not any(cleanup_path.iterdir()):
                    cleanup_path.rmdir()
                    print(f"   ✅ Removed empty directory: {cleanup_dir}")
                else:
                    print(f"   ⚠️ Directory not empty, skipping: {cleanup_dir}")
            except Exception as e:
                print(f"   ❌ Failed to remove {cleanup_dir}: {e}")
    
    # Create __init__.py files for Python packages
    print(f"\n📦 Creating Python package files...")
    package_dirs = [
        'src',
        'src/data_pipeline',
        'src/data_pipeline/collectors',
        'src/data_pipeline/processors',
        'src/data_pipeline/validators', 
        'src/data_pipeline/orchestrators',
        'src/feature_engineering',
        'src/models',
        'src/backtesting',
        'src/risk_metrics',
        'src/utils',
        'tests',
        'tests/unit',
        'tests/integration',
        'tests/data'
    ]
    
    for package_dir in package_dirs:
        init_file = project_root / package_dir / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"   ✅ Created __init__.py in {package_dir}")
    
    # Create a reorganization summary
    summary = {
        'reorganization_date': str(Path.cwd()),
        'directories_created': len(directories_to_create),
        'files_moved': sum([len(m.get('source_files', [])) for m in movements]),
        'new_structure': {
            'src/': 'Core application code',
            'data/': 'Organized data storage',
            'config/': 'Configuration files',
            'scripts/': 'Automation scripts',
            'tests/': 'Test suite',
            'docs/': 'Documentation',
            'logs/': 'Application logs',
            'notebooks/': 'Jupyter notebooks'
        }
    }
    
    summary_file = project_root / 'docs' / 'reorganization_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n🎉 Codebase reorganization completed!")
    print(f"📊 Summary:")
    print(f"   📁 Directories created: {len(directories_to_create)}")
    print(f"   📦 Package files created: {len(package_dirs)}")
    print(f"   📄 Summary saved to: docs/reorganization_summary.json")
    
    print(f"\n📋 New Structure Overview:")
    print(f"   📁 src/ - Core application code")
    print(f"   📁 data/ - Organized data storage")
    print(f"   📁 config/ - Configuration management")
    print(f"   📁 scripts/ - Automation & utilities")
    print(f"   📁 tests/ - Comprehensive test suite")
    print(f"   📁 docs/ - Documentation")
    print(f"   📁 logs/ - Application logs")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Test the new pipeline: python3 scripts/pipeline/run_pipeline.py --dry-run")
    print(f"   2. Run data collection: python3 scripts/data_collection/run_enhanced_collection.py --test")
    print(f"   3. Validate structure: python3 -c 'import src.data_pipeline.orchestrators.pipeline_orchestrator'")
    print(f"   4. Update imports in existing code if needed")

if __name__ == "__main__":
    reorganize_codebase()
