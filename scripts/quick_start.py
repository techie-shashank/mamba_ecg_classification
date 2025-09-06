#!/usr/bin/env python3
"""
Quick start script for running experiments and parsing data
"""

import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Setup paths for imports
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

# Add both to path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))

# Store original directory and project root for file operations
original_dir = os.getcwd()

from experiment_examples import run_comprehensive_search
from src.logger import get_experiment_logger

# Clear any existing handlers to ensure fresh logger
import logging
for handler in logging.getLogger('mamba_ts_forecasting').handlers[:]:
    logging.getLogger('mamba_ts_forecasting').removeHandler(handler)

# Setup logging with file output
logger = get_experiment_logger('quick_start')


def run_multiple_experiments(datasets=None, models=None):
    """Run multiple experiments with different configurations"""
    logger.info("🚀 Starting multiple experiments...")
    
    try:
        runner = run_comprehensive_search()
        runner.run_all_experiments()
        logger.info("✅ Experiments completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error running experiments: {e}")
        return False

def parse_experiments():
    """Parse all experiments and update database"""
    logger.info("📊 Parsing experiment data...")
    
    try:
        cmd = ['python', './scripts/experiment_data_parser.py']
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Data parsing completed successfully")
            # Log the output from the parser
            if result.stdout:
                logger.info(f"Parser output: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ Data parsing failed with return code: {result.returncode}")
            if result.stderr:
                logger.error(f"Parser error: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during data parsing: {e}")
        return False

def main():
    """Main function with comprehensive logging"""
    logger.info("="*60)
    logger.info("QUICK START - ML EXPERIMENT PIPELINE")
    logger.info("="*60)
    
    start_time = datetime.now()
    logger.info(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = True
    
    try:
        # Run experiments
        logger.info("\n" + "="*40)
        logger.info("PHASE 1: RUNNING EXPERIMENTS")
        logger.info("="*40)
        
        if not run_multiple_experiments():
            success = False
            logger.error("❌ Experiment phase failed")
        
        # Parse experiments
        logger.info("\n" + "="*40)
        logger.info("PHASE 2: PARSING EXPERIMENT DATA")
        logger.info("="*40)
        
        if not parse_experiments():
            success = False
            logger.error("❌ Parsing phase failed")
            
    except Exception as e:
        logger.error(f"❌ Critical error in pipeline: {e}")
        success = False
    
    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total duration: {duration}")
    
    if success:
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        logger.error("💥 PIPELINE FAILED!")
        
    logger.info("="*60)


if __name__ == "__main__":
    main()
