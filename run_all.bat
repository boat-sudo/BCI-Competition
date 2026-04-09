@echo off
set DATA_DIR=D:\WESAD
set OUTPUT_DIR=.\paper_run_outputs

python run_all.py --data_dir "%DATA_DIR%" --output_dir "%OUTPUT_DIR%" --window_seconds 300 --step_seconds 300

pause