#!/bin/bash
cd "$(dirname "$0")"
while [ ! -f ../results/exp7_invariance.json ]; do sleep 15; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -u run_all.py exp5_robustness >> ../results/run_all.log 2>&1
python3 -u run_all.py exp8_tightness >> ../results/run_all.log 2>&1
python3 -u run_all.py exp9_modelerror >> ../results/run_all.log 2>&1
python3 -u run_all.py exp10_noise >> ../results/run_all.log 2>&1
python3 -u run_all.py exp2_coverage >> ../results/run_all.log 2>&1
python3 -u run_all.py exp13_tolerance >> ../results/run_all.log 2>&1
python3 -u run_all.py exp11_scaling >> ../results/run_all.log 2>&1
python3 -u run_all.py exp5_robustness >> ../results/run_all.log 2>&1
echo "EXTRA DONE"
