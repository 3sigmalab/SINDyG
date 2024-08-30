#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:55:52 2023

@author: mohammadaminbasiri
"""

# %%  ## Libraries ###

import sys
sys.path.append(r'/Users/sinakhan/Desktop/GSINDY')  # Replace with the actual path to your project directory
from run_experiment import run_experiment # Import the run_experiment function    
run_experiment(
    dt=0.01, Al=0.2, max_edge_value=4, max_freq=2, # 5,2
    min_freq=0.1, d=2, min_time=0, max_time=20,
    num_oscillators=5, min_time_train=0,
    max_time_train=15, poly_deg=3, lamda_reg=0.05,
    STLSQ_thr=0.14, L=10,#10
    test_length=1.0, Graph = "SF", # SF #ER
    plot_train_data = True, print_summary = True
    )