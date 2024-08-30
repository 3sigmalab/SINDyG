#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:55:52 2023

@author: mohammadaminbasiri
"""

# %%  ## Libraries ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


import sys
sys.path.append(r'/...')  # Replace with the actual path to your project directory
from run_experiment import run_experiment # Import the run_experiment function



#%% run simple num of osillators

Variable_name= 'num_oscillators'
Variable_list = [3, 4, 5, 6]  # Adjust as needed [3, 4, 5, 6, 7, 8, 9, 10]
# Variable_name= 'L'
# Variable_list = list(range(2, 40, 20)) #list(range(2, 40, 4))#np.linspace(1, 20, 2)

repetition = 2
# modify results as well


# Main experiment loop
base_path = r"/..." # Replace with the actual path to your project directory
csv_file_path = os.path.join(base_path, Variable_name + ".csv")

total_runs = len(Variable_list) * repetition # 10 repetitions per num_oscillators
current_run = 1
results_i = {}  # Store results for each num_oscillators value

# Create the folder if it doesn't exist
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)



for V in Variable_list:
    results_i[V] = []  # Initialize an empty list
    while len(results_i[V]) < repetition:

        print(f"Running experiment {current_run}/{total_runs} for Variable={V}")

        results = run_experiment(
            dt=0.01, Al=0.2, max_edge_value=4, max_freq=2, # 5,2
            min_freq=0.1, d=2, min_time=0, max_time=20,
            num_oscillators=V, min_time_train=0,
            max_time_train=15, poly_deg=3, lamda_reg=0.05,
            STLSQ_thr=0.14, L=10,#10
            test_length=1.0, Graph = "SF", # SF #ER
            plot_train_data = False, print_summary = False
            )

        if results is not None: # Only append if the experiment finished successfully
            results_i[V].append(results)
            current_run += 1 #Increment current run after the experiment

# After the loop, concatenate all results into a single DataFrame
all_results = []
for V in Variable_list:
    all_results.extend(results_i[V])

df = pd.DataFrame(all_results)

# Append or create the CSV
if os.path.isfile(csv_file_path):
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_file_path, index=False)
    



#%% read
X_Lable = Variable_name
attributes_to_plot = [
    'Train_r2', 'Train_MSE',
    'Complexity', 'total_time',
    'CEI', 'Test_r2', 
    'Test_MSE'
]


df = pd.read_csv(csv_file_path)


for attribute in attributes_to_plot:
    attribute_base = attribute
    attribute = attribute + '_SINDy'
    attributeG = attribute + 'G' # Construct the corresponding attributeG


    # Group and aggregate data 
    grouped_data = df.groupby(X_Lable).agg({
        attribute: ['mean', 'std'],
        attributeG: ['mean', 'std']
    })

    # Calculate standard errors
    grouped_data[(attribute, 'stderr')] = grouped_data[attribute, 'std'] / np.sqrt(repetition)
    grouped_data[(attributeG, 'stderr')] = grouped_data[attributeG, 'std'] / np.sqrt(repetition)
    
    
    

    # Extract data for plotting
    x_values = grouped_data.index

    mean_values = grouped_data[attribute, 'mean']
    stderr_values = grouped_data[attribute, 'stderr']

    mean_values_g = grouped_data[(attributeG, 'mean')]
    stderr_values_g = grouped_data[(attributeG, 'stderr')]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot SINDy
    plt.plot(x_values, mean_values, label='Mean ' + attribute_base + ' (SINDy)', marker='o')
    plt.fill_between(x_values, mean_values - stderr_values, mean_values + stderr_values, alpha=0.3)

    # Plot SINDyG 
    plt.plot(x_values, mean_values_g, label='Mean ' + attribute_base + ' (SINDyG)', marker='o')
    plt.fill_between(x_values, mean_values_g - stderr_values_g, mean_values_g + stderr_values_g, alpha=0.3)

    plt.xlabel(X_Lable)
    plt.ylabel(attribute_base)
    plt.title(f'Sensitivity Plot of {attribute_base} (SINDy and SINDyG)')
    plt.legend()
    plt.grid(True)
    plt.xticks(x_values)
    plt.show()



#%% Table
# run simple SF

Variable_name= 'Graph'
Variable_list = ["SF","ER"] #list(range(2, 40, 4))#np.linspace(1, 20, 2)
repetition = 2
# modify results as well
X_Lable = Variable_name
attributes_to_plot = [
    'Train_r2', 'Train_MSE',
    'Complexity', 'total_time',
    'CEI', 'Test_r2', 
    'Test_MSE'
]


# Main experiment loop
base_path = r"C:\..." # Replace with the actual path to your project directory
csv_file_path = os.path.join(base_path, Variable_name + ".csv")

total_runs = len(Variable_list) * repetition # 10 repetitions per num_oscillators
current_run = 1
results_i = {}  # Store results for each num_oscillators value

# Create the folder if it doesn't exist
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)



for V in Variable_list:
    results_i[V] = []  # Initialize an empty list
    while len(results_i[V]) < repetition:

        print(f"Running experiment {current_run}/{total_runs} for Variable={V}")

        results = run_experiment(
            dt=0.01, Al=0.2, max_edge_value=np.random.uniform(1, 5), max_freq=np.random.uniform(1, 2), # 5,2
            min_freq=0.1, d=2, min_time=0, max_time=20,
            num_oscillators=np.random.randint(3, 7), min_time_train=0,
            max_time_train=np.random.randint(10, 16), poly_deg=3, lamda_reg=0.05,
            STLSQ_thr=0.14, L=10,#10
            test_length=1.0, Graph = V, # SF #ER
            plot_train_data = False, print_summary = False
            )

        if results is not None: # Only append if the experiment finished successfully
            results_i[V].append(results)
            current_run += 1 #Increment current run after the experiment

# After the loop, concatenate all results into a single DataFrame
all_results = []
for V in Variable_list:
    all_results.extend(results_i[V])

df = pd.DataFrame(all_results)

# Append or create the CSV
if os.path.isfile(csv_file_path):
    df.to_csv(csv_file_path, mode='a', header=False, index=False)
else:
    df.to_csv(csv_file_path, index=False)


# Print the values for graph type
for attribute in attributes_to_plot:
    attribute_base = attribute
    attribute = attribute + '_SINDy'
    attributeG = attribute + 'G'  # Construct the corresponding attributeG

    # Group and aggregate data 
    grouped_data = df.groupby(X_Lable).agg({
        attribute: ['mean', 'std'],
        attributeG: ['mean', 'std']
    })

    # Calculate standard errors
    grouped_data[(attribute, 'stderr')] = grouped_data[attribute, 'std'] / np.sqrt(repetition)
    grouped_data[(attributeG, 'stderr')] = grouped_data[attributeG, 'std'] / np.sqrt(repetition)


    print(f"\n--- {attribute_base} ---")
    for graph_type in grouped_data.index:
        mean_value = grouped_data.loc[graph_type, (attribute, 'mean')]
        stderr_value = grouped_data.loc[graph_type, (attribute, 'stderr')]
        print(f"SINDy ({graph_type}): {mean_value:.4f} ± {stderr_value:.4f}")
    
        mean_value_g = grouped_data.loc[graph_type, (attributeG, 'mean')]
        stderr_value_g = grouped_data.loc[graph_type, (attributeG, 'stderr')]
        print(f"SINDyG ({graph_type}): {mean_value_g:.4f} ± {stderr_value_g:.4f}")

