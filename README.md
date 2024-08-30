# Discovering governing equations from Graph-structured data by sparse identification of nonlinear dynamical systems

we developed a new method called Sparse Identification of Nonlinear Dynamical Systems from Graph-structured data (SINDyG), which incorporates the network structure into sparse regression to identify model parameters that explain the underlying network dynamics. SINDyG discovers the governing equations of network dynamics while offering improvements in accuracy and model simplicity. 

This repository contains Python code for modeling and analyzing coupled oscillator systems using the SINDy (Sparse Identification of Nonlinear Dynamics) and SINDyG (SINDy with Graph) algorithms. 


## Authors

1. <strong>Mohammad Amin Basiri:</strong>   </a> <a href="https://scholar.google.com/citations?user=s2LeFW4AAAAJ&hl=en" target="_blank">
        <img src="https://img.shields.io/badge/Google Scholar-Link-lightblue"> 
    
2. <strong>Sina Khanmohammadi:</strong>  </a> <a href="https://scholar.google.co.uk/citations?hl=en&user=K6sMFj4AAAAJ&view_op=list_works&sortby=pubdate" target="_blank">
        <img src="https://img.shields.io/badge/Google Scholar-Link-lightblue">



## Instructions

### Installation

1. **Prerequisites:** Make sure you have the following installed:
   * Python (3.6 or later)
   * Required libraries:  `numpy`, `pysindy`, `matplotlib`, `scipy`, `sklearn`, `networkx`, `sympy`, `pandas`

2. **Install Git:**

   * If you don't have Git, install it using:

     ```bash
     conda install -c anaconda git
     ```
   * Verify the installation and get information about the installed version of git:

     ```bash
     git --version
     ```
3. **Install PySINDy:**
      * PySINDy is a sparse regression package with several implementations for the Sparse Identification of Nonlinear Dynamical systems (SINDy) method introduced in Brunton et al. (2016a)
   * You can install the version V1.7.5 of pysindy by the following command:

     ```bash
     !pip install git+https://github.com/dynamicslab/pysindy@v1.7.5

     ```


   * You can use the following to check if pysindy is installed:

     ```bash
     !pip freeze
     ```

   * make sure the version is the one which is in their GitHub repository "pysindy @ git+https://github.com/dynamicslab"


4. **Add SINDyG Add-ons to PySINDy:**
      * Once PySINDy is installed, navigate to its installation directory (you can find it using `!pip show pysindy`).
      * Then, make the following modifications to add the SINDyG Add-ons to the PySINDY package.  
      * replace `(installation directory)/pysindy/__init__.py` with `SINDyG/PySINDy Add-ons/__init__.py`
      * replace `(installation directory)/pysindy/pysindy.py` with `SINDyG/PySINDy Add-ons/pysindy.py`
      * replace `(installation directory)/pysindy/optimizers/__init__.py` with `SINDyG/PySINDy Add-ons/optimizers/__init__.py`
      * Add `Graph-SINDY/pysindy/optimizers/STLSQG.py` to `(installation directory)/pysindy/optimizers/`

### Run Experiments
* **Single Run:**

   * Execute `SindyGraph_main_singlerun.py` to run a single experiment and visualize results.
   * Customize parameters within the `run_experiment` function call.

```python
run_experiment(dt=0.01, # time step
                   Al=0.2, # -0.2 for damping, 0.2 for oscillating
                   max_edge_value=5, # maximum value for an edge  when generating random edge values #5
                   max_freq=2, # maximum natural frequency
                   min_freq=0.1, # minimum natural frequency
                   d=2, #dimension of each oscillator
                   min_time=0, # starting time of Full signal
                   max_time=20, # finish time of Full signal
                   num_oscillators=5,
                   min_time_train=0, # minimum of train time
                   max_time_train=15, # maximum of train time
                   poly_deg=3, # the polynomial degree.
                   lamda_reg=0.05, # Lambda of Reg
                   STLSQ_thr=0.14, # the threshold of STLSQ
                   L=10, # parameter for adjusting the shape of the function
                   test_length=1.0, length of test set
                   timeout_seconds=20, # stop the run if it takes a long time
                   Graph = "ER", #"SF" Type of the graph "Scale-free" or "ER"
                   plot_train_data = True, # plot the data
                   print_summary = True # print summary of the model + all the scores and metrics
                   )
```


<strong>Train data:</strong> 

<img src='Sample Results/Training.png' type='image'></a>    


<strong>Test data:</strong> 

<img src='Sample Results/Testing.png' type='image'></a> 


* **Multiple Runs (Sensitivity Analysis):**
   * Execute `SindyGraph_main_sensitivity.py` to run multiple experiments with varying parameters.
   * Configure `Variable_name` and `Variable_list` to control the parameter being varied and its range.

   ```python
   Variable_name= 'num_oscillators'
   Variable_list = [3, 4, 5, 6] 
   repetition = 5
   ```
   * Set `repetition` to define the number of runs for each parameter value.
   * Also when the run_experiment function is going to be called in a loop, set the value of the corresponding variable to be chosen from vector V that iterates in Variable_list.

### Analyze Results
   * **CSV File:** The experiment results will be saved in the specified `csv_file_path`.
   * **Sensitivity Plots:** The code will generate sensitivity plots for various attributes (e.g., Complexity, Train_r2) to visualize the impact of different parameters on model performance.






<strong>one example of sensitivity analysis:</strong> 

<img src='Sample Results/Sensitivty.png' type='image'></a>    


### Code Structure

* **`sindy_functions.py`:**  Contains all the core functions for simulating dynamics, training models, plotting, etc. 
* **`run_experiment.py`:**  Defines the `run_experiment` function that encapsulates the logic for a single experiment run.
* **`SindyGraph_main_sensitivity.py`:**  Contains the main experiment loop that iterates over parameter combinations, runs experiments, and saves results to a CSV file. Also includes the code for reading the CSV and generating sensitivity plots.
* **`SindyGraph_main_Singlerun.py`:**  Contains One experiment loop with one parameter combination, Plot Train data, Plot test data, providing summery model, and evaluating using different metrics.

### Customization

* **Parameters:** Adjust the parameters in the main experiment loop or the `run_experiment` function to explore different scenarios.
* **Graph Types:**  The code currently supports 'ER' (Erdős-Rényi) and 'SF' (Scale-Free) graphs. You can add support for other graph types by modifying the `run_experiment` function.
* **Metrics and Visualization:**  You can customize the `attributes_to_plot` list to generate sensitivity plots for different metrics.  You can also modify the plotting code to create different types of visualizations or use other plotting libraries.
   
