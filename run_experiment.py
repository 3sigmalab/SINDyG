#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:55:52 2023

@author: mohammadaminbasiri
"""

# %%  ## Libraries ###
import numpy as np
import pysindy as ps
from scipy.integrate import odeint
from sklearn.metrics import mean_squared_error
import time
import networkx as nx
import threading
import concurrent.futures
import queue
from SindyGraph_functions import *


# %% ## simulating graph and signals ##

def run_experiment(dt=0.01,
                   Al=0.2, # -0.2 for damping, 0.2 for oscillating
                   max_edge_value=5, # maximum value for edge  when generating randomedge values #5
                   max_freq=2, # maximum natural frequency
                   min_freq=0.1, # minimum natural frequency
                   d=2, #dimension of each oscillator
                   min_time=0, # starting time of Full signal
                   max_time=20, # finish time of Full signal
                   num_oscillators=10,
                   min_time_train=0, # minimum of train time
                   max_time_train=15, # maximum of train time
                   poly_deg=3, # the polynomial degree.
                   lamda_reg=0.05, # Lambda of Reg
                   STLSQ_thr=0.14, # the threshold of STLSQ
                   L=10,
                   test_length=1.0,
                   timeout_seconds=20,
                   Graph = "ER", # SR
                   plot_train_data = False,
                   print_summary = False
                   ):
    
    result_queue = queue.Queue()  # Create a queue to store results
    def experiment_task():

        
        # random natural frequency for each oscilator 
        w = 2 * np.pi * ((max_freq-min_freq)*np.random.random(num_oscillators) + min_freq)  # Frequencies between 0.5π and 1.5π
        
        

        # A: (1 indicates impact, 0 indicates no impact)     
        if Graph == "SF": #SF Scale-Free
            G = nx.scale_free_graph(num_oscillators)
            G = nx.Graph(G)  # Convert to simple graph 
            A = nx.to_numpy_array(G) # Get adjacency matrix
        else: # ER
            A = np.random.randint(2, size=(num_oscillators, num_oscillators))

        
        
        num_links = np.count_nonzero(A)
        
        coupling_matrix = A
        
        # if we choose coupling to be random
        coupling_matrix = np.multiply(max_edge_value*(2*np.random.random(size=(num_oscillators, num_oscillators))-1), A) 
        
        
        # Generating random initial state for multiple pairs
        phase = 2 * np.pi * np.random.random(num_oscillators)
        amp = np.random.random(num_oscillators)
        
        # assigning initial states [real, imag]
        initial_state = np.array([
            [Al * b * np.cos(a), Al * b * np.sin(a)] for a, b in zip(phase, amp)
        ])
        
        #flat the states to a vector
        state0 = initial_state.flatten()
        
        # Time steps
        time_steps = np.arange(min_time, max_time, dt)  # Train length
        
        #numerically solves a system of ordinary differential equations
        ode_out = odeint(f, state0, time_steps, args=(num_oscillators, w, coupling_matrix, Al))
        
        x_train = np.reshape(ode_out, (-1, num_oscillators, d))  #d dimension of each oscillator
        
        # Output the results
        #print(x_train)
        
        plot_train(plot_train_data, x_train, num_oscillators, initial_state, time_steps)
        
        
        #####################################################################################
        """        
        # ## TRAINING THE MODEL ##
        ### Minimize Objective function ||y - Xw||^2  +  a||w||^2 (L1 norm)
        """    
        
        
        MT = False # multi trajectory
        
        
        test_length = 1.0 # length of the test
        
        #creating X and t for training using the simulated data
        X = ode_out[int(min_time_train/dt):int(max_time_train/dt)]
        t = time_steps[int(min_time_train/dt):int(max_time_train/dt)]
        
        
        #library with polynomial degree3
        custom_library = ps.PolynomialLibrary(degree=poly_deg)
        
        
        #original sindy
        model_2 = ps.SINDy(
            differentiation_method=ps.FiniteDifference(order=2),
            feature_library=custom_library,
            #optimizer=ps.STLSQ(threshold=0.14), #fit_intercept=(True) #threshold=0.065 , verbose=True
            optimizer=ps.STLSQ(threshold=STLSQ_thr, alpha=lamda_reg, verbose = print_summary), #fit_intercept=(True) #threshold=0.065 , verbose=True #, F_penalize = F # minus value of beta means simple ridge regression
            #feature_names=["x0", "y0", "x1", "y1", "x2", "y2"]
        )
        #Change unbias = True to see the better results since performs a standard linear regression (without any regularization) using only the selected features.
        start_time = time.time()
        model_2.fit(X, t=t, multiple_trajectories = MT, unbias=True) # t = dt  #If unbias=True, an extra unregularized linear regression step is performed to refine the coefficients of the identified terms.
        end_time = time.time()
        total_time_SINDy = end_time - start_time

        
        # state variables and features lists
        State_variables = model_2.feature_names
        Features = model_2.feature_library.get_feature_names()
        
        # Create the presence matrix
        presence_matrix = relaxed_presence_terms(State_variables, Features)

        #creation of the F matrix for regularization
        F = np.ones((len(State_variables), len(Features)))
        

        AA = A.copy()
        np.fill_diagonal(AA, 1) #the diagonal elements of AA are all 1.
        for j,term in enumerate(Features):
            Source_indices = np.where(presence_matrix[:,j] == 1)[0]
            Sink_indices = get_sink_indices(Source_indices, d, len(State_variables))
            # print(f"number = {j}, term = {term}",
            #       f"presence Source = x{Source_indices}, presence Sink = x{Sink_indices}")
            for i in Sink_indices:
                connectivity = AA[Source_indices//d,i//d] #[Source_indices//d,i//d]
                mean = np.mean(connectivity)    
                # mean/len(Source_indices) doing this so it should be less likely to have terms with lots of statevariables
                F[i,j] = 1/(1+np.exp((L*((mean/len(Source_indices))-0.5))))
                F[i,j] = 1/(1+np.exp((L*((mean-0.5)/len(Source_indices)))))
                # print(f"connection between the Sources and x{i} = {connectivity} ,",
                #       f" mean of {len(connectivity)} connections = {mean}, F[{i},{j}] = {F[i,j]}")
        
        
        model_1 = ps.SINDy(
            differentiation_method=ps.FiniteDifference(order=2),
            feature_library=custom_library,
            optimizer=ps.STLSQG(threshold=STLSQ_thr, alpha=lamda_reg, verbose = print_summary, F_penalize = F, beta = 0.8), #fit_intercept=(True) #threshold=0.065 , verbose=True #, F_penalize = F # minus value of beta means simple ridge regression
        )
        
        start_time = time.time()
        model_1.fit(X, t=t, multiple_trajectories = MT, unbias=True) # t = dt  #If unbias=True, an extra unregularized linear regression step is performed to refine the coefficients of the identified terms.
        end_time = time.time()
        total_time_SINDyG = end_time - start_time
        
        Train_r2_SINDyG = model_1.score(X, t=t, multiple_trajectories = MT)
        Train_MSE_SINDyG = model_1.score(X, t=t, metric=mean_squared_error, multiple_trajectories = MT)
        Complexity_SINDyG = model_1.complexity
        Train_r2_SINDy = model_2.score(X, t=t, multiple_trajectories = MT)
        Train_MSE_SINDy = model_2.score(X, t=t, metric=mean_squared_error, multiple_trajectories = MT)
        Complexity_SINDy = model_2.complexity
        
        # CAI
        # Assuming predicted_coefficients_SINDy and predicted_coefficients_SINDyG are obtained
        
        trueCoeffs = calculate_trueC(model_1, num_oscillators, w, Features, Al, coupling_matrix)
        
        CAI_SINDy = calculate_CAI(trueCoeffs, model_2.coefficients())
        CAI_SINDyG = calculate_CAI(trueCoeffs, model_1.coefficients())
        
        
        """
         ## TESTING THE MODEL ## test for short 1sec
        """
        # # Test data and different initial condition 
        # Generating random initial state for multiple pairs
        phase = 2 * np.pi * np.random.random(num_oscillators)
        amp = np.random.random(num_oscillators)
        initialState1 = np.array([[Al * b * np.cos(a), Al * b * np.sin(a)] for a, b in zip(phase, amp)])
        s1 = initialState1.flatten()
        Test_r2_SINDyG, Test_MSE_SINDyG, Test_r2_SINDy, Test_MSE_SINDy = plot_simulation(model_1, model_2, s1, min_time, max_time, dt, num_oscillators, w, coupling_matrix, Al, d,interval_length=test_length, Plot=plot_train_data)
        
        if print_summary:
            print("\n\n\n-----------------------SUMMARY------------------------------")
            print("\nNumber of nodes:\n ", num_oscillators)
        
            print("\n\n***********SindyG Model*********** ")
            model_1.print()
            print("\nTraining r2_Score:\n ", '%.8f' %Train_r2_SINDyG)
            print("\nTraining mean_squared_error:\n ", '%.8f' % Train_MSE_SINDyG)
            print("\nModel Complexity:\n ", Complexity_SINDyG)
            print("\nTotal execution time SINDyG (s): \n", '%.8f' %total_time_SINDyG)
            print("\nCEI: \n", '%.8f' %CAI_SINDyG)
            print('\nTest r2_score: \n', '%.8f' % Test_r2_SINDyG)
            print('\nTest MSE: \n', '%.8f' % Test_MSE_SINDyG)
        
            print("\n\n************Original Sindy Model************ ")
            model_2.print()
            print("\nTraining r2_Score:\n ",'%.8f' % Train_r2_SINDy)
            print("\nTraining mean_squared_error:\n ", '%.8f' % Train_MSE_SINDy)
            print("\nModel Complexity:\n ", Complexity_SINDy)
            print("\nTotal execution time SINDy (s): \n", '%.8f' %total_time_SINDy)
            print("\nCEI: \n", '%.8f' %CAI_SINDy)
            print('\nTest r2_score: \n', '%.8f' % Test_r2_SINDy)
            print('\nTest MSE: \n', '%.8f' % Test_MSE_SINDy)
            
        results = {
        'dt': dt, 'Al': Al, 'max_edge_value': max_edge_value, 'max_freq': max_freq, 
        'min_freq': min_freq, 'd': d, 'min_time': min_time, 'max_time': max_time, 'min_time_train': min_time_train, 
        'max_time_train': max_time_train, 'poly_deg': poly_deg, 'lamda_reg': lamda_reg,
        'STLSQ_thr': STLSQ_thr, 'L': L, 'test_length': test_length, 'Graph': Graph, 
        'num_oscillators': num_oscillators, 'num_links': num_links,
        'Train_r2_SINDyG': Train_r2_SINDyG, 'Train_MSE_SINDyG': Train_MSE_SINDyG,
        'Complexity_SINDyG': Complexity_SINDyG, 'total_time_SINDyG': total_time_SINDyG,
        'CEI_SINDyG': CAI_SINDyG, 'Test_r2_SINDyG': Test_r2_SINDyG, 
        'Test_MSE_SINDyG': Test_MSE_SINDyG, 'Train_r2_SINDy': Train_r2_SINDy,
        'Train_MSE_SINDy': Train_MSE_SINDy, 'Complexity_SINDy': Complexity_SINDy,
        'total_time_SINDy': total_time_SINDy, 'CEI_SINDy': CAI_SINDy, 
        'Test_r2_SINDy': Test_r2_SINDy, 'Test_MSE_SINDy': Test_MSE_SINDy
        }
        if Test_r2_SINDy < -10 or Test_MSE_SINDy > 0.2 or Test_r2_SINDyG < -10 or Test_MSE_SINDyG > 0.2 or Train_r2_SINDy < -10 or Train_MSE_SINDy > 0.2 or Train_r2_SINDyG < -10 or Train_MSE_SINDyG > 0.2 or total_time_SINDyG> 5 or total_time_SINDy> 5:
            results = None
        result_queue.put(results)
        
        
    def target_function():
        # Wrapper function to handle timeout
        thread = threading.Thread(target=experiment_task)
        thread.start()
        thread.join(timeout=timeout_seconds)
        if thread.is_alive():
            print(f"Experiment timed out for num_oscillators={num_oscillators}. Skipping...")
            return None
        else:
            return result_queue.get()

    results = target_function()
    return results



