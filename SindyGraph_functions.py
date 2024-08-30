#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 16:55:52 2023

@author: mohammadaminbasiri
"""

# %%  ## Libraries ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error  # Import the MSE function f
import re
import sympy as sp

# Function that shows the impact on another signal using the coupling term
def func(z1,z2,C):
    return C * z1 * z2

# General function for node dynamics
def node_function(own_state, coupled_states, w, Al):
    x_real, y_img = own_state
    z_cplx = complex(x_real, y_img)
    z_dot = (Al + w * 1j - (abs(z_cplx)) ** 2) * z_cplx
    for state, c in coupled_states:
        z_cplx_coupled = complex(state[0], state[1])
        z_dot += func(z_cplx, z_cplx_coupled, c)
    return z_dot

# The overall function representing the dynamics
def f(state, t, num_oscillators, w, coupling_matrix, Al):
    # reshaping the state to [real, imag]
    state = np.reshape(state, (-1, 2))
    z_dot = np.zeros((num_oscillators, 2))
    for i in range(num_oscillators):
        coupled_states = [(state[j],coupling_matrix[j, i]) for j in range(num_oscillators) if coupling_matrix[j, i]>0]
        z_dot_i = node_function(state[i], coupled_states, w[i], Al)
        z_dot[i] = [z_dot_i.real, z_dot_i.imag]
    return z_dot.flatten()

# Populate true coefficients based on the true dynamics
def calculate_trueC(model_1, num_oscillators, w, Features, Al, coupling_matrix):
    trueC = np.zeros(model_1.coefficients().shape)
    # The coefficients are derived from the SL equation and coupling terms
    x = sp.symbols('x0:%d' % (2*num_oscillators))
    for i in range(2*num_oscillators):
        w_i = w[i//2]
        if (i % 2) == 0:
            other = i + 1
            first = 1
        else:
            other = i - 1
            first = -1
        for j, term in enumerate(Features):
            if term == str(x[i]) + '^3':
                trueC[i, j] = -1
            elif term == str(x[i]):
                trueC[i, j] = Al
            elif term == str(x[other]):
                trueC[i, j] = (-1) * first * w_i
            elif term == str(x[i]) + ' ' + str(x[other]) + '^2' or term == str(x[other]) + '^2 ' + str(x[i]):
                trueC[i, j] = -1
                    
    
    # Include coupling terms
    for i in range(num_oscillators):
        for j in range(num_oscillators):
            if coupling_matrix[i, j] != 0:
                for k, term in enumerate(Features):
                    if i==j:
                        if term == str(x[2*i]) + '^2':
                            trueC[2*i, k] = coupling_matrix[i, j]
                        elif term == str(x[2*i+1]) + '^2':
                            trueC[2*i, k] = -1*coupling_matrix[i, j]
                        elif term == str(x[2*i]) + ' ' + str(x[2*i+1]) or term == str(x[2*i+1]) + ' ' + str(x[2*i]):
                            trueC[2*i+1, k] = 2*coupling_matrix[i, j]
                    else:
                        if term == str(x[2*i]) + ' ' + str(x[2*j]) or term == str(x[2*j]) + ' ' + str(x[2*i]):
                            trueC[2*j, k] = coupling_matrix[i, j]
                        elif term == str(x[2*i+1]) + ' ' + str(x[2*j+1]) or term == str(x[2*j+1]) + ' ' + str(x[2*i+1]):
                            trueC[2*j, k] = -1*coupling_matrix[i, j]
                        elif term == str(x[2*i]) + ' ' + str(x[2*j+1]) or term == str(x[2*j+1]) + ' ' + str(x[2*i]):
                            trueC[2*j+1, k] = coupling_matrix[i, j]
                        elif term == str(x[2*j]) + ' ' + str(x[2*i+1]) or term == str(x[2*i+1]) + ' ' + str(x[2*j]):
                            trueC[2*j+1, k] = coupling_matrix[i, j]
    return trueC

def plot_train(plot_train_data, x_train, num_oscillators, initial_state, time_steps):
    if plot_train_data:
        ## 2D
        fig0, axs0 = plt.subplots(x_train.shape[1], 1, figsize=(5, num_oscillators*5))
        for i in range(x_train.shape[1]):
            axs0[i].plot(x_train[:,i,0], x_train[:,i,1], "--", label="Dynamics", linewidth=2)
            axs0[i].plot(initial_state[i,0], initial_state[i,1], "ko", label="Initial condition", markersize=5)
            axs0[i].legend()
            axs0[i].set(xlabel="x", ylabel='Y')
            axs0[i].set_title('Oscilator ${}$'.format(i))
        fig0.show()
        
        
        
        ## 3D
        fig1, axs1 = plt.subplots(x_train.shape[1], 1, figsize=(num_oscillators*5, num_oscillators*5), subplot_kw=dict(projection='3d'))
        for i in range(x_train.shape[1]):
            axs1[i].plot(time_steps, x_train[:,i,0], x_train[:,i,1], "--", label="Dynamics", linewidth=2)
            axs1[i].plot(0, initial_state[i,0], initial_state[i,1], "ko", label="Initial condition", markersize=5)
            axs1[i].legend()
            axs1[i].set(xlabel="time", ylabel='X', zlabel='Y')
            axs1[i].set_title('Oscilator ${}$'.format(i))
        fig1.savefig('oscillator_plot.pdf')
        fig1.show()


# SINDyG
def relaxed_presence_terms(list1, list2):
    """
    Creates a matrix representing element presence in another list.
    The presence of the current state variable in the term IS NOT important.

    Args:
        list1 (list): The first list of elements to search for.
        list2 (list): The second list containing potential matches.

    Returns:
        numpy.ndarray: A matrix with shape (len(list1), len(list2)) where:
          - 0: Element from list1 not found in list2.
          - 1: Element from list1 found in list2 (matching the pattern 'x' followed by numbers).
    """
    matrix = np.zeros((len(list1), len(list2)), dtype=int)
    pattern = re.compile(r'x\d+')  #r'\bx\d+\b' # Pattern to match 'x' followed by numbers
    
    for i, element in enumerate(list1):
        for j, term in enumerate(list2):
            matches = pattern.findall(term)  # Find all matching patterns
            # if j<300:
            #     print(f"number = {j}, element = {element}, term = {term}, matches = {matches}, presence = {element in matches}")
            if element in matches:
                matrix[i, j] = 1
    
    return matrix


def get_sink_indices(source_indices, d, num_sv): # num_sv: number of state variables
    """Extracts sink indices (groups) from a list of source indices.

    Args:
        source_indices: A list of integers within the range of available numbers.
        d: The dimension of each node (e.g., 2 for pairs, 3 for triplets).
        num_sv: The total number of available state variables (e.g., 6 in simple example).

    Returns:
        A sorted list of sink indices, including groups for each source index.
    """

    sink_indices = set()
    groups = [list(range(i, min(i + d, num_sv))) for i in range(0, num_sv, d)] 

    for num in source_indices:
        if num < 0 or num >= num_sv:
            raise ValueError("Source index out of range num = ", num, " num_sv =", num_sv)

        group_index = num // d  
        sink_indices.update(groups[group_index])

    return sorted(sink_indices)



def plot_simulation(model1, model2, s0, min_time, max_time, dt, num_oscillators, w, coupling_matrix, Al, d, interval_length=1.0, Plot=True):

    
    t_test_full = np.arange(min_time, max_time, dt)  # Full test length

    # True simulation using IC for the full interval
    print("odeint full\n")
    X_test_full = odeint(f, s0, t_test_full, args=(num_oscillators, w, coupling_matrix, Al))

    # Generate a random starting time within the valid range for short interval
    valid_start_times = np.arange(min_time, max_time - interval_length, dt)
    start_time_random = np.random.choice(valid_start_times)

    # Extract the state at the random starting time to use as the initial condition for the short trajectory
    s_start = X_test_full[np.isclose(t_test_full, start_time_random)][0]

    # Define the short time range from the random starting time to the random starting time + interval_length
    t_test_short = np.arange(start_time_random, start_time_random + interval_length, dt)
    print("odeint short\n")
    # True simulation for the short interval
    X_test_short = odeint(f, s_start, t_test_short, args=(num_oscillators, w, coupling_matrix, Al))
    print("SindyG simulate\n")
    # Model1 simulation for the short interval
    sim1_short = model1.simulate(s_start, t=t_test_short)
    print("Sindy simulate\n")
    # Model2 simulation for the short interval
    sim2_short = model2.simulate(s_start, t=t_test_short)

    # Scores and R2 Scores for short interval
    r2_SINDyG = model1.score(X_test_short, t=t_test_short, metric=r2_score)
    MSE_SINDyG = model1.score(X_test_short, t=t_test_short, metric=mean_squared_error)
    r2_SINDy = model2.score(X_test_short, t=t_test_short, metric=r2_score)
    MSE_SINDy = model2.score(X_test_short, t=t_test_short, metric=mean_squared_error)

    if Plot:
        print("\n\n\n-----------------------------------------------------")
        print("***Comparison of SINDyG and SINDy Models for Test set***\n ")
        # Scores and R2 Scores for short interval
        print("\n***SindyG Model***\n")
        print('SindyG Model r2_score for initial condition', [*[f"{value:.4f}" for value in s_start]], ': \n%.8f' % model1.score(X_test_short, t=t_test_short, metric=r2_score))
        print('SindyG Model mean_squared_error for initial condition', [*[f"{value:.4f}" for value in s_start]], ': \n%.8f' % model1.score(X_test_short, t=t_test_short, metric=mean_squared_error))

        print("\n***Original Sindy Model***\n")
        print('Sindy Model r2_score for initial condition', [*[f"{value:.4f}" for value in s_start]], ': \n%.8f' % model2.score(X_test_short, t=t_test_short, metric=r2_score))
        print('Sindy Model mean_squared_error for initial condition', [*[f"{value:.4f}" for value in s_start]], ': \n%.8f' % model2.score(X_test_short, t=t_test_short, metric=mean_squared_error))

        # 2D Plot
        fig1, axs1 = plt.subplots(int(X_test_short.shape[1] / d), 1, figsize=(5, num_oscillators * 5))
        for i in range(int(X_test_short.shape[1] / d)):
            axs1[i].plot(X_test_short[:, d * i], X_test_short[:, d * i + 1], label="Ground truth", linewidth=1)
            axs1[i].plot(sim1_short[:, d * i], sim1_short[:, d * i + 1], "--", label="SINDyG estimate", linewidth=2)
            axs1[i].plot(sim2_short[:, d * i], sim2_short[:, d * i + 1], ":", label="SINDy estimate", linewidth=2)
            axs1[i].plot(s_start[d * i], s_start[d * i + 1], "ko", label="Initial condition", markersize=5)
            axs1[i].legend()
            axs1[i].set(xlabel="x", ylabel="y")
            axs1[i].set_title('Oscillator ${}$ prediction'.format(i))
        fig1.savefig('oscillator_prediction_combined_2D_short.pdf')
        fig1.show()

        # Time Series Plot
        fig2, axs2 = plt.subplots(X_test_short.shape[1], 1, sharex=True, figsize=(10, num_oscillators * 6))
        for i in range(X_test_short.shape[1]):
            axs2[i].plot(t_test_short, X_test_short[:, i], label='Ground truth', linewidth=3)
            axs2[i].plot(t_test_short, sim1_short[:, i], '--', label='SINDyG estimate', linewidth=2)
            axs2[i].plot(t_test_short, sim2_short[:, i], ':', label='SINDy estimate', linewidth=2)
            axs2[i].plot(t_test_short[0], s_start[i], "ko", label="Initial condition", markersize=5)
            axs2[i].legend()
            axs2[i].set(xlabel='time', ylabel='${}$'.format(model1.feature_names[i]))
        fig2.savefig('oscillator_prediction_combined_time_series_short.pdf')
        fig2.show()

        # Derivatives Plot
        x_dot_test_predicted1 = model1.predict(X_test_short)
        x_dot_test_predicted2 = model2.predict(X_test_short)
        x_dot_test_computed = model1.differentiate(X_test_short, t=dt)

        fig3, axs3 = plt.subplots(X_test_short.shape[1], 1, sharex=True, figsize=(10, num_oscillators * 6))
        for i in range(X_test_short.shape[1]):
            axs3[i].plot(t_test_short, x_dot_test_computed[:, i], 'k', label='numerical derivative')
            axs3[i].plot(t_test_short, x_dot_test_predicted1[:, i], 'r--', label='SINDyG prediction')
            axs3[i].plot(t_test_short, x_dot_test_predicted2[:, i], 'b:', label='SINDy prediction')
            axs3[i].legend()
            axs3[i].set(xlabel='time', ylabel=r'$\dot {}$'.format(model1.feature_names[i]))
        fig3.savefig('oscillator_prediction_combined_derivatives_short.pdf')
        fig3.show()

    return r2_SINDyG, MSE_SINDyG, r2_SINDy, MSE_SINDy



def calculate_CAI(true_coeffs, predicted_coeffs):
    K, C = true_coeffs.shape
    cai = np.sum(np.abs(true_coeffs - predicted_coeffs)) / (K * C)
    return cai

