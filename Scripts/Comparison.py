# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:19:40 2020

@author: JC
"""
# %% import packages
from os import chdir
import pickle
import glob
import TuckerFunction
import tensorly as tl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import kmeans
from time import time


# %% Create formatting and pre-processing functions
def labeling(number):
    number = int(number)
    if number <= 5:
        label = "DarkSteelSink1"
    elif number <= 8:
        label = "LightSteelSink1"
    elif number <= 11:
        label = "CeramicSink1"
    elif number <= 14:
        label = "SmallCeramicSink6"
    elif number <= 17:
        label = "LightGraniteSink1"
    elif number <= 25:
        label = "CeramicSink2"
    else:
        label = "ERROR"
    return label


def reformat_string(data):
    temp = np.empty(len(data), dtype=object)
    for k in range(len(data)):
        temp[k] = labeling(data[1][k])
    data["Type"] = temp
    data["Step"] = data[3]
    for k in range(6):
        del data[k]
    return data


def result_summary(name, convergence, score, contingency, digits):
    # Print clustering  results
    iterations = np.size(convergence)
    print(name,"results")
    print("Iteration count", iterations)
    print("Convergence average required steps", np.mean(convergence).round(digits))
    print("Average Adjusted Mutual Information", np.mean(score).round(digits))
    print("Standard Deviation AMI", np.std(score).round(digits))
    print("Best contingency table", "score = "+str(np.max(score).round(digits)),
          contingency, sep="\n")
    print("Average duration", np.mean(duration).round(digits))
    print("Run lasted", np.sum(duration).round(digits))
    print()
    

# %% Set experiment evaluation
np.random.seed(123)
max_iterations = 500
digits = 2

# Description of the stored decompositions parameters
decompositions_features = ["gray scale, rank=(32, 32, 32), frames=200",
                           "BGR, rank=(10, 10, 10, 2), frames=100",
                           "BGR, rank=(2, 2, 2, 2), frames=50"]

# Folders where distinct decompositions are stored
decompositions_paths = ["Decompositions", 
                        "Decompositions2",
                        "Decompositions3"]  
score_type_dictionary = {}
score_step_dictionary = {}

print("Running a total of", max_iterations, "iterations")

for index, path in enumerate(decompositions_paths):
    # Change path into right folder
    chdir(path)

    # Load SubTensors objects
    name_list = glob.glob('*')
    name_list = [n for n in name_list if "HandWash_" in n]

    if not name_list:
        print("Path provided has no tensor decompositions.")
        break

    Tensor_List = []
    for tensor in name_list:
        Handler = open(tensor, "rb")
        Tensor_List.append(pickle.load(Handler))

    for j in range(len(Tensor_List)):
        Tensor_List[j].core = np.asarray(Tensor_List[j].core)

    section_break = "-------------------------------------"
    print(section_break)
    print("Results for", decompositions_features[index])
    print("{")

    # Confusion matrix for Type
    score = []
    convergence = []
    duration = []
    iterations = 0
    while iterations < max_iterations:
        t_i = time()
        algorithm = kmeans.KMeans(Tensor_List, k=6)
        Clusters = algorithm.predict()
        if algorithm.exit_criteria == 'convergence':
            Table = pd.DataFrame()
            Table["Type"] = reformat_string(pd.Series(name_list).str.split("_", expand=True))["Type"]
            Table["Clusters"] = Clusters
            temp_score = metrics.adjusted_mutual_info_score(Table["Type"], Table["Clusters"])
            if not score or temp_score > np.max(score):
                Contingency_Type = pd.crosstab(Table["Clusters"], Table["Type"], margins=True)
            score.append(temp_score)
            convergence.append(algorithm.iterations)
            duration.append(time()-t_i)
            iterations+=1
    best_score_type = np.max(score)
    score_type_dictionary[decompositions_features[index]] = np.asarray(score)
            
    # Print clustering  results
    result_summary(name="Sink-Type", 
                   convergence=convergence, 
                   score=score, 
                   contingency=Contingency_Type,
                   digits=digits)
    
    # Confusion matrix for Type
    score = []
    convergence = []
    duration = []
    iterations = 0
    while iterations < max_iterations:
        t_i = time()
        algorithm = kmeans.KMeans(Tensor_List, k=3)
        Clusters = algorithm.predict()
        if algorithm.exit_criteria == 'convergence':
            Table = pd.DataFrame()
            Table["Step"] = reformat_string(pd.Series(name_list).str.split("_", expand=True))["Step"]
            Table["Clusters"] = Clusters
            temp_score = metrics.adjusted_mutual_info_score(Table["Step"], Table["Clusters"])
            if not score or temp_score > np.max(score):
                Contingency_Step = pd.crosstab(Table["Clusters"], Table["Step"], margins=True)
            score.append(temp_score)
            convergence.append(algorithm.iterations)
            duration.append(time()-t_i)
            iterations+=1
    best_score_step = np.max(score)
    score_step_dictionary[decompositions_features[index]] = np.asarray(score)

    # Print clustering  results
    result_summary(name="Step-number", 
               convergence=convergence, 
               score=score, 
               contingency=Contingency_Step,
               digits=digits)

    #  Adding plots respect to the confusion matrix of cluster per true label
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    # Define max values for temperature range on heatmap.
    max_type_margin = min(Contingency_Type.T.All.min(), 
                      Contingency_Type.All.min())
    max_steps_margin = min(Contingency_Step.T.All.min(),
                           Contingency_Step.All.min())
    
    # Plotting heatmaps
    sns.heatmap(Contingency_Type, 
                ax=axes[0], 
                vmin=-0, 
                vmax=max_type_margin, 
                annot=True, 
                linewidths=0.5)
    axes[0].set_title("SinkType AMI="+str(best_score_type.round(digits)))

    sns.heatmap(Contingency_Step, 
                ax=axes[1],
                vmin=0,
                vmax=max_steps_margin,
                robust=True,
                annot=True,
                linewidths=0.5)
    axes[1].set_title("StepNumber AMI="+str(best_score_step.round(digits)))
    fig.tight_layout(pad=2)
    fig.suptitle(decompositions_features[index])
    plt.show()

    # Go back on path
    chdir("..")
    print("}")
    print("\n\n")
    
