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


# %% Set experiment evaluation
np.random.seed(123)
min_iterations = 500
max_iterations = 1000
digits = 2
tol = 1e-5
decompositions_features = ["gray scale, rank=(32, 32, 32), frames=200",
                           "BGR, rank=(10, 10, 10, 2), frames=100",
                           "BGR, rank=(2, 2, 2, 2), frames=50"]
decompositions_paths = ["Decompositions", 
                        "Decompositions2",
                        "Decompositions3"]  # Folders where distinct decompositions are stored

print("Maximum running a total of", max_iterations, "iterations")

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

    # Confusion matrix for Type
    score = []
    convergence = []
    duration = []
    iterations = 0
    while(True):
        t_i = time()
        algorithm = kmeans.KMeans(Tensor_List, k=6)
        Clusters = algorithm.predict(verbose=True)
        Table = pd.DataFrame()
        Table["Type"] = reformat_string(pd.Series(name_list).str.split("_", expand=True))["Type"]
        Table["Clusters"] = Clusters
        temp_score = metrics.adjusted_mutual_info_score(Table["Type"], Table["Clusters"])
        if not score or temp_score > np.max(score):
            Contingency_Type = pd.crosstab(Table["Clusters"], Table["Type"], margins=True)
        if not score:
            old_avg = 0
        else:
            old_avg = np.mean(score)
        score.append(temp_score)
        new_avg = np.mean(score)
        convergence.append(algorithm.iterations)
        duration.append(time()-t_i)
        iterations+=1
        Stopper = abs(old_avg-new_avg) < tol or max_iterations <= iterations
        Stopper = Stopper and min_iterations < iterations
        if Stopper:
            if max_iterations <= iterations:
                print("CONVERGED RESULT NOT OBTAINED")
            break
    this_score = score
            
    # Print clustering  results
    print("Sink-Type results")
    print("Iteration count", iterations)
    print("Convergence average required steps", np.mean(convergence).round(digits))
    print("Average Adjusted Mutual Information", np.mean(score).round(digits))
    print("Standard Deviation AMI", np.std(score).round(digits))
    # noinspection PyUnboundLocalVariable
    print("Best contingency table", "score = "+str(np.max(score).round(digits)),
          Contingency_Type, sep="\n")
    best_score_type = np.max(score)
    print("Average duration", np.mean(duration).round(digits))
    print("Run lasted", np.sum(duration).round(digits))
    print()
    
    # Confusion matrix for Type
    score = []
    convergence = []
    duration = []
    iterations = 0
    while(True):
        t_i = time()
        algorithm = kmeans.KMeans(Tensor_List, k=3)
        Clusters = algorithm.predict(verbose=True)
        Table = pd.DataFrame()
        Table["Step"] = reformat_string(pd.Series(name_list).str.split("_", expand=True))["Step"]
        Table["Clusters"] = Clusters
        temp_score = metrics.adjusted_mutual_info_score(Table["Step"], Table["Clusters"])
        if not score or temp_score > np.max(score):
            Contingency_Step = pd.crosstab(Table["Clusters"], Table["Step"], margins=True)
        if not score:
            old_avg = 0
        else:
            old_avg = np.mean(score)
        score.append(temp_score)
        new_avg = np.mean(score)
        convergence.append(algorithm.iterations)
        duration.append(time()-t_i)
        iterations+=1
        Stopper = abs(old_avg-new_avg) < tol or max_iterations <= iterations
        Stopper = Stopper and min_iterations < iterations
        if Stopper:
            if max_iterations <= iterations:
                print("CONVERGED RESULT NOT OBTAINED")
            break
        
    # Print clustering  results
    print("Step-number results")
    print("Iteration count", iterations)
    print("Convergence average required steps", np.mean(convergence).round(digits))
    print("Average Adjusted Mutual Information", np.mean(score).round(digits))
    print("Standard Deviation AMI", np.std(score).round(digits))
    print("Best contingency table", "score = "+str(np.max(score).round(digits)),
          Contingency_Step, sep="\n")
    best_score_step = np.max(score)
    print("Average duration", np.mean(duration).round(digits))
    print("Run lasted", np.sum(duration).round(digits))

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
    print("\n\n")
    
