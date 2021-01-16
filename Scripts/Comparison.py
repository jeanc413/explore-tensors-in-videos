# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:19:40 2020

@author: JC
"""
# %% import packages
import pickle
import glob
import tensorly as tl
from TuckerFunction import TuckerFunction, sub_tensor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% define Tensor subset class
name_list  = glob.glob('*')
name_list = [n for n in name_list if "HandWash_" in n]
Tensor_List = []
for tensor in name_list:
    Handler = open(tensor,"rb")
    Tensor_List.append(pickle.load(Handler))

n=len(Tensor_List)

Comparisons = pd.DataFrame(columns=["Left","Right","Difference"],dtype=float)

for i in range(n):
    for j in range(n):
        if i == j:
            Append = pd.Series([name_list[i], name_list[j], 0], index=Comparisons.columns)
            Comparisons = Comparisons.append(Append, ignore_index=True)
        else:
            Append = pd.Series([name_list[i], name_list[j], tl.norm(Tensor_List[i].core - Tensor_List[j].core)], index=Comparisons.columns)
            Comparisons = Comparisons.append(Append, ignore_index=True)



def Label(Num):
    Num = int(Num)
    Label=""
    if Num<=5: Label="DarkSteelSink1"
    elif Num<=8: Label = "LightSteelSink1"
    elif Num<=11: Label = "CeramicSink1"
    elif Num<=14: Label = "SmallCeramicSink6"
    elif Num<=17: Label = "LightGraniteSink1"
    elif Num<=25: Label = "CeramicSink2"
    else: Label = "ERROR"
    return Label

def ReformatString(Data):
    temp = np.empty(len(Data), dtype=object)
    for i in range(len(Data)):
        temp[i]=Label(Data[1][i])
    Data["Type"] = temp
    Data["Step"] = Data[3]
    for i  in range(6):
        del Data[i]
    return Data
    



Left = ReformatString(Comparisons["Left"].str.split("_", expand=True))
Right = ReformatString(Comparisons["Right"].str.split("_", expand=True))

Comparisons["Left Type"] = Left["Type"] 
Comparisons["Left Step"] = Left["Step"]
Comparisons["Right Type"] = Right["Type"] 
Comparisons["Right Step"] = Right["Step"]


# %% View Results

# Histogram of difference
plt.hist(Comparisons["Difference"], color='pink', ec='black')
plt.title("Histogram of differences between videos")
plt.show()

# Boxplot 
Filter = (Comparisons["Difference"]!=0) & (Comparisons["Right Type"]==Comparisons["Left Type"])
sns.boxplot(data=Comparisons[Filter], y="Right Type", x="Difference").set_title("Boxplot for difference between same type of sink")
plt.show()

# Full comparison matrix
Comp_Matrix = pd.pivot_table(Comparisons, values='Difference', index='Left', columns='Right', aggfunc=np.mean)

# Comparing all from Step 01
Filter = (Comparisons["Left Step"] == '01') & (Comparisons["Right Step"] == '01')
Comp_Step_01 = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left', columns='Right', aggfunc=np.mean)

# Comparing all from DarkSteelSink1
Filter = (Comparisons["Left Type"] == 'DarkSteelSink1') & (Comparisons["Right Type"] == 'DarkSteelSink1')
Comp_DarkSteelSink1 = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left', columns='Right', aggfunc=np.mean)

# Comparing between steps
Comp_Step = pd.pivot_table(Comparisons, values='Difference', index='Left Step', columns='Right Step', aggfunc=np.mean)

# Comparing between Types
Comp_Type = pd.pivot_table(Comparisons, values='Difference', index='Left Type', columns='Right Type', aggfunc=np.mean)


# Heatmaps
sns.heatmap(Comp_Matrix, linewidths=0.5, yticklabels=True, xticklabels=True)
sns.heatmap(Comp_Type, linewidths=1)
sns.heatmap(Comp_Step, linewidths=1)

for step  in ["01","02","04"]:
    Filter = (Comparisons["Left Step"] == step) & (Comparisons["Right Step"] == step)
    Comp_Step_Type = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left Type', columns='Right Type', aggfunc=np.mean)
    sns.heatmap(Comp_Step_Type,linewidths=0.5).set(title="Type of sink average difference from step "+step)
    plt.show()

All_Types = pd.unique(Comparisons["Right Type"])

for Type  in All_Types:
    Filter = (Comparisons["Left Type"] == Type) & (Comparisons["Right Type"] == Type)
    Comp_Type_Step = pd.pivot_table(Comparisons[Filter], values='Difference', index='Left Step', columns='Right Step', aggfunc=np.mean)
    sns.heatmap(Comp_Type_Step,linewidths=0.5).set(title="Step average difference from "+Type)
    plt.show()


# %% Inverse engineering the decomposition
Tensor_List[0]
np.shape(Tensor_List[0].factors[0])
np.shape(Tensor_List[0].factors[1])
np.shape(Tensor_List[0].factors[2])
np.shape(Tensor_List[0].factors[3])
Tensor_List[0].factors[3]
Tensor_List[0].core

