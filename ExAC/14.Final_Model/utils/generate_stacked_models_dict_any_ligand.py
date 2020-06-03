import numpy as np
import pickle
from collections import defaultdict
import pandas as pd

def generate_models_dict(ligand, classifier_method, ligands, ligands_positives_df, ligands_negatives_df, folds_num, no_features):
    """
    Create a dictionary of models with specific hyperparameters values for every ligand and fold combination.
    The function return the dictionary as output and also saves it in a pickle format.
    """
    models = ["XGB", "RF", "SVM", "Logistic", "NN"]
    
    #For balanced weight of XGB
    ligand_pos = ligands_positives_df[ligand].shape[0]
    ligand_neg = ligands_negatives_df[ligand].shape[0]
    scale_weight = ligand_neg/float(ligand_pos) 

    #Initialize classifier dict
    classifiers = defaultdict(dict)
    #classifiers[classifier_method][ligand] = dict.fromkeys(np.arange(1,folds_num+1))
            
    #For input size of NN
    features_num = ligands_positives_df[ligand].shape[1]


    #Update the specific hyperparameters values

    ###XGB###
    if (classifier_method == "XGB"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["XGB"]["dna"] = {"mean_epoch_count": 221, 
                                         "max_depth": 98, 
                                         "min_child_weight": 3.886844, 
                                         "colsample_bytree": 0.795993, 
                                         "scale_pos_weight": "balanced",
                                         "gamma": 0.837097, 
                                         "learning_rate": 0.104252,
                                         "models": models,
                                         "ligands": ["dna"],
                                          "trial_idx": 50} #0.058
            
        #==rna==#
        elif (ligand == "rna"):  
            ## for all the features
            classifiers["XGB"]["rna"] = {"mean_epoch_count": 1260, 
                                         "max_depth": 22, 
                                         "min_child_weight": 2.621105, 
                                         "colsample_bytree": 0.110833, 
                                         "scale_pos_weight": 1,
                                         "gamma": 0.002221, 
                                         "learning_rate": 0.014611,
                                         "models": ["XGB"],
                                         "ligands": ligands,
                                          "trial_idx": 74} #0.207

        #==ion==#
        elif (ligand == "ion"):
            classifiers["XGB"]["ion"] = {"mean_epoch_count": 1536, 
                                         "max_depth": 84, 
                                         "min_child_weight": 3.592074, 
                                         "colsample_bytree": 0.122211, 
                                         "scale_pos_weight": 1,
                                         "gamma": 0.001591, 
                                         "learning_rate": 0.023871,
                                         "models": ["XGB"],
                                         "ligands": ligands,
                                          "trial_idx": 36} #0.28
            
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["XGB"]["peptide"]= {"mean_epoch_count": 1086, 
                                         "max_depth": 79, 
                                         "min_child_weight": 8.291350, 
                                         "colsample_bytree": 0.256701, 
                                         "scale_pos_weight": 1,
                                         "gamma": 0.026661, 
                                         "learning_rate": 0.055305,
                                         "models": ["XGB"],
                                         "ligands": ligands,
                                          "trial_idx": 62} #0.022
            
        #==sm==#
        #elif (ligand == "sm"):
            #classifiers["XGB"]["sm"] = {"mean_epoch_count": 48, 
                                         #"max_depth": 24, 
                                         #"min_child_weight": 18.827554, 
                                         #"colsample_bytree": 0.819282, 
                                         #"scale_pos_weight": 0.1,
                                         #"gamma": 0.077865, 
                                         #"learning_rate": 0.114817,
                                         #"models": models,
                                         #"ligands": ligands,
                                         #"trial_idx": 49} #0.22
        elif (ligand == "sm"):
            classifiers["XGB"]["sm"] = {"mean_epoch_count": 1000, 
                                         "max_depth": 92, 
                                         "min_child_weight": 15.770910, 
                                         "colsample_bytree": 0.151163, 
                                         "scale_pos_weight": 1,
                                         "gamma": 0.123308, 
                                         "learning_rate": 0.053142,
                                         "models": models,
                                         "ligands": ligands,
                                          "trial_idx": 59} #0.22

    ###Logistic###
    elif (classifier_method == "Logistic"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["Logistic"]["dna"] = {"C": 0.056808,
                                              "class_weight": None,
                                              "trial_idx": 51} #0.030
            
        #==rna==#
        elif (ligand == "rna"):
            classifiers["Logistic"]["rna"] = {"C": 0.018704,
                                              "class_weight": {0: 10, 1: 1},
                                              "trial_idx": 80} #0.26
        #==ion==#
        elif (ligand == "ion"):
            classifiers["Logistic"]["ion"] = {"C": 0.003279,
                                              "class_weight": None,
                                              "trial_idx": 47} #0.34
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["Logistic"]["peptide"] = {"C": 0.003279,
                                                  "class_weight": None,
                                                  "trial_idx": 47} #0.018
        #==sm==#
        elif (ligand == "sm"):
            classifiers["Logistic"]["sm"] = {"C": 0.002985,
                                             "class_weight": {0: 10, 1: 1},
                                             "trial_idx": 79} #0.20
   
    ###NN###
    elif (classifier_method == "NN"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["NN"]["dna"] = {"hidden_units_1": 380,
                                        "hidden_units_2": 493,
                                        "batch_size": 77,
                                        "learning_rate": 0.000095,
                                        "beta": 0.830649,
                                        "weight_decay": 1.504995e-21,
                                        "mean_epoch_count": 7,
                                        "weight": "0.1",
                                        "trial_idx": 10}
            
        #==rna==#
        elif (ligand == "rna"):
            classifiers["NN"]["rna"] = {"hidden_units_1": 380,
                                        "hidden_units_2": 493,
                                        "batch_size": 77,
                                        "learning_rate": 0.000095,
                                        "beta": 0.830649,
                                        "weight_decay": 1.504995e-21,
                                        "mean_epoch_count": 7,
                                        "weight": "0.1",
                                        "trial_idx": 10} #0.24

        #==ion==#
        elif (ligand == "ion"):
            classifiers["NN"]["ion"] = {"hidden_units_1": 328,
                                        "hidden_units_2": 625,
                                        "batch_size": 288,
                                        "learning_rate": 0.000104,
                                        "beta": 0.936102,
                                        "weight_decay": 4.324713e-23,
                                        "mean_epoch_count": 24,
                                        "weight": "None",
                                        "trial_idx": 17} #0.22

        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["NN"]["peptide"] = {"hidden_units_1": 559,
                                        "hidden_units_2": 359,
                                        "batch_size": 147,
                                        "learning_rate": 0.000125,
                                        "beta": 0.963010,
                                        "weight_decay": 7.679182e-09,
                                        "mean_epoch_count": 49,
                                        "weight": "None",
                                        "trial_idx": 0} #0.016
        #==sm==#
        elif (ligand == "sm"):
            classifiers["NN"]["sm"] = {"hidden_units_1": 266,
                                        "hidden_units_2": 806,
                                        "batch_size": 191,
                                        "learning_rate": 0.000008,
                                        "beta": 0.806328,
                                        "weight_decay": 9.566788e-09,
                                        "mean_epoch_count": 137,
                                        "weight": "0.1",
                                        "trial_idx": 72} #0.18
    
    return classifiers