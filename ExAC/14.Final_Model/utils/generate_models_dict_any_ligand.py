import numpy as np
import pickle
from collections import defaultdict
import pandas as pd

def generate_models_dict(ligand, classifier_method, ligands, ligands_positives_df, ligands_negatives_df, folds_num, no_features):
    """
    Create a dictionary of models with specific hyperparameters values for every ligand and fold combination.
    The function return the dictionary as output and also saves it in a pickle format.
    """
    models = ["XGB", "SVM", "RF", "Logistic", "NN"]
    
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
            classifiers["XGB"]["dna"] = {"mean_epoch_count": 1113, 
                                         "max_depth": 87, 
                                         "min_child_weight": 13.261564, 
                                         "colsample_bytree": 0.336990, 
                                         "scale_pos_weight": "balanced",
                                         "gamma": 0.001153, 
                                         "learning_rate": 0.045115,
                                          "trial_idx": 46} #46, any ligand-0.035, dna-0.240
        #==rna==#
        elif (ligand == "rna"):  
            ## for all the features
            classifiers["XGB"]["rna"] = {"mean_epoch_count": 846, 
                                         "max_depth": 88, 
                                         "min_child_weight": 1.851923, 
                                         "colsample_bytree": 0.295705, 
                                         "scale_pos_weight": 1,
                                         "gamma": 0.176914, 
                                         "learning_rate": 0.036385,
                                          "trial_idx": 55} #0.19

        #==ion==#
        elif (ligand == "ion"):
            classifiers["XGB"]["ion"] = {"mean_epoch_count": 188, 
                                         "max_depth": 94, 
                                         "min_child_weight": 1.254259, 
                                         "colsample_bytree": 0.481629, 
                                         "scale_pos_weight": 0.1,
                                         "gamma": 0.005971, 
                                         "learning_rate": 0.093677,
                                          "trial_idx": 85} #0.33
            
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["XGB"]["peptide"]= {"mean_epoch_count": 834, 
                                         "max_depth": 33, 
                                         "min_child_weight": 10.436966, 
                                         "colsample_bytree": 0.473196, 
                                         "scale_pos_weight": 1,
                                         "gamma": 0.006218, 
                                         "learning_rate": 0.051265,
                                          "trial_idx": 6} #0.024
            
        #==sm==#
        elif (ligand == "sm"):
            classifiers["XGB"]["sm"] = {"mean_epoch_count": 686, 
                                         "max_depth": 87, 
                                         "min_child_weight": 13.261564, 
                                         "colsample_bytree": 0.336990, 
                                         "scale_pos_weight": 1,
                                         "gamma": 0.001153, 
                                         "learning_rate": 0.045115,
                                          "trial_idx": 46} #0.19

    ###RF###
    elif (classifier_method == "RF"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["RF"]["dna"] = {"n_estimators": 1214,
                                           "max_depth": 80,
                                           "min_samples_leaf": 16, 
                                           "min_samples_split": 22,
                                           "class_weight": "balanced",
                                           "trial_idx": 23} #0.024
            
        #==rna==#
        elif (ligand == "rna"):
            classifiers["RF"]["rna"] = {"n_estimators": 749,
                                           "max_depth": 22,
                                           "min_samples_leaf": 18, 
                                           "min_samples_split": 29,
                                           "class_weight": "balanced",
                                           "trial_idx": 33} #0.159
        #==ion==#
        elif (ligand == "ion"):
            classifiers["RF"]["ion"] = {"n_estimators": 749,
                                           "max_depth": 22,
                                           "min_samples_leaf": 18, 
                                           "min_samples_split": 29,
                                           "class_weight": "balanced",
                                           "trial_idx": 33} #0.26
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["RF"]["peptide"] = {"n_estimators": 983,
                                           "max_depth": 23,
                                           "min_samples_leaf": 10, 
                                           "min_samples_split": 2,
                                           "class_weight": "balanced",
                                           "trial_idx": 29} #0.025
 
        #==sm==#
        elif (ligand == "sm"):
            classifiers["RF"]["sm"] = {"n_estimators": 920,
                                           "max_depth": 41,
                                           "min_samples_leaf": 33, 
                                           "min_samples_split": 3,
                                           "class_weight": "balanced",
                                           "trial_idx": 10} #0.19
    
    ###SVM###
    elif (classifier_method == "SVM"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["SVM"]["dna"] = {"C": 0.413600,
                                         "gamma": 0.001512,
                                         "kernel": "rbf",
                                         "class_weight": "balanced",
                                         "trial_idx": 1} #0.033
        #==rna==#
        elif (ligand == "rna"):
            classifiers["SVM"]["rna"] = {"C": 0.054563,
                                         "gamma": 0.001878,
                                         "kernel": "rbf",
                                         "class_weight": "balanced",
                                         "trial_idx": 16} #0.26
        #==ion==#
        elif (ligand == "ion"):
            classifiers["SVM"]["ion"] = {"C": 0.413600,
                                         "gamma": 0.001512,
                                         "kernel": "rbf",
                                         "class_weight": "balanced",
                                         "trial_idx": 1} #0.23
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["SVM"]["peptide"] = {"C": 2.737189,
                                         "gamma": 0.000916,
                                         "kernel": "rbf",
                                         "class_weight": "balanced",
                                         "trial_idx": 97} #0.020
        #==sm==#
        elif (ligand == "sm"):
            classifiers["SVM"]["sm"] = {"C": 0.914200,
                                         "gamma": 0.000145,
                                         "kernel": "rbf",
                                         "class_weight": "balanced",
                                         "trial_idx": 64} #0.15

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