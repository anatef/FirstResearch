import numpy as np
import pickle
from collections import defaultdict
import pandas as pd

#Import scikit-learn classifiers
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import shuffle

#Import Net class
from NN_classes import Net
#====================================================================================================================#


def generate_models_dict(ligand, classifier_method, ligand_positives, ligand_negatives, folds_num, no_features):
    """
    Create a dictionary of models with specific hyperparameters values for every ligand and fold combination.
    The function return the dictionary as output and also saves it in a pickle format.
    """
    models = ["XGB", "SVM", "RF", "ADA", "KNN", "Logistic", "NN"]
    
    #For balanced weight of XGB
    ligand_pos = ligand_positives.shape[0]
    ligand_neg = ligand_negatives.shape[0]
    scale_weight = ligand_neg/float(ligand_pos) 

    #Initialize classifier dict
    classifiers = defaultdict(dict)
    classifiers[classifier_method][ligand] = dict.fromkeys(np.arange(1,folds_num+1))
            
    #For input size of NN
    features_num = ligand_positives.shape[1]


    #Update the specific hyperparameters values

    ###XGB###
    if (classifier_method == "XGB"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["XGB"]["dna"][1] = XGBClassifier(n_estimators=46, n_jobs=-1, random_state=0, max_depth=15, min_child_weight=3.377195, colsample_bytree=0.812449, 
                                                       scale_pos_weight=1, gamma=0.004231, learning_rate=0.404226) #24, 0.95, 0.71
            classifiers["XGB"]["dna"][2] = XGBClassifier(n_estimators=14, n_jobs=-1, random_state=0, max_depth=2, min_child_weight=3.114230, colsample_bytree=0.918415, 
                                                       scale_pos_weight=1, gamma=0.087880, learning_rate=0.235029) #14, 0.90, 0.78 
            classifiers["XGB"]["dna"][3] = XGBClassifier(n_estimators=16, n_jobs=-1, random_state=0, max_depth=2, min_child_weight=4.893092, colsample_bytree=0.949790, 
                                                       scale_pos_weight=scale_weight, gamma=0.008374, learning_rate=0.128065) #4, 0.87, 0.63
            classifiers["XGB"]["dna"][4] = XGBClassifier(n_estimators=12, n_jobs=-1, random_state=0, max_depth=15, min_child_weight=2.461341, colsample_bytree=0.928890, 
                                                       scale_pos_weight=1, gamma=0.009383, learning_rate=0.081862) #57, 0.72, 0.56
            classifiers["XGB"]["dna"][5] = XGBClassifier(n_estimators=10, n_jobs=-1, random_state=0, max_depth=10, min_child_weight=1.903646, colsample_bytree=0.908483, 
                                                       scale_pos_weight=1, gamma=0.071410, learning_rate=0.202134) #89, 0.81, 0.47

        #==rna==#
        elif (ligand == "rna"):  
            ## for all the features
            classifiers["XGB"]["rna"][1] = XGBClassifier(n_estimators=33, n_jobs=-1, random_state=0, max_depth=3, min_child_weight=0.971711, colsample_bytree=0.943332, 
                                                       scale_pos_weight=scale_weight, gamma=0.088821, learning_rate=0.212930) #58, 0.94, 0.79
            classifiers["XGB"]["rna"][2] = XGBClassifier(n_estimators=7, n_jobs=-1, random_state=0, max_depth=19, min_child_weight=1.489209, colsample_bytree=0.850283, 
                                                       scale_pos_weight=scale_weight, gamma=0.001743, learning_rate=0.444518) #60, 0.98, 0.88
            classifiers["XGB"]["rna"][3] = XGBClassifier(n_estimators=5, n_jobs=-1, random_state=0, max_depth=5, min_child_weight=4.591177, colsample_bytree=0.804206, 
                                                      scale_pos_weight=1, gamma=0.013501, learning_rate=0.216659) #17, 0.82, 0.58
            classifiers["XGB"]["rna"][4] = XGBClassifier(n_estimators=13, n_jobs=-1, random_state=0, max_depth=15, min_child_weight=3.377195, colsample_bytree=0.812449, 
                                                       scale_pos_weight=1, gamma=0.004231, learning_rate=0.404226) #24, 0.83, 0.47
            classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=3, n_jobs=-1, random_state=0, max_depth=16, min_child_weight=3.603163, colsample_bytree=0.895505, 
                                                       scale_pos_weight=1, gamma=0.011878, learning_rate=0.111754) #5, 0.73, 0.27
            
        #==ion==#
        elif (ligand == "ion"):
            classifiers["XGB"]["ion"][1] = XGBClassifier(n_estimators=19, n_jobs=-1, random_state=0, max_depth=2, min_child_weight=2.417043, colsample_bytree=0.947185, 
                                                       scale_pos_weight=1, gamma=0.002877, learning_rate=0.238124) #47, 0.74, 0.70
            classifiers["XGB"]["ion"][2] = XGBClassifier(n_estimators=49, n_jobs=-1, random_state=0, max_depth=18, min_child_weight=3.517474, colsample_bytree=0.795408, 
                                                      scale_pos_weight=1, gamma=0.010587, learning_rate=0.036915) #55, 0.86, 0.76
            classifiers["XGB"]["ion"][3] = XGBClassifier(n_estimators=18, n_jobs=-1, random_state=0, max_depth=5, min_child_weight=4.591177, colsample_bytree=0.804206, 
                                                       scale_pos_weight=1, gamma=0.013501, learning_rate=0.216659) #17, 0.85, 0.73 
            classifiers["XGB"]["ion"][4] = XGBClassifier(n_estimators=39, n_jobs=-1, random_state=0, max_depth=5, min_child_weight=0.905755, colsample_bytree=0.947136, 
                                                       scale_pos_weight=1, gamma=0.001299, learning_rate=0.076189) #68, 0.76, 0.72
            classifiers["XGB"]["ion"][5] = XGBClassifier(n_estimators=40, n_jobs=-1, random_state=0, max_depth=10, min_child_weight=4.480192, colsample_bytree=0.909730, 
                                                       scale_pos_weight=1, gamma=0.060689, learning_rate=0.068573) #22, 0.81, 0.65
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["XGB"]["peptide"][1] = XGBClassifier(n_estimators=129, n_jobs=-1, random_state=0, max_depth=25, min_child_weight=9.617871, colsample_bytree=0.922165, 
                                                       scale_pos_weight=1, gamma=0.057670, learning_rate=0.163615) #17, 0.64, 0.20
            classifiers["XGB"]["peptide"][2] = XGBClassifier(n_estimators=34, n_jobs=-1, random_state=0, max_depth=7, min_child_weight=4.374987, colsample_bytree=0.892393, 
                                                       scale_pos_weight=1, gamma=0.008021, learning_rate=0.248787) #42, 0.70, 0.42 
            classifiers["XGB"]["peptide"][3] = XGBClassifier(n_estimators=58, n_jobs=-1, random_state=0, max_depth=47, min_child_weight=9.845365, colsample_bytree=0.928890, 
                                                       scale_pos_weight=scale_weight, gamma=0.009383, learning_rate=0.030194) #60, 0.67, 0.58
            classifiers["XGB"]["peptide"][4] = XGBClassifier(n_estimators=152, n_jobs=-1, random_state=0, max_depth=13, min_child_weight=6.165723, colsample_bytree=0.808216, 
                                                       scale_pos_weight=scale_weight, gamma=0.010159, learning_rate=0.217783) #81, 0.67, 0.51
            classifiers["XGB"]["peptide"][5] = XGBClassifier(n_estimators=101, n_jobs=-1, random_state=0, max_depth=25, min_child_weight=6.760152, colsample_bytree=0.918688, 
                                                       scale_pos_weight=scale_weight, gamma=0.004309, learning_rate=0.052992) #13, 0.58, 0.45
 
        #==sm==#
        elif (ligand == "sm"):
            classifiers["XGB"]["sm"][1] = XGBClassifier(n_estimators=17, n_jobs=-1, random_state=0, max_depth=20, min_child_weight=2.609242, colsample_bytree=0.853665, 
                                                       scale_pos_weight=1, gamma=0.003382, learning_rate=0.123149) #6, 0.81, 0.81
            classifiers["XGB"]["sm"][2] = XGBClassifier(n_estimators=58, n_jobs=-1, random_state=0, max_depth=38, min_child_weight=2.878756, colsample_bytree=0.955192, 
                                                       scale_pos_weight=1, gamma=0.065718, learning_rate=0.159189) #84, 0.81, 0.78
            classifiers["XGB"]["sm"][3] = XGBClassifier(n_estimators=17, n_jobs=-1, random_state=0, max_depth=25, min_child_weight=0.472290, colsample_bytree=0.908402, 
                                                       scale_pos_weight=1, gamma=0.079176, learning_rate=0.338233) #85, 0.71, 0.53
            classifiers["XGB"]["sm"][4] = XGBClassifier(n_estimators=41, n_jobs=-1, random_state=0, max_depth=38, min_child_weight=2.878756, colsample_bytree=0.955192, 
                                                       scale_pos_weight=1, gamma=0.065718, learning_rate=0.159189) #84, 0.64, 0.71
            classifiers["XGB"]["sm"][5] = XGBClassifier(n_estimators=10, n_jobs=-1, random_state=0, max_depth=44, min_child_weight=3.791893, colsample_bytree=0.830004, 
                                                       scale_pos_weight=1, gamma=0.005847, learning_rate=0.038766) #47, 0.73, 0.74
    
    return classifiers