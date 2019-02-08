import numpy as np
import pickle
from collections import defaultdict
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.utils import shuffle

#Import from other utils files
from NN_classes import Net

def generate_models_dict(ligand, classifier_method, ligands, ligands_positives_df, ligands_negatives_df, folds_num, no_features):
    """
    Create a dictionary of models with specific hyperparameters values for every ligand and fold combination.
    The function return the dictionary as output and also saves it in a pickle format.
    """
    models = ["XGB", "SVM", "RF", "ADA", "KNN", "Logistic", "NN"]
    
    #For balanced weight of XGB
    ligand_pos = ligands_positives_df[ligand].shape[0]
    ligand_neg = ligands_negatives_df[ligand].shape[0]
    scale_weight = ligand_neg/float(ligand_pos) 

    #Initialize classifier dict
    classifiers = defaultdict(dict)
    classifiers[classifier_method][ligand] = dict.fromkeys(np.arange(1,folds_num+1))
            
    #For input size of NN
    features_num = ligands_positives_df[ligand].shape[1]


    #Update the specific hyperparameters values

    ###XGB###
    if (classifier_method == "XGB"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["XGB"]["dna"][1] = XGBClassifier(n_estimators=1242, n_jobs=-1, random_state=0, max_depth=24, min_child_weight=1.690038, colsample_bytree= 0.756064, 
                                                       scale_pos_weight=scale_weight, gamma=0.008946, learning_rate=0.052992) #
            classifiers["XGB"]["dna"][2] = XGBClassifier(n_estimators=4224, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270, colsample_bytree=0.372216, 
                                                       scale_pos_weight=scale_weight, gamma=0.082468, learning_rate=0.005201) #
            classifiers["XGB"]["dna"][3] = XGBClassifier(n_estimators=4999, n_jobs=-1, random_state=0, max_depth=18, min_child_weight=4.363253, colsample_bytree=0.761365, 
                                                       scale_pos_weight=scale_weight, gamma=0.011982, learning_rate=0.0033856) #
            classifiers["XGB"]["dna"][4] = XGBClassifier(n_estimators=1802, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.718740, colsample_bytree=0.313751, 
                                                       scale_pos_weight=1, gamma=0.001920, learning_rate=0.016505) #
            classifiers["XGB"]["dna"][5] = XGBClassifier(n_estimators=1659, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=4.962148, colsample_bytree=0.315347, 
                                                       scale_pos_weight=scale_weight, gamma=0.001150, learning_rate=0.082067) #

        #==rna==#
        elif (ligand == "rna"):  
            ## for all the features
            classifiers["XGB"]["rna"][1] = XGBClassifier(n_estimators=138, n_jobs=-1, random_state=0, max_depth=3, min_child_weight=2.541577, colsample_bytree=0.375136, 
                                                       scale_pos_weight=scale_weight, gamma=0.217347, learning_rate=0.106480) #
            classifiers["XGB"]["rna"][2] = XGBClassifier(n_estimators=195, n_jobs=-1, random_state=0, max_depth=76, min_child_weight=1.541431, colsample_bytree=0.424649, 
                                                       scale_pos_weight=scale_weight, gamma=0.032380, learning_rate=0.217783) #
            classifiers["XGB"]["rna"][3] = XGBClassifier(n_estimators=1266, n_jobs=-1, random_state=0, max_depth=24, min_child_weight=1.690038, colsample_bytree=0.756064, 
                                                      scale_pos_weight=scale_weight, gamma=0.008946, learning_rate=0.052992) #
            classifiers["XGB"]["rna"][4] = XGBClassifier(n_estimators=3882, n_jobs=-1, random_state=0, max_depth=83, min_child_weight=0.644631, colsample_bytree=0.486571, 
                                                       scale_pos_weight=scale_weight, gamma=0.012335, learning_rate=0.009901) #
            classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=1323, n_jobs=-1, random_state=0, max_depth=20, min_child_weight=3.394398, colsample_bytree=0.790474, 
                                                       scale_pos_weight=0.1, gamma=0.055726, learning_rate=0.007600) #trying to improve auprc_ratio
            #classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=396, n_jobs=-1, random_state=0, max_depth=20, min_child_weight=1.980491, colsample_bytree=0.922029, 
                                                       #scale_pos_weight=scale_weight, gamma=0.082559, learning_rate=0.131959) #
            
        #==ion==#
        elif (ligand == "ion"):
            classifiers["XGB"]["ion"][1] = XGBClassifier(n_estimators=1809, n_jobs=-1, random_state=0, max_depth=85, min_child_weight=3.171370, colsample_bytree=0.969212, 
                                                       scale_pos_weight=scale_weight, gamma=0.090860, learning_rate=0.016700) #
            classifiers["XGB"]["ion"][2] = XGBClassifier(n_estimators=1076, n_jobs=-1, random_state=0, max_depth=56, min_child_weight=3.437441, colsample_bytree=0.411631, 
                                                      scale_pos_weight=0.1, gamma=0.695203, learning_rate=0.036141) #
            classifiers["XGB"]["ion"][3] = XGBClassifier(n_estimators=4134, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270, colsample_bytree=0.372216, 
                                                       scale_pos_weight=scale_weight, gamma=0.082468, learning_rate=0.005201) #
            classifiers["XGB"]["ion"][4] = XGBClassifier(n_estimators=234, n_jobs=-1, random_state=0, max_depth=40, min_child_weight=2.964223, colsample_bytree=0.883199, 
                                                       scale_pos_weight=0.1, gamma=0.374832, learning_rate=0.092338) #
            classifiers["XGB"]["ion"][5] = XGBClassifier(n_estimators=1184, n_jobs=-1, random_state=0, max_depth=56, min_child_weight=3.437441, colsample_bytree=0.411631, 
                                                       scale_pos_weight=0.1, gamma=0.695203, learning_rate=0.036141) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["XGB"]["peptide"][1] = XGBClassifier(n_estimators=665, n_jobs=-1, random_state=0, max_depth=35, min_child_weight=1.119623, colsample_bytree=0.509014, 
                                                       scale_pos_weight=0.1, gamma=0.608477, learning_rate=0.029205) #
            classifiers["XGB"]["peptide"][2] = XGBClassifier(n_estimators=480, n_jobs=-1, random_state=0, max_depth=88, min_child_weight=0.462981, colsample_bytree=0.413088, 
                                                       scale_pos_weight=1, gamma=0.176914, learning_rate=0.036385) #
            classifiers["XGB"]["peptide"][3] = XGBClassifier(n_estimators=948, n_jobs=-1, random_state=0, max_depth=24, min_child_weight=1.690038, colsample_bytree=0.756064, 
                                                       scale_pos_weight=scale_weight, gamma=0.008946, learning_rate=0.052992) #
            classifiers["XGB"]["peptide"][4] = XGBClassifier(n_estimators=2049, n_jobs=-1, random_state=0, max_depth=87, min_child_weight=1.568462, colsample_bytree=0.968088, 
                                                       scale_pos_weight=scale_weight, gamma=0.026017, learning_rate=0.030942) #
            classifiers["XGB"]["peptide"][5] = XGBClassifier(n_estimators=2790, n_jobs=-1, random_state=0, max_depth=29, min_child_weight=4.197565, colsample_bytree=0.641778, 
                                                       scale_pos_weight=0.1, gamma=0.034612, learning_rate=0.018811) #
 
        #==sm==#
        elif (ligand == "sm"):
            classifiers["XGB"]["sm"][1] = XGBClassifier(n_estimators=494, n_jobs=-1, random_state=0, max_depth=35, min_child_weight=1.119623, colsample_bytree=0.509014, 
                                                       scale_pos_weight=0.1, gamma=0.608477, learning_rate=0.029205) #
            classifiers["XGB"]["sm"][2] = XGBClassifier(n_estimators=3166, n_jobs=-1, random_state=0, max_depth=25, min_child_weight=3.577806, colsample_bytree=0.614620, 
                                                       scale_pos_weight=scale_weight, gamma=0.133550, learning_rate=0.005540) #
            classifiers["XGB"]["sm"][3] = XGBClassifier(n_estimators=3858, n_jobs=-1, random_state=0, max_depth=46, min_child_weight=2.045270476865308, colsample_bytree=0.372216, 
                                                       scale_pos_weight=scale_weight, gamma=0.082468, learning_rate=0.005201) #
            classifiers["XGB"]["sm"][4] = XGBClassifier(n_estimators=3625, n_jobs=-1, random_state=0, max_depth=83, min_child_weight=0.644631, colsample_bytree=0.486571, 
                                                       scale_pos_weight=scale_weight, gamma=0.012335, learning_rate=0.009901) #
            classifiers["XGB"]["sm"][5] = XGBClassifier(n_estimators=3174, n_jobs=-1, random_state=0, max_depth=83, min_child_weight=0.644631, colsample_bytree=0.486571, 
                                                       scale_pos_weight=scale_weight, gamma=0.012335, learning_rate=0.009901) #
    
    ###RF###
    elif (classifier_method == "RF"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["RF"]["dna"][1] = RandomForestClassifier(n_estimators=1415, max_depth=26, min_samples_leaf=4, min_samples_split=20, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][2] = RandomForestClassifier(n_estimators=920, max_depth=41, min_samples_leaf=33, min_samples_split=3, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][3] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][4] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][5] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["RF"]["rna"][1] = RandomForestClassifier(n_estimators=951, max_depth=89, min_samples_leaf=2, min_samples_split=8, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][2] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][3] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][4] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][5] = RandomForestClassifier(n_estimators=951, max_depth=89, min_samples_leaf=2, min_samples_split=8, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["RF"]["ion"][1] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][2] = RandomForestClassifier(n_estimators=1154, max_depth=57, min_samples_leaf=1, min_samples_split=13, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][3] = RandomForestClassifier(n_estimators=610, max_depth=14, min_samples_leaf=2, min_samples_split=40, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][4] = RandomForestClassifier(n_estimators=749, max_depth=22, min_samples_leaf=18, min_samples_split=29, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][5] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["RF"]["peptide"][1] = RandomForestClassifier(n_estimators=829, max_depth=20, min_samples_leaf=35, min_samples_split=32, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][2] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][3] = RandomForestClassifier(n_estimators=565, max_depth=34, min_samples_leaf=12, min_samples_split=42, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][4] = RandomForestClassifier(n_estimators=765, max_depth=71, min_samples_leaf=16, min_samples_split=49, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][5] = RandomForestClassifier(n_estimators=610, max_depth=14, min_samples_leaf=2, min_samples_split=40, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["RF"]["sm"][1] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][2] = RandomForestClassifier(n_estimators=565, max_depth=34, min_samples_leaf=12, min_samples_split=42, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][3] = RandomForestClassifier(n_estimators=518, max_depth=37, min_samples_leaf=31, min_samples_split=31, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][4] = RandomForestClassifier(n_estimators=1310, max_depth=84, min_samples_leaf=5, min_samples_split=24, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][5] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
    
    
    ###SVM###
    elif (classifier_method == "SVM"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["SVM"]["dna"][1] = SVC(kernel='rbf', class_weight=None, C=0.062786, gamma=0.001267, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.510614, gamma=0.001933, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][3] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][4] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][5] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["SVM"]["rna"][1] = SVC(kernel='rbf', class_weight=None, C=10.657846, gamma=0.000194, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][2] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=2.402881, gamma=0.000863, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][3] = SVC(kernel='poly', class_weight={0: 10, 1: 1}, C=1.533865, gamma=0.000152, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][4] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=2.402881, gamma=0.000863, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][5] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=0.024509, gamma=0.000426, probability=True, random_state=0, cache_size=200) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["SVM"]["ion"][1] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][3] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][4] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][5] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["SVM"]["peptide"][1] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C= 2.726349, gamma=0.000131, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][3] = SVC(kernel='rbf', class_weight="balanced", C=85.161446, gamma=0.000202, probability=True, random_state=0, cache_size=200) # 
            classifiers["SVM"]["peptide"][4] = SVC(kernel='rbf', class_weight="balanced", C=0.062786, gamma=0.000541, probability=True, random_state=0, cache_size=200) # 
            classifiers["SVM"]["peptide"][5] = SVC(kernel='rbf', class_weight="balanced", C=8.432559, gamma=0.000196, probability=True, random_state=0, cache_size=200) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["SVM"]["sm"][1] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][2] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) # 
            classifiers["SVM"]["sm"][3] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][4] = SVC(kernel='rbf', class_weight="balanced", C=1.273404, gamma=0.000196, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][5] = SVC(kernel='rbf', class_weight="balanced", C=0.076938, gamma=0.000517, probability=True, random_state=0, cache_size=200) # 

    ###Logistic###
    elif (classifier_method == "Logistic"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["Logistic"]["dna"][1] = LogisticRegression(C=0.001741, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][2] = LogisticRegression(C=0.001189, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") # 
            classifiers["Logistic"]["dna"][4] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") # 
            classifiers["Logistic"]["dna"][5] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") # 
        #==rna==#
        elif (ligand == "rna"):
            classifiers["Logistic"]["rna"][1] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][2] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][4] = LogisticRegression(C=0.001203, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["rna"][5] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["Logistic"]["ion"][1] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["ion"][2] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["ion"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["ion"][4] = LogisticRegression(C=0.001203, random_state=0, n_jobs=-1, class_weight="balanced") # 
            classifiers["Logistic"]["ion"][5] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["Logistic"]["peptide"][1] = LogisticRegression(C=0.003279, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["peptide"][2] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["peptide"][3] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["peptide"][4] = LogisticRegression(C=0.001805, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["peptide"][5] = LogisticRegression(C=0.001805, random_state=0, n_jobs=-1, class_weight="balanced") #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["Logistic"]["sm"][1] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["sm"][2] = LogisticRegression(C=0.002423, random_state=0, n_jobs=-1, class_weight=None) # 
            classifiers["Logistic"]["sm"][3] = LogisticRegression(C=0.002231, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["sm"][4] = LogisticRegression(C=0.001044, random_state=0, n_jobs=-1, class_weight="balanced") #
            classifiers["Logistic"]["sm"][5] = LogisticRegression(C=0.002423, random_state=0, n_jobs=-1, class_weight=None) #
    
    ###NN###
    elif (classifier_method == "NN"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["NN"]["dna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=935, hidden_units_2=566, batch_size=196, learning_rate=0.000002, beta=0.912266,
                                              weight_decay=2.507831e-14, epoch_count=400, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=457, hidden_units_2=671, batch_size=135, learning_rate=0.000003, beta=0.916482,
                                              weight_decay=9.976996e-17, epoch_count=1223, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=539, hidden_units_2=624, batch_size=295, learning_rate=0.000006, beta=0.975980,
                                              weight_decay=3.194446e-10, epoch_count=372, weight="balanced", input_size=no_features) #
            classifiers["NN"]["dna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=890, hidden_units_2=642, batch_size=125, learning_rate=0.000002, beta=0.818449,
                                              weight_decay=2.629448e-09, epoch_count=316, weight=None, input_size=no_features) #
            classifiers["NN"]["dna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=939, hidden_units_2=498, batch_size=252, learning_rate=0.000004, beta=0.931570,
                                              weight_decay=1.918448e-24, epoch_count=822, weight=None, input_size=no_features) #  
        #==rna==#
        elif (ligand == "rna"):
            classifiers["NN"]["rna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=539, hidden_units_2=624, batch_size=295, learning_rate=0.000006, beta=0.975980, 
                                              weight_decay=3.194446e-10, epoch_count=488, weight="balanced", input_size=no_features) #
            classifiers["NN"]["rna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=930, hidden_units_2=744, batch_size=103, learning_rate=0.000002, beta=0.886173, 
                                              weight_decay=8.883323e-12, epoch_count=507, weight=None, input_size=no_features) #
            classifiers["NN"]["rna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=708, hidden_units_2=799, batch_size=188, learning_rate=0.000003, beta=0.867566, 
                                              weight_decay=5.769333e-16, epoch_count=1197, weight=0.1, input_size=no_features) #
            classifiers["NN"]["rna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=578, hidden_units_2=639, batch_size=42, learning_rate=0.000001, beta=0.912524, 
                                              weight_decay=3.136689e-20, epoch_count=909, weight=0.1, input_size=no_features) #
            classifiers["NN"]["rna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=982, hidden_units_2=610, batch_size=278, learning_rate=0.000014, beta=0.981022, 
                                              weight_decay=2.894894e-21, epoch_count=63, weight=None, input_size=no_features) #

        #==ion==#
        elif (ligand == "ion"):
            classifiers["NN"]["ion"][1] = Net(dropout_parameter = 0.5, hidden_units_1=347, hidden_units_2=934, batch_size=123, learning_rate=0.001957, beta=0.919507,
                                              weight_decay=4.178006e-09, epoch_count=7, weight=None, input_size=no_features) #
            classifiers["NN"]["ion"][2] = Net(dropout_parameter = 0.5, hidden_units_1=732, hidden_units_2=558, batch_size=33, learning_rate=0.001095, beta=0.932919,
                                              weight_decay=1.035851e-10, epoch_count=5, weight=0.1, input_size=no_features) #
            classifiers["NN"]["ion"][3] = Net(dropout_parameter = 0.5, hidden_units_1=605, hidden_units_2=551, batch_size=113, learning_rate=0.008970, beta=0.804689,
                                              weight_decay=3.909071e-22, epoch_count=23, weight=None, input_size=no_features) #
            classifiers["NN"]["ion"][4] = Net(dropout_parameter = 0.5, hidden_units_1=496, hidden_units_2=678, batch_size=124, learning_rate=0.001378, beta=0.982816,
                                              weight_decay=5.056854e-11, epoch_count=5, weight="balanced", input_size=no_features) #
            classifiers["NN"]["ion"][5] = Net(dropout_parameter = 0.5, hidden_units_1=866, hidden_units_2=986, batch_size=83, learning_rate=0.006419, beta=0.939022,
                                              weight_decay=9.662474e-11, epoch_count=8, weight="balanced", input_size=no_features) #

        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["NN"]["peptide"][1] = Net(dropout_parameter = 0.5, hidden_units_1=328, hidden_units_2=625, batch_size=288, learning_rate=0.000010, beta=0.936102, 
                                                  weight_decay=4.324713e-23, epoch_count=556, weight=None, input_size=no_features) #
            classifiers["NN"]["peptide"][2] = Net(dropout_parameter = 0.5, hidden_units_1=330, hidden_units_2=727, batch_size=73, learning_rate=0.000012, beta=0.856267, 
                                                  weight_decay=2.534727e-23, epoch_count=91, weight="balanced", input_size=no_features) #
            classifiers["NN"]["peptide"][3] = Net(dropout_parameter = 0.5, hidden_units_1=330, hidden_units_2=727, batch_size=73, learning_rate=0.000012, beta=0.856267, 
                                                  weight_decay=2.534727e-23, epoch_count=154, weight="balanced", input_size=no_features) #
            classifiers["NN"]["peptide"][4] = Net(dropout_parameter = 0.5, hidden_units_1=328, hidden_units_2=625, batch_size=288, learning_rate=0.000010, beta=0.936102, 
                                                  weight_decay=4.324713e-23, epoch_count=1066, weight=None, input_size=no_features) #
            classifiers["NN"]["peptide"][5] = Net(dropout_parameter = 0.5, hidden_units_1=330, hidden_units_2=727, batch_size=73, learning_rate=0.000012, beta=0.856267, 
                                                  weight_decay=2.534727e-23, epoch_count=145, weight="balanced", input_size=no_features) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["NN"]["sm"][1] = Net(dropout_parameter = 0.5, hidden_units_1=939, hidden_units_2=498, batch_size=252, learning_rate=0.000004, beta=0.931570, 
                                             weight_decay=1.918448e-24, epoch_count=457, weight=None, input_size=no_features) #
            classifiers["NN"]["sm"][2] = Net(dropout_parameter = 0.5, hidden_units_1=965, hidden_units_2=672, batch_size=99, learning_rate=0.000003, beta=0.908219, 
                                             weight_decay=2.371305e-21, epoch_count=275, weight=0.1, input_size=no_features) #
            classifiers["NN"]["sm"][3] = Net(dropout_parameter = 0.5, hidden_units_1=814, hidden_units_2=729, batch_size=145, learning_rate=0.000003, beta=0.802965, 
                                             weight_decay=4.586610e-25, epoch_count=602, weight=None, input_size=no_features) #
            classifiers["NN"]["sm"][4] = Net(dropout_parameter = 0.5, hidden_units_1=930, hidden_units_2=744, batch_size=103, learning_rate=0.000002, beta=0.886173, 
                                             weight_decay=8.883323e-12, epoch_count=380, weight=None, input_size=no_features) #
            classifiers["NN"]["sm"][5] = Net(dropout_parameter = 0.5, hidden_units_1=750, hidden_units_2=998, batch_size=136, learning_rate=0.000003, beta=0.811977, 
                                             weight_decay=3.509669e-12, epoch_count=529, weight=None, input_size=no_features) #

    
    return classifiers