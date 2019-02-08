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
            classifiers["XGB"]["dna"][1] = XGBClassifier(n_estimators=1668, n_jobs=-1, random_state=0, max_depth=36, min_child_weight=1.843626, colsample_bytree= 0.865745, 
                                                       scale_pos_weight=1, gamma=0.001956, learning_rate=0.085666) #
            classifiers["XGB"]["dna"][2] = XGBClassifier(n_estimators=1688, n_jobs=-1, random_state=0, max_depth=26, min_child_weight=0.355180, colsample_bytree=0.315347, 
                                                       scale_pos_weight=scale_weight, gamma=0.001150, learning_rate=0.082067) #
            classifiers["XGB"]["dna"][3] = XGBClassifier(n_estimators=2225, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962148, colsample_bytree=0.313751, 
                                                       scale_pos_weight=1, gamma=0.001920, learning_rate=0.016505) #
            classifiers["XGB"]["dna"][4] = XGBClassifier(n_estimators=2471, n_jobs=-1, random_state=0, max_depth=33, min_child_weight=2.609242, colsample_bytree=0.560996, 
                                                       scale_pos_weight=1, gamma=0.006218, learning_rate=0.051265) #
            classifiers["XGB"]["dna"][5] = XGBClassifier(n_estimators=1573, n_jobs=-1, random_state=0, max_depth=36, min_child_weight=1.843626, colsample_bytree=0.865745, 
                                                       scale_pos_weight=1, gamma=0.001956, learning_rate=0.085666) #

        #==rna==#
        elif (ligand == "rna"):  
            ## for all the features
            classifiers["XGB"]["rna"][1] = XGBClassifier(n_estimators=65, n_jobs=-1, random_state=0, max_depth=84, min_child_weight=3.229471, colsample_bytree=0.578190, 
                                                       scale_pos_weight=1, gamma=0.473499, learning_rate=0.235950) #
            classifiers["XGB"]["rna"][2] = XGBClassifier(n_estimators=1193, n_jobs=-1, random_state=0, max_depth=20, min_child_weight=3.394398, colsample_bytree=0.790474, 
                                                       scale_pos_weight=0.1, gamma=0.055726, learning_rate=0.007600) #
            classifiers["XGB"]["rna"][3] = XGBClassifier(n_estimators=62, n_jobs=-1, random_state=0, max_depth=3, min_child_weight=2.541577, colsample_bytree=0.375136, 
                                                      scale_pos_weight=scale_weight, gamma=0.217347, learning_rate=0.106480) #
            classifiers["XGB"]["rna"][4] = XGBClassifier(n_estimators=2905, n_jobs=-1, random_state=0, max_depth=4, min_child_weight=0.603285, colsample_bytree=0.985525, 
                                                       scale_pos_weight=scale_weight, gamma=0.016533, learning_rate=0.009823) #
            classifiers["XGB"]["rna"][5] = XGBClassifier(n_estimators=433, n_jobs=-1, random_state=0, max_depth=3, min_child_weight=2.54157, colsample_bytree=0.375136, 
                                                       scale_pos_weight=scale_weight, gamma=0.217347, learning_rate=0.106480) #
            
        #==ion==#
        elif (ligand == "ion"):
            classifiers["XGB"]["ion"][1] = XGBClassifier(n_estimators=485, n_jobs=-1, random_state=0, max_depth=33, min_child_weight=2.609242, colsample_bytree=0.560996, 
                                                       scale_pos_weight=1, gamma=0.006218, learning_rate=0.051265) #
            classifiers["XGB"]["ion"][2] = XGBClassifier(n_estimators=1793, n_jobs=-1, random_state=0, max_depth=40, min_child_weight=0.290146, colsample_bytree=0.575812, 
                                                      scale_pos_weight=1, gamma=0.008618, learning_rate=0.027365) #
            classifiers["XGB"]["ion"][3] = XGBClassifier(n_estimators=639, n_jobs=-1, random_state=0, max_depth=54, min_child_weight=2.845504, colsample_bytree=0.555387, 
                                                       scale_pos_weight=1, gamma=0.001613, learning_rate=0.027606) #
            classifiers["XGB"]["ion"][4] = XGBClassifier(n_estimators=1207, n_jobs=-1, random_state=0, max_depth=33, min_child_weight=2.609242, colsample_bytree=0.560996, 
                                                       scale_pos_weight=1, gamma=0.006218, learning_rate=0.051265) #
            classifiers["XGB"]["ion"][5] = XGBClassifier(n_estimators=628, n_jobs=-1, random_state=0, max_depth=88, min_child_weight=0.462981, colsample_bytree=0.413088, 
                                                       scale_pos_weight=scale_weight, gamma=0.176914, learning_rate=0.036385) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["XGB"]["peptide"][1] = XGBClassifier(n_estimators=2921, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962148, colsample_bytree=0.313751, 
                                                       scale_pos_weight=1, gamma=0.001920, learning_rate=0.016505) #
            #classifiers["XGB"]["peptide"][1] = XGBClassifier(n_estimators=3993, n_jobs=-1, random_state=0, max_depth=83, min_child_weight=0.644631, colsample_bytree=0.486571, 
             #                                          scale_pos_weight=scale_weight, gamma=0.012335, learning_rate=0.009901) #
            classifiers["XGB"]["peptide"][2] = XGBClassifier(n_estimators=273, n_jobs=-1, random_state=0, max_depth=94, min_child_weight=0.313565, colsample_bytree=0.568024, 
                                                       scale_pos_weight=0.1, gamma=0.005971, learning_rate=0.093677) #  
            #classifiers["XGB"]["peptide"][2] = XGBClassifier(n_estimators=480, n_jobs=-1, random_state=0, max_depth=88, min_child_weight=0.462981, colsample_bytree=0.413088, 
             #                                          scale_pos_weight=1, gamma=0.176914, learning_rate=0.036385) #  
            classifiers["XGB"]["peptide"][3] = XGBClassifier(n_estimators=1378, n_jobs=-1, random_state=0, max_depth=67, min_child_weight=4.962148, colsample_bytree=0.313751, 
                                                       scale_pos_weight=1, gamma=0.001920, learning_rate=0.016505) #
            classifiers["XGB"]["peptide"][4] = XGBClassifier(n_estimators=748, n_jobs=-1, random_state=0, max_depth=44, min_child_weight=3.088290, colsample_bytree=0.656874, 
                                                       scale_pos_weight=0.1, gamma=0.366303, learning_rate=0.040126) #
            classifiers["XGB"]["peptide"][5] = XGBClassifier(n_estimators=1234, n_jobs=-1, random_state=0, max_depth=24, min_child_weight=1.690038, colsample_bytree=0.756064, 
                                                       scale_pos_weight=scale_weight, gamma=0.008946, learning_rate=0.052992) #
 
        #==sm==#
        elif (ligand == "sm"):
            classifiers["XGB"]["sm"][1] = XGBClassifier(n_estimators=2161, n_jobs=-1, random_state=0, max_depth=27, min_child_weight=0.467297, colsample_bytree=0.569729, 
                                                       scale_pos_weight=0.1, gamma=0.026282, learning_rate=0.010732) #
            classifiers["XGB"]["sm"][2] = XGBClassifier(n_estimators=1777, n_jobs=-1, random_state=0, max_depth=58, min_child_weight=0.735069, colsample_bytree=0.369903, 
                                                       scale_pos_weight=0.1, gamma=0.002445, learning_rate=0.010623) #
            classifiers["XGB"]["sm"][3] = XGBClassifier(n_estimators=575, n_jobs=-1, random_state=0, max_depth=94, min_child_weight=0.313565, colsample_bytree=0.568024, 
                                                       scale_pos_weight=0.1, gamma=0.005971, learning_rate=0.093677) #
            classifiers["XGB"]["sm"][4] = XGBClassifier(n_estimators=2263, n_jobs=-1, random_state=0, max_depth=27, min_child_weight=0.467297, colsample_bytree=0.569729, 
                                                       scale_pos_weight=0.1, gamma=0.026282, learning_rate=0.010732) #
            classifiers["XGB"]["sm"][5] = XGBClassifier(n_estimators=499, n_jobs=-1, random_state=0, max_depth=13, min_child_weight=2.847868, colsample_bytree=0.589082, 
                                                       scale_pos_weight=1, gamma=0.814161, learning_rate=0.024094) #
    
    ###RF###
    elif (classifier_method == "RF"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["RF"]["dna"][1] = RandomForestClassifier(n_estimators=814, max_depth=50, min_samples_leaf=26, min_samples_split=5, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][2] = RandomForestClassifier(n_estimators=1214, max_depth=80, min_samples_leaf=16, min_samples_split=22, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][3] = RandomForestClassifier(n_estimators=983, max_depth=23, min_samples_leaf=10, min_samples_split=2, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][4] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["dna"][5] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["RF"]["rna"][1] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][2] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][3] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][4] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["rna"][5] = RandomForestClassifier(n_estimators=1415, max_depth=26, min_samples_leaf=4, min_samples_split=20, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["RF"]["ion"][1] = RandomForestClassifier(n_estimators=1415, max_depth=26, min_samples_leaf=4, min_samples_split=20, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][2] = RandomForestClassifier(n_estimators=983, max_depth=23, min_samples_leaf=10, min_samples_split=2, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][3] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) # 
            classifiers["RF"]["ion"][4] = RandomForestClassifier(n_estimators=983, max_depth=23, min_samples_leaf=10, min_samples_split=2, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["ion"][5] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["RF"]["peptide"][1] = RandomForestClassifier(n_estimators=765, max_depth=71, min_samples_leaf=16, min_samples_split=49, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][2] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][3] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][4] = RandomForestClassifier(n_estimators=765, max_depth=71, min_samples_leaf=16, min_samples_split=49, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["peptide"][5] = RandomForestClassifier(n_estimators=959, max_depth=31, min_samples_leaf=4, min_samples_split=37, class_weight="balanced", n_jobs=-1, random_state=0) #
 
        #==sm==#
        elif (ligand == "sm"):
            classifiers["RF"]["sm"][1] = RandomForestClassifier(n_estimators=1261, max_depth=85, min_samples_leaf=8, min_samples_split=10, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][2] = RandomForestClassifier(n_estimators=1457, max_depth=63, min_samples_leaf=22, min_samples_split=35, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][3] = RandomForestClassifier(n_estimators=518, max_depth=37, min_samples_leaf=31, min_samples_split=31, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][4] = RandomForestClassifier(n_estimators=765, max_depth=71, min_samples_leaf=16, min_samples_split=49, class_weight="balanced", n_jobs=-1, random_state=0) #
            classifiers["RF"]["sm"][5] = RandomForestClassifier(n_estimators=1214, max_depth=80, min_samples_leaf=16, min_samples_split=22, class_weight="balanced", n_jobs=-1, random_state=0) #
    
    ###SVM###
    elif (classifier_method == "SVM"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["SVM"]["dna"][1] = SVC(kernel='rbf', class_weight="balanced", C=0.914200, gamma=0.000742, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][2] = SVC(kernel='rbf', class_weight=None, C=0.135235, gamma=0.001754, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][3] = SVC(kernel='rbf', class_weight="balanced", C=0.914200, gamma=0.000742, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][4] = SVC(kernel='rbf', class_weight=None, C=0.007809, gamma=0.001233, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["dna"][5] = SVC(kernel='rbf', class_weight=None, C=0.135235, gamma=0.001754, probability=True, random_state=0, cache_size=200) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["SVM"]["rna"][1] = SVC(kernel='rbf', class_weight="balanced", C=1.573532, gamma=0.000780, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][2] = SVC(kernel='rbf', class_weight="balanced", C=1.573532, gamma=0.000780, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][3] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=7.634611, gamma=0.001547, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][4] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=2.402881, gamma=0.000863, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["rna"][5] = SVC(kernel='rbf', class_weight="balanced", C=1.573532, gamma=0.000780, probability=True, random_state=0, cache_size=200) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["SVM"]["ion"][1] = SVC(kernel='rbf', class_weight=None, C=0.817748, gamma=0.001969, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][2] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=5.783440, gamma=0.000470, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][3] = SVC(kernel='rbf', class_weight={0: 10, 1: 1}, C=7.634611, gamma=0.001547, probability=True, random_state=0, cache_size=200) # 
            classifiers["SVM"]["ion"][4] = SVC(kernel='rbf', class_weight=None, C=2.398520, gamma=0.000327, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["ion"][5] = SVC(kernel='rbf', class_weight=None, C=0.817748, gamma=0.001969, probability=True, random_state=0, cache_size=200) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["SVM"]["peptide"][1] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C= 2.402881, gamma=0.000863, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][2] = SVC(kernel='rbf', class_weight=None, C=10.657846, gamma=0.000194, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][3] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=7.634611, gamma=0.001547, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][4] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=7.634611, gamma=0.001547, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["peptide"][5] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=7.634611, gamma=0.001547, probability=True, random_state=0, cache_size=200) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["SVM"]["sm"][1] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=0.909128, gamma=0.003861, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][2] = SVC(kernel='rbf', class_weight=None, C=3.039016, gamma=0.001267, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][3] = SVC(kernel='rbf', class_weight=None, C=3.039016, gamma=0.001267, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][4] = SVC(kernel='rbf', class_weight={0: 10, 1:1}, C=0.909128, gamma=0.003861, probability=True, random_state=0, cache_size=200) #
            classifiers["SVM"]["sm"][5] = SVC(kernel='rbf', class_weight=None, C=3.039016, gamma=0.001267, probability=True, random_state=0, cache_size=200) # 

    ###Logistic###
    elif (classifier_method == "Logistic"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["Logistic"]["dna"][1] = LogisticRegression(C=0.004418, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][2] = LogisticRegression(C=0.004418, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][3] = LogisticRegression(C=0.004418, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][4] = LogisticRegression(C=0.004418, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["dna"][5] = LogisticRegression(C=0.018269, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["Logistic"]["rna"][1] = LogisticRegression(C=0.056808, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["rna"][2] = LogisticRegression(C=0.056808, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["rna"][3] = LogisticRegression(C=0.018269, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["rna"][4] = LogisticRegression(C=0.028500, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["rna"][5] = LogisticRegression(C=0.055992, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
        #==ion==#
        elif (ligand == "ion"):
            classifiers["Logistic"]["ion"][1] = LogisticRegression(C=0.018704, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["ion"][2] = LogisticRegression(C=0.018704, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["ion"][3] = LogisticRegression(C=0.018704, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["ion"][4] = LogisticRegression(C=0.006847, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["ion"][5] = LogisticRegression(C=0.013528, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["Logistic"]["peptide"][1] = LogisticRegression(C=0.018269, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["peptide"][2] = LogisticRegression(C=0.056808, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["peptide"][3] = LogisticRegression(C=0.009500, random_state=0, n_jobs=-1, class_weight={0: 10, 1: 1}) #
            classifiers["Logistic"]["peptide"][4] = LogisticRegression(C=0.055992, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["peptide"][5] = LogisticRegression(C=0.055992, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["Logistic"]["sm"][1] = LogisticRegression(C=0.002423, random_state=0, n_jobs=-1, class_weight=None) #
            classifiers["Logistic"]["sm"][2] = LogisticRegression(C=0.003025, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["sm"][3] = LogisticRegression(C=0.005405, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["sm"][4] = LogisticRegression(C=0.005405, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
            classifiers["Logistic"]["sm"][5] = LogisticRegression(C=0.003025, random_state=0, n_jobs=-1, class_weight={0: 10, 1:1}) #
   
    ###NN###
    elif (classifier_method == "NN"):
        #==dna==#
        if (ligand == "dna"):
            classifiers["NN"]["dna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=863, hidden_units_2=630, batch_size=90, learning_rate=0.000057, beta=0.966985,
                                              weight_decay=1.268751e-16, epoch_count=7, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=976, hidden_units_2=559, batch_size=238, learning_rate=0.000046, beta=0.834510,
                                              weight_decay=1.174615e-11, epoch_count=14, weight=None, input_size=no_features) #
            classifiers["NN"]["dna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=559, hidden_units_2=359, batch_size=147, learning_rate=0.000013, beta=0.963010,
                                              weight_decay=7.679182e-09, epoch_count=37, weight=None, input_size=no_features) #
            classifiers["NN"]["dna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=541, hidden_units_2=732, batch_size=292, learning_rate=0.000052, beta=0.801498,
                                              weight_decay=1.609669e-11, epoch_count=16, weight=0.1, input_size=no_features) #
            classifiers["NN"]["dna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=877, hidden_units_2=887, batch_size=60, learning_rate=0.000006, beta=0.907928,
                                              weight_decay=9.662474e-11, epoch_count=37, weight="balanced", input_size=no_features) #
        #==rna==#
        elif (ligand == "rna"):
            classifiers["NN"]["rna"][1] = Net(dropout_parameter = 0.5, hidden_units_1=641, hidden_units_2=512, batch_size=30, learning_rate=0.000005, beta=0.809251, 
                                              weight_decay=1.763482e-08, epoch_count=82, weight=None, input_size=no_features) # 
            classifiers["NN"]["rna"][2] = Net(dropout_parameter = 0.5, hidden_units_1=565, hidden_units_2=455, batch_size=50, learning_rate=0.000011, beta=0.900717, 
                                              weight_decay=5.451840e-18, epoch_count=41, weight=None, input_size=no_features) #  
            classifiers["NN"]["rna"][3] = Net(dropout_parameter = 0.5, hidden_units_1=939, hidden_units_2=723, batch_size=104, learning_rate=0.000022, beta=0.936203, 
                                              weight_decay=4.378871e-23, epoch_count=60, weight="balanced", input_size=no_features) #
            classifiers["NN"]["rna"][4] = Net(dropout_parameter = 0.5, hidden_units_1=877, hidden_units_2=887, batch_size=69, learning_rate=0.000006, beta=0.907928, 
                                              weight_decay=3.783541e-15, epoch_count=57, weight="balanced", input_size=no_features) #
            classifiers["NN"]["rna"][5] = Net(dropout_parameter = 0.5, hidden_units_1=557, hidden_units_2=757, batch_size=33, learning_rate=0.000027, beta=0.960465, 
                                              weight_decay=9.179193e-16, epoch_count=17, weight="balanced", input_size=no_features)  #

        #==ion==#
        elif (ligand == "ion"):
            classifiers["NN"]["ion"][1] = Net(dropout_parameter = 0.5, hidden_units_1=914, hidden_units_2=594, batch_size=295, learning_rate=0.000172, beta=0.944137,
                                              weight_decay=5.590679e-15, epoch_count=19, weight=None, input_size=no_features) #
            classifiers["NN"]["ion"][2] = Net(dropout_parameter = 0.5, hidden_units_1=817, hidden_units_2=487, batch_size=149, learning_rate=0.000106, beta=0.890890,
                                              weight_decay=4.330996e-06, epoch_count=52, weight="balanced", input_size=no_features) #   
            classifiers["NN"]["ion"][3] = Net(dropout_parameter = 0.5, hidden_units_1=250, hidden_units_2=935, batch_size=566, learning_rate=0.000161, beta=0.912266,
                                              weight_decay=2.507831e-14, epoch_count=23, weight=0.1, input_size=no_features) #
            classifiers["NN"]["ion"][4] = Net(dropout_parameter = 0.5, hidden_units_1=935, hidden_units_2=566, batch_size=196, learning_rate=0.000161, beta=0.912266,
                                              weight_decay=2.507831e-14, epoch_count=30, weight=0.1, input_size=no_features) #
            classifiers["NN"]["ion"][5] = Net(dropout_parameter = 0.5, hidden_units_1=935, hidden_units_2=566, batch_size=196, learning_rate=0.000161, beta=0.912266,
                                              weight_decay=2.507831e-14, epoch_count=6, weight=0.1, input_size=no_features) #

        #==peptide==#
        elif (ligand == "peptide"):
            classifiers["NN"]["peptide"][1] = Net(dropout_parameter = 0.5, hidden_units_1=871, hidden_units_2=943, batch_size=130, learning_rate=0.000123, beta=0.946507, 
                                                  weight_decay=4.008984e-14, epoch_count=75, weight=None, input_size=no_features) #
            classifiers["NN"]["peptide"][2] = Net(dropout_parameter = 0.5, hidden_units_1=871, hidden_units_2=943, batch_size=130, learning_rate=0.000123, beta=0.946507, 
                                                  weight_decay=4.008984e-14, epoch_count=14, weight=None, input_size=no_features) #  
            classifiers["NN"]["peptide"][3] = Net(dropout_parameter = 0.5, hidden_units_1=877, hidden_units_2=887, batch_size=69, learning_rate=0.000058, beta=0.907928, 
                                                  weight_decay=3.783541e-15, epoch_count=93, weight="balanced", input_size=no_features) #
            classifiers["NN"]["peptide"][4] = Net(dropout_parameter = 0.5, hidden_units_1=915, hidden_units_2=631, batch_size=43, learning_rate=0.000035, beta=0.929177, 
                                                  weight_decay=5.090415e-08, epoch_count=100, weight="balanced", input_size=no_features) #
            classifiers["NN"]["peptide"][5] = Net(dropout_parameter = 0.5, hidden_units_1=401, hidden_units_2=932, batch_size=733, learning_rate=0.000016, beta=0.810586, 
                                                  weight_decay=1.557713e-15, epoch_count=134, weight=0.1, input_size=no_features) #
        #==sm==#
        elif (ligand == "sm"):
            classifiers["NN"]["sm"][1] = Net(dropout_parameter = 0.5, hidden_units_1=605, hidden_units_2=551, batch_size=113, learning_rate=0.000090, beta=0.804689, 
                                             weight_decay=3.909071e-22, epoch_count=7, weight=None, input_size=no_features) #38
            classifiers["NN"]["sm"][2] = Net(dropout_parameter = 0.5, hidden_units_1=554, hidden_units_2=952, batch_size=56, learning_rate=0.000028, beta=0.880998, 
                                             weight_decay=7.399274e-24, epoch_count=13, weight=0.1, input_size=no_features) #
            classifiers["NN"]["sm"][3] = Net(dropout_parameter = 0.5, hidden_units_1=821, hidden_units_2=878, batch_size=169, learning_rate=0.000091, beta=0.954622, 
                                             weight_decay=8.886669e-20, epoch_count=5, weight=None, input_size=no_features) #
            classifiers["NN"]["sm"][4] = Net(dropout_parameter = 0.5, hidden_units_1=226, hidden_units_2=598, batch_size=198, learning_rate=0.000053, beta=0.848667, 
                                             weight_decay=8.530848e-24, epoch_count=24, weight=0.1, input_size=no_features) #
            classifiers["NN"]["sm"][5] = Net(dropout_parameter = 0.5, hidden_units_1=559, hidden_units_2=359, batch_size=147, learning_rate=0.000013, beta=0.963010, 
                                             weight_decay=7.679182e-09, epoch_count=65, weight=None, input_size=no_features) #
  
    return classifiers