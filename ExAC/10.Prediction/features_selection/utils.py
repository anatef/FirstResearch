#Basic imports
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import os
import sys

#Classifier imports
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

#ML framework imports
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale

#Downsamplers imports - prototype generation
from imblearn.under_sampling import ClusterCentroids

#Downsamplers imports - prototype selection - controlled
from imblearn.under_sampling import RandomUnderSampler, NearMiss

#Downsamplers imports - prototype selection - Cleaning techniques
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours

#Downsamplers imports - prototype selection - Cleaning techniques - Condensed nearest neighbors and derived algorithms
from imblearn.under_sampling import CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule

#Downsamplers imports - prototype selection - Cleaning techniques
from imblearn.under_sampling import InstanceHardnessThreshold

#Import utils functions
sys.path.append(os.getcwd()+"/../utils")
from CV_funcs import add_domain_name_from_table_idx, calc_CV_idx_iterative

def test_model_iterative_fixed(pred_dict, auc_dict, auprc_dict, ligand_bind_features,
        ligand_negatives_features, ligand_name, downsample_method, classifer, classifiers, splits_dict, features=[]):
    """
    Test different models in 10-folds cross-validation.
    """

    #Default: Exclude no features
    if len(features) == 0:
        features = np.ones([ligand_bind_features.shape[1],]).astype(bool)

    #Arranging the features table by the CV order, for each model
    features_pred_dfs = dict.fromkeys(classifiers.keys())

    models_req_scaling = ["SVM", "KNN"]

    for classifier in classifiers.keys():
        classifier = classifier_method
        model = classifiers[classifier]
        features_pred_dfs[classifier] = pd.DataFrame()

        #Create X and y with included features
        X = pd.concat([ligand_bind_features.iloc[:,features], ligand_negatives_features.iloc[:,features]])

        if (classifier in models_req_scaling):
            idx = X.index
            cols = X.columns
            X = pd.DataFrame(scale(X)) #Is z-scoring the data needed?
            X.index = idx #Restoring indices after scaling
            X.columns = cols

        y = [1] * ligand_bind_features.shape[0]
        y.extend([0] * ligand_negatives_features.shape[0])
        y = np.array(y)
        y_df = pd.DataFrame(y)
        y_df.index = X.index
        y_df.columns = ["label"]

        cv_idx = calc_CV_idx_iterative(X, splits_dict)

        for k in range(len(cv_idx)):
            pred_idx = k+1
            print "fold #: "+str(pred_idx)
            test_index = cv_idx[k]["test"]
            train_index = cv_idx[k]["train"]
            X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
            y_train, y_test = y_df.loc[train_index,:], y_df.loc[test_index,:]

            #Down-sample negative examples to have balanced classes
            if (downsample_method == "NoDown"):
                X_train_sampled = X_train
                y_train_sampled = y_train
            else:
                if (downsample_method == "InstanceHardnessThreshold"):
                    downsampler = downsamplers[downsample_method][classifier]
                else:
                    downsampler = downsamplers[downsample_method]

                X_train_sampled, y_train_sampled = downsampler.fit_sample(X_train, y_train["label"])

            #fit to training data
            model = classifiers[classifier]
            model.fit(X_train_sampled, y_train_sampled["label"])

            #probs = model.predict(X_test)
            #probs_list = probs
            probs_list = []
            probs = model.predict_proba(X_test)
            for l in probs:
                probs_list.append(l[1])

            pred_dict["obs"].extend(y_test["label"])
            pred_dict["prob"].extend(probs_list)
            fold_list = [pred_idx] * len(probs_list)
            pred_dict["fold"].extend(fold_list)

            model_list = [classifier] * len(probs_list)
            pred_dict["model"].extend(model_list)

            #Update auc auprc dictionaries
            auc_dict[classifier].append(roc_auc_score(y_test["label"], probs[:, 1]))
            precision, recall, _ = precision_recall_curve(y_test["label"], probs[:, 1])

            #auc_dict[classifier].append(roc_auc_score(y_test, probs))
            #precision, recall, _ = precision_recall_curve(y_test, probs)

            auprc_dict[classifier].append(auc(recall, precision))

            #Update features table
            features_pred_dfs[classifier] = features_pred_dfs[classifier].append(X_test)
            pred_idx += 1

            print "AUC = "+str(auc_dict[classifier][-1])
            print "AUPRC = "+str(auprc_dict[classifier][-1])

        avg_auc = np.sum(auc_dict[classifier])/10.0
        print "avg auc = "+str(avg_auc)

        avg_auprc = np.sum(auprc_dict[classifier])/10.0
        print "avg auprc = "+str(avg_auprc)

        print "Finished "+ligand+" "+classifier
        break

    return features_pred_dfs


def test_model_iterative_fixed(pred_dict, auc_dict, auprc_dict, ligand_bind_features,
        ligand_negatives_features, ligand_name, downsample_method, classifer, classifiers, features=[]):
    """
    Test different models in 10-folds cross-validation.
    """

    #Default: Exclude no features
    if len(features) == 0:
        features = np.ones([ligand_bind_features.shape[1],]).astype(bool)

    #Arranging the features table by the CV order, for each model
    features_pred_dfs = dict.fromkeys(classifiers.keys())

    models_req_scaling = ["SVM", "KNN"]

    features_pred_dfs[classifier] = pd.DataFrame()

    #Create X and y with included features
    X = pd.concat([ligand_bind_features.iloc[:,features], ligand_negatives_features.iloc[:,features]])

    if (classifier in models_req_scaling):
        idx = X.index
        cols = X.columns
        X = pd.DataFrame(scale(X)) #Is z-scoring the data needed?
        X.index = idx #Restoring indices after scaling
        X.columns = cols

    y = [1] * ligand_bind_features.shape[0]
    y.extend([0] * ligand_negatives_features.shape[0])
    y = np.array(y)

    binding_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    pred_idx = 1

    for train_index, test_index in binding_skf.split(X, y):
        print "fold #: "+str(pred_idx)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        #Down-sample negative examples to have balanced classes
        if (downsample_method == "NoDown"):
            X_train_sampled = X_train
            y_train_sampled = y_train
        else:
            if (downsample_method == "InstanceHardnessThreshold"):
                downsampler = downsamplers[downsample_method][classifier]
            else:
                downsampler = downsamplers[downsample_method]

            X_train_sampled, y_train_sampled = downsampler.fit_sample(X_train, y_train)

        #fit to training data
        model = classifiers[classifier]
        model.fit(X_train_sampled, y_train_sampled)


        #probs = model.predict(X_test)
        #probs_list = probs
        probs_list = []
        probs = model.predict_proba(X_test)
        for l in probs:
            probs_list.append(l[1])

        pred_dict["obs"].extend(y_test)
        pred_dict["prob"].extend(probs_list)
        fold_list = [pred_idx] * len(probs_list)
        pred_dict["fold"].extend(fold_list)

        model_list = [classifier] * len(probs_list)
        pred_dict["model"].extend(model_list)

        #Update auc auprc dictionaries
        auc_dict[classifier].append(roc_auc_score(y_test, probs[:, 1]))
        precision, recall, _ = precision_recall_curve(y_test, probs[:, 1])

        #auc_dict[classifier].append(roc_auc_score(y_test, probs))
        #precision, recall, _ = precision_recall_curve(y_test, probs)

        auprc_dict[classifier].append(auc(recall, precision))

        #Update features table
        features_pred_dfs[classifier] = features_pred_dfs[classifier].append(X_test)
        pred_idx += 1

        print "AUC = "+str(auc_dict[classifier][-1])
        print "AUPRC = "+str(auprc_dict[classifier][-1])

    avg_auc = np.sum(auc_dict[classifier])/10.0
    print "avg auc = "+str(avg_auc)

    avg_auprc = np.sum(auprc_dict[classifier])/10.0
    print "avg auprc = "+str(avg_auprc)

    print "Finished "+ligand+" "+classifier

    return features_pred_dfs


# Create groups of features and print features in each group
def create_groups(ligands_features_df):
    group_names = ["population","dna-con","sub_matrix",'dn/ds','pfam','sift/poly', "ortho-para",
               'chemical_major','chemical_substitution',"surface","whole-con", "go_terms","windowed"]

    #Group features indices
    POP_BEG = 0
    POP_END = 21
    CONGEN_BEG = 21
    CONGEN_END = 103
    SUBMAT_BEG = 103
    SUBMAT_END = 113
    DNDS_BEG = 113
    DNDS_END = 116
    PFAM_BEG = 116
    PFAM_END = 138
    SIFTPOLY_BEG = 138
    SIFTPOLY_END = 149
    ORTHOPARA_BEG = 149
    ORTHOPARA_END = 166
    MAJOR_BEG = 166
    MAJOR_END = 232
    SUB_BEG = 232
    SUB_END = 355
    SURFACE_BEG = 355
    SURFACE_END = 386
    WHOLECON_BEG = 386
    WHOLECON_END = 390
    GO_BEG = 390
    GO_END = 397
    WINDOW_BEG = 397
    WINDOW_END = 637


    group_indices = [range(POP_BEG,POP_END),range(CONGEN_BEG,CONGEN_END),range(SUBMAT_BEG,SUBMAT_END),range(DNDS_BEG,DNDS_END),
                     range(PFAM_BEG,PFAM_END),range(SIFTPOLY_BEG,SIFTPOLY_END), range(ORTHOPARA_BEG,ORTHOPARA_END),range(MAJOR_BEG,MAJOR_END),
                     range(SUB_BEG,SUB_END),range(SURFACE_BEG,SURFACE_END),range(WHOLECON_BEG,WHOLECON_END), range(GO_BEG,GO_END),
                    range(WINDOW_BEG,WINDOW_END)]

    features_groups = dict(zip(group_names,group_indices))

    features_list = np.array([c.encode('ascii','ignore') for c in ligands_features_df['dna'].columns])
    for g in group_names:
        group_list = features_list[features_groups[g]]
        print("------- "+g+" --------")
        print("")
        for i in group_list:
            print(i)
        print("")

    return features_groups
