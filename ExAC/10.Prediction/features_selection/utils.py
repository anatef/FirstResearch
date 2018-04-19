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
    group_names = ['population','conservation_genome','sub_matrix','selection','pfam','sub_predictor',
                   'chemical_major_allele','chemical_substitution','spider','whole_domain_conservation','go_terms']
    group_indices = [range(0,21),range(21,103),range(103,113),range(113,116)+range(149,166),range(116,138),range(138,149),
                     range(166,232),range(232,355),range(355,386),range(386,390),range(390,397)]

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
