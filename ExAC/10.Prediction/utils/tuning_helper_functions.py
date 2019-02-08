#Imports
import pandas as pd
import numpy as np
import pickle
from os import getcwd

#Classifier imports
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Neural Net imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from sklearn.model_selection import RepeatedStratifiedKFold

#General learning framework imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve
from sklearn.utils import shuffle

from CV_funcs import calc_CV_idx_iterative
from NN_classes import Net, Net_tune, curr_device
#from generate_models_dict_global_auprc import Net

#CV splits dictionary
pfam_version = "31"
datafile_date = "08.06.18"
prec_th_str = "dna0.5_rna0.5_ion0.75"
folds_num = 5
curr_dir = getcwd()
exac_dir = curr_dir[:curr_dir.find("ExAC")]
with open(exac_dir+"ExAC/10.Prediction/CV_splits/pfam-v"+pfam_version+"/domain_"+str(folds_num)+"_folds_combined_"+prec_th_str+"_prec_dict.pik", 'rb') as handle:
    splits_dict = pickle.load(handle)


models_req_scaling = ["SVM", "KNN", "Logistic", "NN"]
#====================================================================================================================#

def generate_model(classifier_method, hyperparameters, no_pos=1, no_neg=1, features_num=750, test_mode=False, rseed=0):
    
    xgb_trees_limit = 5000
    
    if (classifier_method == "XGB"):
        if (hyperparameters["scale_pos_weight"] == "balanced"):
            scale_weight = no_neg/float(no_pos)
        else:
            scale_weight = hyperparameters["scale_pos_weight"]
        if (test_mode):
            trees_num = hyperparameters["mean_epoch_count"]
        else:
            trees_num = xgb_trees_limit
        model = XGBClassifier(n_estimators=trees_num, n_jobs=-1, random_state=rseed, max_depth=hyperparameters["max_depth"], 
                              min_child_weight=hyperparameters["min_child_weight"], colsample_bytree=hyperparameters["colsample_bytree"], 
                              gamma=hyperparameters["gamma"], learning_rate=hyperparameters["learning_rate"], scale_pos_weight=scale_weight)
        
    elif (classifier_method == "RF"):
        model = RandomForestClassifier(n_estimators=hyperparameters["n_estimators"], n_jobs=-1, random_state=rseed,
                                      max_depth=hyperparameters["max_depth"], min_samples_leaf=hyperparameters["min_samples_leaf"],
                                      min_samples_split=hyperparameters["min_samples_split"], class_weight=hyperparameters["class_weight"])
        
    elif(classifier_method == "Logistic"):
        model = LogisticRegression(C=hyperparameters["C"], random_state=rseed, n_jobs=-1, class_weight=hyperparameters["class_weight"])
        
    elif (classifier_method == "KNN"):
        model = KNeighborsClassifier(n_neighbors=hyperparameters["n_neighbors"], n_jobs=-1, weights=hyperparameters["weights"])
        
    elif (classifier_method == "ADA"):
        model = AdaBoostClassifier(n_estimators=hyperparameters["n_estimators"], random_state=rseed, learning_rate=hyperparameters["learning_rate"])
        
    elif (classifier_method == "SVM"):
        model = SVC(C=hyperparameters["C"], gamma = hyperparameters["gamma"], kernel=hyperparameters["kernel"], probability=True, random_state=rseed, cache_size=400,
                    class_weight = hyperparameters["class_weight"])
        
    elif (classifier_method =="NN"):
        if (test_mode):
            model = Net(dropout_parameter = 0.5, hidden_units_1 = hyperparameters["hidden_units_1"], 
                 hidden_units_2 = hyperparameters["hidden_units_2"], batch_size = hyperparameters["batch_size"], 
                 learning_rate = hyperparameters["learning_rate"], beta = hyperparameters["beta"], 
                 weight_decay = hyperparameters["weight_decay"], epoch_count=hyperparameters["mean_epoch_count"],
                 weight = hyperparameters["weight"], input_size=features_num, rseed=rseed)
        else:
            model = Net_tune(dropout_parameter = 0.5, hidden_units_1 = hyperparameters["hidden_units_1"], 
                 hidden_units_2 = hyperparameters["hidden_units_2"], batch_size = hyperparameters["batch_size"], 
                 learning_rate = hyperparameters["learning_rate"], beta = hyperparameters["beta"], 
                 weight_decay = hyperparameters["weight_decay"], input_size=features_num)
        model = model.to(device=curr_device)
    return model
#====================================================================================================================#


def compute_per_domain_auc(y_test, pred_probs, pred_idx, classifier):
    """
    Compute the average per_domain auc and auprc for the test set
    """
    
    y_test_copy = y_test.copy(deep=True)
    y_test_copy["pred_probs"] = pred_probs
    
    domain_auc_list = []
    domain_auprc_list = []
    domain_auprc_ratio_list = []
    domain_name_list = []
    domain_pos_num_list = []
    domain_neg_num_list = []
    
    idx = y_test.index
    y_test_copy["domain_name"] = [x[:x.rfind("_")] for x in idx]
    domains_list = y_test_copy["domain_name"].unique().tolist()
        
    for domain_name in domains_list:
        
        #Get only the domain positions
        domain_df = y_test_copy[y_test_copy["domain_name"] == domain_name]

        #Find the binding and non-binding positions of this domain 
        bind_list = domain_df[domain_df["label"] == 1].index
        bind_idx = [int(x[len(domain_name)+1:]) for x in bind_list]
        bind_num = len(bind_idx)
        non_bind_list = domain_df[domain_df["label"] == 0].index
        non_bind_idx = [int(x[len(domain_name)+1:]) for x in non_bind_list]
        non_bind_num = len(non_bind_idx)
        if (bind_num == 0 or non_bind_num == 0):
            #No positions of one of the classes "binding/non-binding" - skipping"
            continue
            
        #Add number of positives and number of negatives
        domain_pos_num_list.append(bind_num)
        domain_neg_num_list.append(non_bind_num)
        #Compute domain AUC
        domain_auc = roc_auc_score(domain_df["label"], domain_df["pred_probs"])
        domain_auc_list.append(domain_auc)
        #Compute domain AUPRC
        precision, recall, thresholds = precision_recall_curve(domain_df["label"], domain_df["pred_probs"])
        domain_auprc = auc(recall, precision)
        domain_auprc_list.append(domain_auprc)
        #Add positives fraction to list
        pos_frac_ratio = bind_num/float(domain_df.shape[0])
        #Add ratio of AUPRC and positives fraction to list
        domain_auprc_ratio_list.append(domain_auprc/float(pos_frac_ratio))
        #Add domain name for AUC/AUPRC/Ratio tables
        domain_name_list.append(domain_name)
        
    #Compute the means for the lists 
    domain_auc_mean = np.mean(domain_auc_list)
    domain_auprc_mean = np.mean(domain_auprc_list)
    domain_auprc_ratio_mean = np.mean(domain_auprc_ratio_list)
    
    return (domain_auc_mean, domain_auprc_mean, domain_auprc_ratio_mean)
#====================================================================================================================#
#Weight Vector: https://towardsdatascience.com/dealing-with-imbalanced-classes-in-machine-learning-d43d6fa19d2 (look at section on "Cost-sensitive Learning")
#Implementing Early Stopping for XGBoost: https://cambridgespark.com/content/tutorials/hyperparameter-tuning-in-xgboost/index.html

def test_model_on_validation(hyperparameters, hyperparameters_dict,ligand_bind_features, ligand_negatives_features, ligand_name, classifier_method, fold,
                             trial_idx, features=[], xgb_early_stopping_rounds=750, xgb_increase_rounds_limit=2000, final_model=False):
    
    """
    Test different models in k-folds cross-validation.
    """
    
    #Default: Exclude no features
    if len(features) == 0:
        features = np.ones([ligand_bind_features.shape[1],]).astype(bool)
    
    classifier = classifier_method

    #Create X and y with included features
    X = pd.concat([ligand_bind_features.iloc[:,features], ligand_negatives_features.iloc[:,features]])
    y = [1] * ligand_bind_features.shape[0]
    y.extend([0] * ligand_negatives_features.shape[0])
    y = np.array(y)
    y_df = pd.DataFrame(y)
    y_df.index = X.index
    y_df.columns = ["label"]
    
    #Get the fold indices
    cv_idx = calc_CV_idx_iterative(X, splits_dict)
    k = (int(fold)-1)
    
    pred_idx = k+1
    print "fold #: "+str(pred_idx)
    #test_index = cv_idx[k]["test"]
    full_train_index = cv_idx[k]["train"]
        
    # phase 1: testing on validation set, hyperparameter tuning
    
    if (final_model):
        num_folds_heldout = 0
    else:
        num_folds_heldout = 1
    
    trial_auprc_results = np.zeros(folds_num-num_folds_heldout)
    trial_auc_results = np.zeros(folds_num-num_folds_heldout)
    trial_domain_auc_results = np.zeros(folds_num-num_folds_heldout)
    trial_domain_auprc_results = np.zeros(folds_num-num_folds_heldout)
    trial_domain_auprc_ratio_results = np.zeros(folds_num-num_folds_heldout)
    
    epoch_counts = np.zeros(folds_num-num_folds_heldout, dtype = "int")
    
    for i in range(folds_num-num_folds_heldout):
        #Getting the correct samples indices for training and validation
        if (final_model):
            train_index = cv_idx[i]["train"]
            valid_index = cv_idx[i]["test"]
            print "len(train_index)= "+str(len(train_index))
            print "len(valid_index)= "+str(len(valid_index))
        else:
            valid_k = (k + 1 + i) % folds_num
            valid_index = cv_idx[valid_k]["test"]
            train_index = [index for index in full_train_index if index not in valid_index]
        
        #Splitting the dataset according to indices    
        X_train, X_valid = X.loc[train_index,:], X.loc[valid_index,:]
        y_train, y_valid = y_df.loc[train_index,:], y_df.loc[valid_index,:]

        if (classifier in models_req_scaling):
            cols = X_train.columns

            # phase 1 scaling with just training data
            scaler_1 = StandardScaler() 
            scaler_1.fit(X_train) 
            X_train = pd.DataFrame(scaler_1.transform(X_train))
            # apply same transformation to validation data
            X_valid = pd.DataFrame(scaler_1.transform(X_valid))

            #Restoring indices after scaling
            X_train.index = train_index 
            X_valid.index = valid_index

            #Restoring features names
            X_train.columns = cols
            X_valid.columns = cols

        #No down-sampling
        X_train_sampled = X_train
        y_train_sampled = y_train
        
        #Shuffle training data rows
        np.random.seed(0)
        idx_perm_train = np.random.permutation(X_train_sampled.index)
        X_train_sampled_perm = X_train_sampled.reindex(idx_perm_train)
        y_train_sampled_perm = y_train_sampled.reindex(idx_perm_train)
        
        #Shuffle validation data rows
        idx_perm_valid = np.random.permutation(X_valid.index)
        X_valid_perm = X_valid.reindex(idx_perm_valid)
        y_valid_perm = y_valid.reindex(idx_perm_valid)
        
        #pos and neg numbers in the training
        no_pos = np.count_nonzero(y_train_sampled["label"] == 1)
        no_neg = np.count_nonzero(y_train_sampled["label"] == 0)  
        
        #fit to training data
        if (classifier == "NN"):
            if hyperparameters["weight"] == "balanced":              
                #weight vector
                neg_weight = float(no_pos) / float(no_neg + no_pos) 
                pos_weight = 1 - neg_weight
            elif hyperparameters["weight"] == "0.1":
                neg_weight = 10
                pos_weight = 1
            elif hyperparameters["weight"] == "None":
                neg_weight = 1
                pos_weight = 1
            
            weight = torch.Tensor([neg_weight, pos_weight]).to(device=curr_device)
            model = generate_model(classifier_method, hyperparameters, features_num=X_train_sampled.shape[1])
            auprc_score,epoch_count = model.fit(X_train_sampled, y_train_sampled["label"],X_valid, y_valid["label"], weight)
            probs_list = probs_list = model.predict_proba(X_valid_perm)
            auc_score = roc_auc_score(y_valid_perm, probs_list)

        elif (classifier == "XGB"):
            num_early_stopping_rounds = xgb_early_stopping_rounds
            model = generate_model(classifier_method, hyperparameters, no_pos = no_pos, no_neg = no_neg)
            model.fit(X_train_sampled_perm, y_train_sampled_perm["label"], eval_set = [(X_valid_perm,y_valid_perm["label"])], eval_metric = "map", 
                      verbose=False, early_stopping_rounds = num_early_stopping_rounds)
            
            #Handeling cases where more iterations are needed to see improvment
            while (model.best_iteration == 0 and num_early_stopping_rounds <= xgb_increase_rounds_limit):
                num_early_stopping_rounds = num_early_stopping_rounds+250
                print "Increasing num_early_stopping_rounds to "+str(num_early_stopping_rounds)
                model.fit(X_train_sampled_perm, y_train_sampled_perm["label"], eval_set = [(X_valid_perm,y_valid_perm["label"])], eval_metric = "map", 
                      verbose=False, early_stopping_rounds = num_early_stopping_rounds)

            probs_list = []
            probs = model.predict_proba(X_valid_perm, ntree_limit=model.best_ntree_limit)
            for l in probs:
                probs_list.append(l[1])
            precision, recall, _ = precision_recall_curve(y_valid_perm, probs_list)
            auprc_score = auc(recall, precision)
            auc_score = roc_auc_score(y_valid_perm, probs_list)
            print "model.best_iteration = "+str(model.best_iteration)
            epoch_count = model.best_ntree_limit

        else:            
            model = generate_model(classifier_method, hyperparameters)
            model.fit(X_train_sampled_perm, y_train_sampled_perm["label"])
            probs_list = []
            probs = model.predict_proba(X_valid_perm)
            for l in probs:
                probs_list.append(l[1])
            precision, recall, _ = precision_recall_curve(y_valid_perm, probs_list)
            auprc_score = auc(recall, precision)
            auc_score = roc_auc_score(y_valid_perm, probs_list)
            
        (domain_auc_mean, domain_auprc_mean, domain_auprc_ratio_mean) = compute_per_domain_auc(y_valid_perm, probs_list, pred_idx, classifier)

        print "AUPRC = "+str(auprc_score)
        print "AUC = "+str(auc_score)
        print "domain AUC mean = "+str(domain_auc_mean)
        print "domain AUPRC mean = "+str(domain_auprc_mean)
        print "domain AUPRC ratio mean = "+str(domain_auprc_ratio_mean)
        
        #Removing from mean calculations the trials with high AUPRC and random AUC
        if (auc_score > 0.51):
            trial_auprc_results[i] = auprc_score 
            trial_auc_results[i] = auc_score 
            trial_domain_auc_results[i] = domain_auc_mean
            trial_domain_auprc_results[i] = domain_auprc_mean
            trial_domain_auprc_ratio_results[i] = domain_auprc_ratio_mean
            if classifier == "NN" or classifier == "XGB": 
                epoch_counts[i] = epoch_count
    
    mean_auprc_result = np.mean(np.array(trial_auprc_results)[np.nonzero(trial_auprc_results)[0].tolist()])
    mean_auc_result = np.mean(np.array(trial_auc_results)[np.nonzero(trial_auc_results)[0].tolist()])
    mean_domain_auc_result = np.mean(np.array(trial_domain_auc_results)[np.nonzero(trial_domain_auc_results)[0].tolist()])
    mean_domain_auprc_result = np.mean(np.array(trial_domain_auprc_results)[np.nonzero(trial_domain_auprc_results)[0].tolist()])
    mean_domain_auprc_ratio_result = np.mean(np.array(trial_domain_auprc_ratio_results)[np.nonzero(trial_domain_auprc_ratio_results)[0].tolist()])
    
    
    if classifier == "NN" or classifier == "XGB":
        #Handle the case of everything is 0?
        try:
            mean_epoch_count = int(np.mean(np.array(epoch_counts)[np.nonzero(epoch_counts)[0].tolist()]))
        except: 
            mean_epoch_count = 0
        hyperparameters_dict["mean_epoch_count"] = mean_epoch_count

    hyperparameters_dict["mean_AUPRC"] = mean_auprc_result
    hyperparameters_dict["mean_AUC"] = mean_auc_result
    hyperparameters_dict["mean_dom_AUC"] = mean_domain_auc_result
    hyperparameters_dict["mean_dom_AUPRC"] = mean_domain_auprc_result
    hyperparameters_dict["mean_dom_AUPRC_ratio"] = mean_domain_auprc_ratio_result
    hyperparameters_dict["trial_idx"] = trial_idx
    hyperparameters_dict["num_legal_folds"] = len(np.nonzero(epoch_counts)[0].tolist())

    # Update dictionary with all hyperparameters
    keys = hyperparameters.keys()
    for key in keys:
        hyperparameters_dict[key].append(hyperparameters[key])
    pred_idx += 1

    print "Finished "+ligand_name+" "+classifier+" fold: "+fold+" trial: "+str(trial_idx)
#====================================================================================================================#
    
def test_model_on_heldout(hyperparameters_dict, hyperparameters_input, ligand_bind_features, ligand_negatives_features, ligand_name, 
                          classifier_method, fold, features=[], final_model=False, test_positives=None, test_negatives=None, rseed=0):
    
    #Default: Exclude no features
    if len(features) == 0:
        features = np.ones([ligand_bind_features.shape[1],]).astype(bool)
    
    models_req_scaling = ["SVM", "KNN", "Logistic", "NN"]

    classifier = classifier_method

    #Create X and y with included features
    X = pd.concat([ligand_bind_features.iloc[:,features], ligand_negatives_features.iloc[:,features]])
    y = [1] * ligand_bind_features.shape[0]
    y.extend([0] * ligand_negatives_features.shape[0])
    y = np.array(y)
    y_df = pd.DataFrame(y)
    y_df.index = X.index
    y_df.columns = ["label"]
    
    #Get the fold indices
    cv_idx = calc_CV_idx_iterative(X, splits_dict)
    k = (int(fold)-1)
    
    pred_idx = k+1
    print "fold #: "+str(pred_idx)
    

    if (final_model):
        #All the training data is used to train the model
        X_train = X
        y_train = y_df
        
        #Create X and y from the test set
        X_t = pd.concat([test_positives.iloc[:,features], test_negatives.iloc[:,features]])
        y_t = [1] * test_positives.shape[0]
        y_t.extend([0] * test_negatives.shape[0])
        y_t = np.array(y_t)
        y_t_df = pd.DataFrame(y_t)
        y_t_df.index = X_t.index
        y_t_df.columns = ["label"]
        
        #Define test sets
        X_test = X_t
        y_test = y_t_df
        
        #Define indices
        test_index = X_test.index
        train_index = X_train.index
        
    else:
        test_index = cv_idx[k]["test"]
        train_index = cv_idx[k]["train"]
        X_train, X_test = X.loc[train_index,:], X.loc[test_index,:]
        y_train, y_test = y_df.loc[train_index,:], y_df.loc[test_index,:]
    
    if (classifier in models_req_scaling):
        cols = X_train.columns
        scaler = StandardScaler() 
        #scale only using the training data
        scaler.fit(X_train) 
        X_train = pd.DataFrame(scaler.transform(X_train))
        # apply same transformation to test data
        X_test = pd.DataFrame(scaler.transform(X_test))
        #Restoring indices after scaling
        X_train.index = train_index 
        X_test.index = test_index 
        #Restoring features names
        X_train.columns = cols
        X_test.columns = cols

    #No down-sampling
    X_train_sampled = X_train
    y_train_sampled = y_train
    
    #Shuffle training data rows
    np.random.seed(rseed)
    idx_perm_train = np.random.permutation(X_train_sampled.index)
    X_train_sampled_perm = X_train_sampled.reindex(idx_perm_train)
    y_train_sampled_perm = y_train_sampled.reindex(idx_perm_train)
        
    #Shuffle test data rows
    idx_perm_test = np.random.permutation(X_test.index)
    X_test_perm = X_test.reindex(idx_perm_test)
    y_test_perm = y_test.reindex(idx_perm_test)

    #pos and neg numbers in the training
    no_pos = np.count_nonzero(y_train_sampled["label"] == 1)
    no_neg = np.count_nonzero(y_train_sampled["label"] == 0)
    
    #fit to training data
    model = generate_model(classifier_method, hyperparameters_input, no_pos=no_pos, no_neg=no_neg, features_num=X_train_sampled_perm.shape[1], test_mode=True, rseed=rseed)
    
    #pos and neg numbers in the training
    no_pos = np.count_nonzero(y_train_sampled_perm["label"] == 1)
    no_neg = np.count_nonzero(y_train_sampled_perm["label"] == 0)  
    if classifier == "NN":     
        #weight vector for NN
        if model.weight == "balanced":              
            #weight vector
            neg_weight = float(no_pos) / float(no_neg + no_pos) 
            pos_weight = 1 - neg_weight
        elif model.weight == "0.1":
            neg_weight = 10
            pos_weight = 1
        elif model.weight == "None":
            neg_weight = 1
            pos_weight = 1
        
        
        weight = torch.Tensor([neg_weight, pos_weight]).to(device=curr_device)
        model.fit(X_train_sampled_perm, y_train_sampled_perm["label"], weight)
        probs_list = model.predict_proba(X_test_perm)
    
    elif classifier == "ADA":
        print "fiting calibrated model"
        calib_model = CalibratedClassifierCV(base_estimator=model)
        calib_model.fit(X_train_sampled_perm, y_train_sampled_perm["label"])
        probs_list = []
        probs = calib_model.predict_proba(X_test_perm)
        for l in probs:
            probs_list.append(l[1])
    else:
        model.fit(X_train_sampled_perm, y_train_sampled_perm["label"])
        probs_list = []
        probs = model.predict_proba(X_test_perm)
        for l in probs:
            probs_list.append(l[1])

    #Update auc auprc dictionaries
    hyperparameters_dict["test_AUC"] = roc_auc_score(y_test_perm["label"], probs_list)
    precision, recall, _ = precision_recall_curve(y_test_perm["label"], probs_list)
    hyperparameters_dict["test_AUPRC"] = auc(recall, precision)
    
    #Compute per domain AUC and AUPRC
    (domain_auc_mean, domain_auprc_mean, domain_auprc_ratio_mean) = compute_per_domain_auc(y_test_perm, probs_list, pred_idx, classifier)
    #Update relevant dictionaries for per-domain folds mean
    hyperparameters_dict["test_dom_AUC"] = domain_auc_mean
    hyperparameters_dict["test_dom_AUPRC"] = domain_auprc_mean
    hyperparameters_dict["test_dom_AUPRC_ratio"] = domain_auprc_ratio_mean
    
    pred_idx += 1

    print "test AUC = "+str(hyperparameters_dict["test_AUC"])
    print "test AUPRC = "+str(hyperparameters_dict["test_AUPRC"])
    print "test domain AUC mean = "+str(hyperparameters_dict["test_dom_AUC"])
    print "test domain AUPRC mean = "+str(hyperparameters_dict["test_dom_AUPRC"])
    print "test domain AUPRC ratio mean = "+str(hyperparameters_dict["test_dom_AUPRC_ratio"])

    print "Finished "+ligand_name+" "+classifier+" fold: "+fold
    
    if (rseed > 0):
        return (model)
