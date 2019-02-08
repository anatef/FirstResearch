#General functions and constants to use in 10.Prediction code files
import pandas as pd
import numpy as np
from os import getcwd
import pickle
import sys

#ML framework imports
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, precision_score, fbeta_score
from sklearn.preprocessing import StandardScaler

#Import from other utils files
from CV_funcs import add_domain_name_from_table_idx, calc_CV_idx_iterative
from NN_classes import curr_device

ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "druglike", "sm", "all"]
score_cols_suffix = ["_propensity", "_prop_th_0.1", "_prop_th_0.25", "_prop_th_0.5", "_prop_th_0.75"]

#CV splits dictionary
curr_dir = getcwd()
exac_dir = curr_dir[:curr_dir.find("ExAC")]
pfam_version = "31"
folds_num = 5
with open(exac_dir+"ExAC/10.Prediction/CV_splits/pfam-v"+pfam_version+"/domain_"+str(folds_num)+"_folds_combined_dna0.5_rna0.5_ion0.75_prec_dict.pik", 'rb') as handle:
    splits_dict = pickle.load(handle)
#====================================================================================================================#

def get_features_cols(features_all):
    """
    Returning a list of features column names
    """
    
    features_cols = features_all.columns.tolist()
    #removing binding scores and domain name
    for ligand in ligands:
        for suffix in score_cols_suffix:
            features_cols.remove(ligand+suffix)
    features_cols.remove("domain_name")
    
    return features_cols
#====================================================================================================================#

def remove_unimportant_features(features_table, features_cols, additional_removal_features = [], update_features_cols=False):
    """
    Removing features that aren't useful for the prediction.
    """
    
    GO_BEG1 = 425
    GO_END1 = 432
    
    #Remove domain id feature that was added just as a sanity check
    features_for_removal = ["domain_id"]
    
    #Remove GO terms as these features are missing got most domains and incomplete
    GO_features = features_cols[GO_BEG1:GO_END1]
    features_for_removal.extend(GO_features)
    
    #Removing also features from the input
    features_for_removal.extend(additional_removal_features)
    
    for feature in features_for_removal:
        del features_table[feature]   
    
    #Remove the features from the featues_cols list
    if (update_features_cols):
        for feature in features_for_removal:
                features_cols.remove(feature)
#====================================================================================================================#

def compute_per_domain_auc(y_test, pred_probs, domain_pred_dict, pred_idx, classifier):
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
      
        domain_pred_dict["obs"].extend(domain_df["label"])
        domain_pred_dict["prob"].extend(domain_df["pred_probs"])
        fold_list = [pred_idx] * len(domain_df["pred_probs"])
        domain_pred_dict["fold"].extend(fold_list)
        model_list = [classifier] * len(domain_df["pred_probs"])
        domain_pred_dict["model"].extend(model_list)
        domain_str_list = [domain_name] * len(domain_df["pred_probs"])
        domain_pred_dict["domain"].extend(domain_str_list)
        
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
    
    return (domain_auc_mean, domain_auprc_mean, domain_auprc_ratio_mean, domain_auc_list, domain_auprc_list, domain_auprc_ratio_list, domain_name_list, domain_pos_num_list, domain_neg_num_list)
#====================================================================================================================#

def area_under_precision_prob_curve(y_true, y_probs):
    
    #probs_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0]
    probs_list = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0]
    probs_vals = []
    precision_vals = []
    
    for prob in probs_list:
        binary_decision = [1 if x >= prob else 0 for x in y_probs]
        if (np.count_nonzero(binary_decision) == 0):
            continue
        precision_vals.append(fbeta_score(y_true, binary_decision, 0.001))
        probs_vals.append(prob)
    
    return auc(probs_vals, precision_vals)
#====================================================================================================================#

def test_model_iterative_fixed(pred_dict, domain_pred_dict, auc_dict, auprc_dict, domain_auc_mean_dict, domain_auprc_mean_dict, domain_auprc_ratio_mean_dict, domain_auc_dict, 
                               domain_auprc_dict, domain_auprc_ratio_dict, prec_prob_dict,
                               ligand_bind_features, ligand_negatives_features, ligand, model, classifier_method, fold, features=[]):
    
    """
    Train on (k-1) folds, predict for the kth fold, given a model (no tuning is done here)
    Return all predictions probabilities and prediction performance.
    """
    
    #Default: Exclude no features
    if len(features) == 0:
        features = np.ones([ligand_bind_features.shape[1],]).astype(bool)
        
    #Arranging the features table by the CV order, for each model
    features_pred_dfs = dict.fromkeys(classifier_method)
    
    models_req_scaling = ["SVM", "KNN", "Logistic", "NN"]

    classifier = classifier_method
    features_pred_dfs[classifier] = pd.DataFrame()

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
    
    #pos and neg numbers in the training
    no_pos = np.count_nonzero(y_train_sampled["label"] == 1)
    no_neg = np.count_nonzero(y_train_sampled["label"] == 0)  
    if classifier == "NN":  
        model = model.to(device=curr_device)
        #weight vector for NN
        if model.weight == "balanced":              
            #weight vector
            neg_weight = float(no_pos) / float(no_neg + no_pos) 
            pos_weight = 1 - neg_weight
        elif model.weight == 0.1:
            neg_weight = 10
            pos_weight = 1
        elif model.weight == None:
            neg_weight = 1
            pos_weight = 1

        weight = torch.Tensor([neg_weight, pos_weight]).to(device=curr_device)
        model.fit(X_train_sampled, y_train_sampled["label"], weight)
        probs_list = model.predict_proba(X_test)
    
    else:
        
        model.fit(X_train_sampled, y_train_sampled["label"])
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
    
    #Adding the position number to the table to help with analysis
    pred_dict["idx"].extend(test_index)

    #Update auc auprc dictionaries
    auc_dict[classifier].append(roc_auc_score(y_test["label"], probs_list))
    precision, recall, _ = precision_recall_curve(y_test["label"], probs_list)
    auprc_dict[classifier].append(auc(recall, precision))
    prec_prob_dict[classifier].append(area_under_precision_prob_curve(y_test["label"], probs_list))
    
    #Compute per domain AUC and AUPRC
    #(domain_auc_mean, domain_auprc_mean, domain_auprc_ratio_mean, domain_auc_list, domain_auprc_list, domain_auprc_ratio_list, domain_name_list, domain_pos_num_list, domain_neg_num_list) = compute_per_domain_auc(y_test, probs[:, 1], domain_pred_dict,pred_idx, classifier)
    (domain_auc_mean, domain_auprc_mean, domain_auprc_ratio_mean, domain_auc_list, domain_auprc_list, domain_auprc_ratio_list, domain_name_list, domain_pos_num_list, domain_neg_num_list) = compute_per_domain_auc(y_test, probs_list, domain_pred_dict,pred_idx, classifier)
    #Update relevant dictionaries for per-domain folds mean
    domain_auc_mean_dict[classifier].append(domain_auc_mean)
    domain_auprc_mean_dict[classifier].append(domain_auprc_mean)
    domain_auprc_ratio_mean_dict[classifier].append(domain_auprc_ratio_mean)
    
    #Update relevant dictionaries for per-domain individual metrices scores
    domain_auc_dict[classifier].extend(domain_auc_list)
    domain_auc_dict["domain"].extend(domain_name_list)
    domain_auc_dict["pos_num"].extend(domain_pos_num_list)
    domain_auc_dict["neg_num"].extend(domain_neg_num_list)
    
    domain_auprc_dict[classifier].extend(domain_auprc_list)
    domain_auprc_dict["domain"].extend(domain_name_list)
    domain_auprc_dict["pos_num"].extend(domain_pos_num_list)
    domain_auprc_dict["neg_num"].extend(domain_neg_num_list)
    
    domain_auprc_ratio_dict[classifier].extend(domain_auprc_ratio_list)
    domain_auprc_ratio_dict["domain"].extend(domain_name_list)
    domain_auprc_ratio_dict["pos_num"].extend(domain_pos_num_list)
    domain_auprc_ratio_dict["neg_num"].extend(domain_neg_num_list)
    
    #Update features table
    features_pred_dfs[classifier] = features_pred_dfs[classifier].append(X_test)
    pred_idx += 1

    print "AUC = "+str(auc_dict[classifier][-1])
    print "AUPRC = "+str(auprc_dict[classifier][-1])
    print "AU prec prob = "+str(prec_prob_dict[classifier][-1])
    print "domain AUC mean = "+str(domain_auc_mean_dict[classifier][-1])
    print "domain AUPRC mean = "+str(domain_auprc_mean_dict[classifier][-1])
    print "domain AUPRC ratio mean = "+str(domain_auprc_ratio_mean_dict[classifier][-1])

    print "Finished "+ligand+" "+classifier+" fold: "+fold
    
    return (features_pred_dfs, model)
#====================================================================================================================#

def test_model_no_performance(pred_dict, ligand_bind_features, ligand_negatives_features, ligand_missing_df, ligand, model,
                              classifier, fold, features=[]):
    
    """
    Test model without calculating performance (test data has no labels)
    """
    
    #Default: Exclude no features
    if len(features) == 0:
        features = np.ones([ligand_bind_features.shape[1],]).astype(bool)
        
    #Arranging the features table by the CV order, for each model
    features_pred_dfs = dict.fromkeys(classifier)
    
    models_req_scaling = ["SVM", "KNN", "Logistic", "NN"]

    features_pred_dfs[classifier] = pd.DataFrame()

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

    #fit to training data
    
    
    #pos and neg numbers in the training
    no_pos = np.count_nonzero(y_train_sampled["label"] == 1)
    no_neg = np.count_nonzero(y_train_sampled["label"] == 0)  
    
    if classifier == "NN":  
        model = model.to(device=curr_device)
        #weight vector for NN
        if model.weight == "balanced":              
            #weight vector
            neg_weight = float(no_pos) / float(no_neg + no_pos) 
            pos_weight = 1 - neg_weight
        elif model.weight == 0.1:
            neg_weight = 10
            pos_weight = 1
        elif model.weight == None:
            neg_weight = 1
            pos_weight = 1

        weight = torch.Tensor([neg_weight, pos_weight]).to(device=curr_device)
        model.fit(X_train_sampled, y_train_sampled["label"], weight)
        probs_list = model.predict_proba(ligand_missing_df)
    
    else:
        
        model.fit(X_train_sampled, y_train_sampled["label"])
        probs_list = []
        probs = model.predict_proba(ligand_missing_df)
        for l in probs:
            probs_list.append(l[1])

    pred_dict["prob"].extend(probs_list)
    fold_list = [pred_idx] * len(probs_list)
    pred_dict["fold"].extend(fold_list)

    model_list = [classifier] * len(probs_list)
    pred_dict["model"].extend(model_list)
    
    #Adding the position number to the table to help with analysis
    pred_dict["idx"].extend(ligand_missing_df.index)
    
    #Update features table
    features_pred_dfs[classifier] = features_pred_dfs[classifier].append(ligand_missing_df)
    pred_idx += 1

    print "Finished "+ligand+" "+classifier+" fold: "+fold
    
    return features_pred_dfs