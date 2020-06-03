import pickle
import subprocess
from os import getcwd
import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, precision_score, fbeta_score

curr_dir = getcwd()
exac_dir = curr_dir[:curr_dir.find("ExAC")]
sys.path.append(exac_dir+"ExAC/10.Prediction/utils")
from prediction_general_funcs import ligands, compute_per_domain_auc
from tuning_helper_functions import models_req_scaling

#Ligands & models order is important for using pickled models
ligands = ["dna", "rna", "ion", "peptide", "sm"]
all_models_list = ["XGB", "RF", "SVM", "Logistic", "NN"]
#====================================================================================================================#

def get_final_model(ligand, classifier_method, stacked=False, ens_dir=""):
    "Return the saved model with highest AUPRC"
    
    #Getting the list of available models
    if (stacked):
        models_path = "/home/anat/Research/ExAC/14.Final_Model/stacked_pik_models/"+ens_dir
    else:
        models_path = "/home/anat/Research/ExAC/14.Final_Model/pik_models"
    cmd = "ls "+models_path+" | grep "+ligand+"_"+classifier_method+"* | cut -d'_' -f3 | cut -d'.' -f1"
    ls_cmd_out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    rand_trials = []
    
    for line in ls_cmd_out.stdout.readlines():
        rand_trials.append(int(line))
    rand_trials = list(set(rand_trials))
    

    #Finding the model with the best AUPRC
    performance_df = pd.DataFrame()
    best_auprc = 0
    for trial in rand_trials:

        trial_filename = models_path+"/"+ligand+"_"+classifier_method+"_"+str(trial)+".csv"
        trial_hyperparameters_df = pd.read_csv(trial_filename, sep='\t', index_col=0)
        trial_hyperparameters_df.index = [trial]
        performance_df = performance_df.append(trial_hyperparameters_df)

        curr_auprc = float(trial_hyperparameters_df["test_AUPRC"])
        if (curr_auprc > best_auprc):
            best_auprc = curr_auprc
            best_trial = trial

    #Get the best model
    
    with open(models_path+"/"+ligand+"_"+classifier_method+"_"+str(best_trial)+".pik", 'rb') as handle:
        model = pickle.load(handle)
        
    return (model, best_trial)
#====================================================================================================================#

def predcit_using_model(pred_dict, domain_pred_dict, auc_dict, auprc_dict, domain_auc_mean_dict, domain_auprc_mean_dict, domain_auprc_ratio_mean_dict, domain_auc_dict, 
                               domain_auprc_dict, domain_auprc_ratio_dict, prec_prob_dict, test_positives, test_negatives, model, classifier, rseed, stacked=False, ens=""):
    
    
    #Create X and y from the test set
    X_t = pd.concat([test_positives, test_negatives])
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
    
    if (classifier in models_req_scaling):
        cols = X_test.columns
        #Read the saved Scaler
        if (stacked):
            with open(curr_dir+"/stacked_pik_models/"+ens+"/scaler.pik", 'rb') as handle:
                scaler = pickle.load(handle)
        else:
            with open(curr_dir+"/pik_models/scaler.pik", 'rb') as handle:
                scaler = pickle.load(handle)
        # apply same transformation to test data
        X_test = pd.DataFrame(scaler.transform(X_test))
        #Restoring indices after scaling
        X_test.index = test_index 
        #Restoring features names
        X_test.columns = cols
    
    #Shuffle test data rows
    np.random.seed(rseed)
    idx_perm_test = np.random.permutation(X_test.index)
    X_test_perm = X_test.reindex(idx_perm_test)
    y_test_perm = y_test.reindex(idx_perm_test)
    
    #Predict on the test set
    probs = model.predict_proba(X_test_perm)
    if (classifier == "NN"): 
        probs_list = probs
    else:
        probs_list = []
        for l in probs:
            probs_list.append(l[1])
            
    
    pred_dict["obs"].extend(y_test_perm["label"])
    pred_dict["prob"].extend(probs_list)
    
    model_list = [classifier] * len(probs_list)
    pred_dict["model"].extend(model_list)
    
    #Adding the position number to the table to help with analysis
    pred_dict["idx"].extend(idx_perm_test)
            
    #Update auc auprc dictionaries
    auc_dict[classifier].append(roc_auc_score(y_test_perm["label"], probs_list))
    precision, recall, _ = precision_recall_curve(y_test_perm["label"], probs_list)
    auprc_dict[classifier].append(auc(recall, precision))
    
    #Compute per domain AUC and AUPRC
    (domain_auc_mean, domain_auprc_mean, domain_auprc_ratio_mean, domain_auc_list, domain_auprc_list, domain_auprc_ratio_list, domain_name_list, domain_pos_num_list, domain_neg_num_list) = compute_per_domain_auc(y_test_perm, probs_list, domain_pred_dict,"1", classifier)
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
    
    print "AUC = "+str(auc_dict[classifier][-1])
    print "AUPRC = "+str(auprc_dict[classifier][-1])
    print "domain AUC mean = "+str(domain_auc_mean_dict[classifier][-1])
    print "domain AUPRC mean = "+str(domain_auprc_mean_dict[classifier][-1])
    print "domain AUPRC ratio mean = "+str(domain_auprc_ratio_mean_dict[classifier][-1])