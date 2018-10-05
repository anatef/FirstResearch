import time
import multiprocessing as mp
import shlex
import subprocess
import numpy as np
import os
import sys

folds_num = 1
trials_num = 1

#One run
ligands = ["peptide"]        
classifiers = ["XGB"]

def run_trials(trial, classifier, ligand, fold):
        
    hp_dict = dict()
    # hp_dict["XGB"] = {"max_depth_ub": "100",
    #                   "max_depth_lb": "1",               
    #                   "min_child_weight_ub": "2",
    #                   "min_child_weight_lb": "0",
    #                   "colsample_bytree_ub": "1",
    #                   "colsample_bytree_lb": "0.25",
    #                   "gamma_ub": "0",
    #                   "gamma_lb": "-4",
    #                   "learning_rate_ub": "-0.5",
    #                   "learning_rate_lb": "-5"}
    hp_dict["XGB"] = {"max_depth_ub": "35",
                      "max_depth_lb": "25",               
                      "sec_max_depth_ub": "95",
                      "sec_max_depth_lb": "40",
                      "max_depth_weight_1": "0.5",
                      "max_depth_weight_2": "0.5",
                      "min_child_weight_ub": "0.25",
                      "min_child_weight_lb": "0",
                      "sec_min_child_weight_ub": "5",
                      "sec_min_child_weight_lb": "1",
                      "min_child_weight_weight_1": "0.25",
                      "min_child_weight_weight_2": "0.75",
                      "colsample_bytree_ub": "0.7",
                      "colsample_bytree_lb": "0.4",
                      "sec_colsample_bytree_ub": "1",
                      "sec_colsample_bytree_lb": "0.9",
                      "colsample_bytree_weight_1": "0.85",
                      "colsample_bytree_weight_2": "0.15",
                      "gamma_ub": "-1.3",
                      "gamma_lb": "-2.8",
                      "sec_gamma_ub": "-0.15",
                      "sec_gamma_lb": "-1.05",
                      "gamma_weight_1": "0.6",
                      "gamma_weight_2": "0.4",
                      "learning_rate_ub": "-2.05",
                      "learning_rate_lb": "-2.7",
                      "sec_learning_rate_ub": "-1.3",
                      "sec_learning_rate_lb": "-2",
                      "lr_weight_1": "0.5",
                      "lr_weight_2": "0.5"}

    hp_dict["RF"] = {"n_estimators_ub": "1500",
                    "n_estimators_lb": "100",
                    "max_depth_ub": "20",
                    "max_depth_lb": "2",
                    "min_samples_leaf_ub": "50",
                    "min_samples_leaf_lb": "1",
                    "min_samples_split_ub": "50",
                    "min_samples_split_lb": "2"}

    hp_dict["ADA"] = {"n_estimators_ub": "1500",
                     "n_estimators_lb": "100",
                     "learning_rate_ub": "0",
                     "learning_rate_lb": "-3"}

    hp_dict["SVM"] = {"C_ub": "2",
                     "C_lb": "-4",
                     "gamma_ub": "-1",
                     "gamma_lb": "-5"}

    hp_dict["Logistic"] = {"C_ub": "0",
                          "C_lb": "-3"}

    hp_dict["KNN"] = {"n_neighbors_ub": "300",
                     "n_neighbors_lb": "150",
                     "sec_n_neighbors_ub": "1000",
                     "sec_n_neighbors_lb": "450",
                     "n_neighbors_weight_1": "0.25",
                     "n_neighbors_weight_2": "0.75"}

    hp_dict["NN"] = {"learning_rate_ub":"-2",
                    "learning_rate_lb":"-6",
                    "batch_size_ub":"300",
                    "batch_size_lb":"30",
                    "weight_decay_ub":"-5",
                    "weight_decay_lb":"-25",
                    "beta_ub":"0.95",
                    "beta_lb":"0.85",
                    "hidden_units_1_ub":"300",
                    "hidden_units_1_lb":"50",
                    "hidden_units_2_ub":"1000",
                    "hidden_units_2_lb":"350"}
    
    params = hp_dict[classifier]
    
    print "ligand = "+ligand
    print "fold  = " +fold
    print "classifier = "+classifier
    print "trial = "+str(trial)
    
    if (classifier == "XGB"):
        script_text = ("ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial="+str(trial)+""
        " max_depth_ub="+params["max_depth_ub"]+" max_depth_lb="+params["max_depth_lb"]+""
        " sec_max_depth_ub="+params["sec_max_depth_ub"]+" sec_max_depth_lb="+params["sec_max_depth_lb"]+""
        " max_depth_weight_1="+params["max_depth_weight_1"]+" max_depth_weight_2="+params["max_depth_weight_2"]+""
        " min_child_weight_ub="+params["min_child_weight_ub"]+" min_child_weight_lb="+params["min_child_weight_lb"]+""
        " sec_min_child_weight_ub="+params["sec_min_child_weight_ub"]+" sec_min_child_weight_lb="+params["sec_min_child_weight_lb"]+""
        " min_child_weight_weight_1="+params["min_child_weight_weight_1"]+" min_child_weight_weight_2="+params["min_child_weight_weight_2"]+""
        " colsample_bytree_ub="+params["colsample_bytree_ub"]+" colsample_bytree_lb="+params["colsample_bytree_lb"]+""
#                 " sec_colsample_bytree_ub="+params["sec_colsample_bytree_ub"]+" sec_colsample_bytree_lb="+params["sec_colsample_bytree_lb"]+""
#                 " colsample_bytree_weight_1="+params["colsample_bytree_weight_1"]+" colsample_bytree_weight_2="+params["colsample_bytree_weight_2"]+""
        " gamma_ub="+params["gamma_ub"]+" gamma_lb="+params["gamma_lb"]+""
        " sec_gamma_ub="+params["sec_gamma_ub"]+" sec_gamma_lb="+params["sec_gamma_lb"]+""
        " gamma_weight_1="+params["gamma_weight_1"]+" gamma_weight_2="+params["gamma_weight_2"]+""
        " learning_rate_ub="+params["learning_rate_ub"]+" learning_rate_lb="+params["learning_rate_lb"]+""
        " sec_learning_rate_ub="+params["sec_learning_rate_ub"]+" sec_learning_rate_lb="+params["sec_learning_rate_lb"]+""
        " lr_weight_1="+params["lr_weight_1"]+" lr_weight_2="+params["lr_weight_2"]+""
        " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_"+str(trial)+"_5w.ipynb")

    elif (classifier == "RF"):
        script_text = ("ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial="+str(trial)+""
        " n_estimators_ub="+params["n_estimators_ub"]+" n_estimators_lb="+params["n_estimators_lb"]+""
        " max_depth_ub="+params["max_depth_ub"]+" max_depth_lb="+params["max_depth_lb"]+""
        " min_samples_leaf_ub="+params["min_samples_leaf_ub"]+" min_samples_leaf_lb="+params["min_samples_leaf_lb"]+""
        " min_samples_split_ub="+params["min_samples_split_ub"]+" min_samples_split_lb="+params["min_samples_split_lb"]+""
        " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_"+str(trial)+"_5w.ipynb")

    elif (classifier == "ADA"):
        script_text = ("ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial="+str(trial)+""
        " n_estimators_ub="+params["n_estimators_ub"]+" n_estimators_lb="+params["n_estimators_lb"]+""
        " learning_rate_ub="+params["learning_rate_ub"]+" learning_rate_lb="+params["learning_rate_lb"]+""
        " runipy > reports/"+ligand+"_"+classifier+"_"+fold+"_"+str(trial)+"_5w.ipynb")

    elif (classifier == "SVM"):
        script_text = ("ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial="+str(trial)+""
        " C_ub="+C_ub+" C_lb="+C_lb+""
        " gamma_ub="+params["gamma_ub"]+" gamma_lb="+params["gamma_lb"]+""
        " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_"+str(trial)+"_5w.ipynb")

    elif (classifier == "Logistic"):
        script_text = ("cligand="+ligand+" fold="+fold+" classifier="+classifier+" trial="+str(trial)+""
        " C_ub="+params["C_ub"]+" C_lb="+params["C_lb"]+""
        " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_"+str(trial)+"_5w.ipynb")

    elif (classifier == "KNN"):
        script_text = ("ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial="+str(trial)+""
        " n_neighbors_ub="+params["n_neighbors_ub"]+" n_neighbors_lb="+params["n_neighbors_lb"]+""
        " sec_n_neighbors_ub="+params["sec_n_neighbors_ub"]+" sec_n_neighbors_lb="+params["sec_n_neighbors_lb"]+""
        " n_neighbors_weight_1="+params["n_neighbors_weight_1"]+" n_neighbors_weight_2="+params["n_neighbors_weight_2"]+""               
        " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_"+str(trial)+"_5w.ipynb")

    elif (classifier == "NN"):
        script_text = ("ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial="+str(trial)+""
        " learning_rate_ub="+params["learning_rate_ub"]+" learning_rate_lb="+params["learning_rate_lb"]+" batch_size_ub="+params["batch_size_ub"]+""
        " batch_size_lb="+params["batch_size_lb"]+" weight_decay_ub="+params["weight_decay_ub"]+" weight_decay_lb="+params["weight_decay_lb"]+""
        " beta_ub="+params["beta_ub"]+" beta_lb="+params["beta_lb"]+" hidden_units_1_ub="+params["hidden_units_1_ub"]+" hidden_units_1_lb="+params["hidden_units_1_lb"]+""
        " hidden_units_2_ub="+params["hidden_units_2_ub"]+" hidden_units_2_lb="+params["hidden_units_2_lb"]+""
        " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_"+str(trial)+"_5w.ipynb")
    
    time.sleep(trial*30)
    print "starting "+ligand+" "+classifier+" fold = "+fold+" trial= "+str(trial)
    myinput = open("/home/ubuntu/ExAC/10.Prediction/phase1_models_tuning.ipynb", 'r')
    subprocess.Popen(script_text, stdin=myinput, shell=True)
    #os.system(script_text)
    

for j in range(len(classifiers)):
    classifier = classifiers[j]
    
    
    for i in range(len(ligands)):
        ligand = ligands[i]
       
        #for k in range(1,folds_num+1):
        for k in range(3,4):
            fold = str(k)
            trials = range(20)
            NUM_CPUS = 20
            pool = mp.Pool(processes = NUM_CPUS)
            jobs = [pool.apply_async(run_trials, args = (args, classifier, ligand, fold)) for args in trials]
            results = [l.get() for l in jobs]
            
            
            
