import time
import os

#Add: #SBATCH --gres=gpu:1 for running on GPU
#Add: SBATCH --time=00:00:00 for running on GPU
folds_num = 5
trials_num = 100

#Params to run on
#ligands = ["dna","sm", "metabolite", "peptide", "ion", "rna"]
#ligands_names = ["dna","sm", "met", "pep", "ion", "rna"]
# classifiers = ["XGB", "RF", "ADA", "SVM", "Logistic", "KNN"]
# classifiers_names = ["XB", "RF", "AD", "SV", "LG", "KN"]

#One run
ligands = ["ion"]
ligands_names = ["ion"]
classifiers = ["XGB"]
classifiers_names = ["XB"]

#Hyperparameters ranges dictionaries
hp_dict = dict()
# XGB test case where just max_depth has a disjoint interval to sample from
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
hp_dict["XGB"] = {"max_depth_ub": "40",
                  "max_depth_lb": "25",               
                  "sec_max_depth_ub": "90",
                  "sec_max_depth_lb": "40",
                  "max_depth_weight_1": "0.5",
                  "max_depth_weight_2": "0.5",
                  "min_child_weight_ub": "1.5",
                  "min_child_weight_lb": "1",
                  "sec_min_child_weight_ub": "5",
                  "sec_min_child_weight_lb": "1.8",
                  "min_child_weight_weight_1": "0.5",
                  "min_child_weight_weight_2": "0.5",
                  "colsample_bytree_ub": "0.95",
                  "colsample_bytree_lb": "0.3",
#                   "sec_colsample_bytree_ub": "1",
#                   "sec_colsample_bytree_lb": "0.6",
#                   "colsample_bytree_weight_1": "0.8",
#                   "colsample_bytree_weight_2": "0.2",
                  "gamma_ub": "-1.3",
                  "gamma_lb": "-2.8",
                  "sec_gamma_ub": "-0.085",
                  "sec_gamma_lb": "-0.1",
                  "gamma_weight_1": "0.85",
                  "gamma_weight_2": "0.15",
                  "learning_rate_ub": "-2.05",
                  "learning_rate_lb": "-2.7",
                  "sec_learning_rate_ub": "-1.15",
                  "sec_learning_rate_lb": "-2",
                  "lr_weight_1": "0.3",
                  "lr_weight_2": "0.7"}
                   

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


#Looping over the jobs
for j in range(len(classifiers)):
    classifier = classifiers[j]
    params = hp_dict[classifier]
    
    for i in range(len(ligands)):
        ligand = ligands[i]
       
        for k in range(1,folds_num+1):
            fold = str(k)
            
            header ="""#!/bin/bash

#SBATCH --mem=40960
#SBATCH --qos=1day
#SBATCH --job-name={0}_{1}_{2}
#SBATCH --mail-user=anatf@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(classifiers_names[j],ligands_names[i], fold)

            if (classifier == "XGB"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
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
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "RF"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " n_estimators_ub="+params["n_estimators_ub"]+" n_estimators_lb="+params["n_estimators_lb"]+""
                " max_depth_ub="+params["max_depth_ub"]+" max_depth_lb="+params["max_depth_lb"]+""
                " min_samples_leaf_ub="+params["min_samples_leaf_ub"]+" min_samples_leaf_lb="+params["min_samples_leaf_lb"]+""
                " min_samples_split_ub="+params["min_samples_split_ub"]+" min_samples_split_lb="+params["min_samples_split_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "ADA"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " n_estimators_ub="+params["n_estimators_ub"]+" n_estimators_lb="+params["n_estimators_lb"]+""
                " learning_rate_ub="+params["learning_rate_ub"]+" learning_rate_lb="+params["learning_rate_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "SVM"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " C_ub="+C_ub+" C_lb="+C_lb+""
                " gamma_ub="+params["gamma_ub"]+" gamma_lb="+params["gamma_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "Logistic"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " C_ub="+params["C_ub"]+" C_lb="+params["C_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "KNN"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " n_neighbors_ub="+params["n_neighbors_ub"]+" n_neighbors_lb="+params["n_neighbors_lb"]+""
                " sec_n_neighbors_ub="+params["sec_n_neighbors_ub"]+" sec_n_neighbors_lb="+params["sec_n_neighbors_lb"]+""
                " n_neighbors_weight_1="+params["n_neighbors_weight_1"]+" n_neighbors_weight_2="+params["n_neighbors_weight_2"]+""               
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "NN"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " learning_rate_ub="+params["learning_rate_ub"]+" learning_rate_lb="+params["learning_rate_lb"]+" batch_size_ub="+params["batch_size_ub"]+""
                " batch_size_lb="+params["batch_size_lb"]+" weight_decay_ub="+params["weight_decay_ub"]+" weight_decay_lb="+params["weight_decay_lb"]+""
                " beta_ub="+params["beta_ub"]+" beta_lb="+params["beta_lb"]+" hidden_units_1_ub="+params["hidden_units_1_ub"]+" hidden_units_1_lb="+params["hidden_units_1_lb"]+""
                " hidden_units_2_ub="+params["hidden_units_2_ub"]+" hidden_units_2_lb="+params["hidden_units_2_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")
 
            runscript  = open("slurm_run","w")
            runscript.write(header)
            runscript.write(script_text)
            runscript.close()
            #os.system("sbatch --array=16 slurm_run")
            os.system("sbatch --array=0-"+str(trials_num-1)+" slurm_run")
