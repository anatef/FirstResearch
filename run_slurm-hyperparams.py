import time
import os

#Add: #SBATCH --gres=gpu:1 for running on GPU
#Add: SBATCH --time=00:00:00 for running on GPU
folds_num = 5
trials_num = 100

#Params to run on
# ligands = ["sm", "metabolite", "peptide", "ion", "dna", "rna"]
# ligands_names = ["sm", "met", "pep", "ion", "dna", "rna"]
# classifiers = ["XGB", "RF", "ADA", "SVM", "Logistic", "KNN"]
# classifiers_names = ["XB", "RF", "AD", "SV", "LG", "KN"]

#One run
ligands = ["sm"]
ligands_names = ["sm"]
classifiers = ["ADA"]
classifiers_names = ["AD"]

#Hyperparameters ranges dictionaries
hp_dict = dict()
hp_dict["XGB"] = {"max_depth_ub": "20",
                  "max_depth_lb": "1",
                  "min_child_weight_ub": "2",
                  "min_child_weight_lb": "0",
                  "colsample_bytree_ub": "1",
                  "colsample_bytree_lb": "0.25",
                  "gamma_ub": "0",
                  "gamma_lb": "-3",
                  "learning_rate_ub": "-0.5",
                  "learning_rate_lb": "-3"}

hp_dict["RF"] = {"n_estimators_ub": "1500",
                "n_estimators_lb": "100",
                "max_depth_ub": "20",
                "max_depth_lb": "2",
                "min_samples_leaf_ub": "50",
                "min_samples_leaf_lb": "1",
                "min_samples_split_ub": "50",
                "min_samples_split_lb": "1"}

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

hp_dict["KNN"] = {"n_neighbors_ub": "500",
                 "n_neighbors_lb": "10"}


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
                " min_child_weight_ub="+params["min_child_weight_ub"]+" max_depth_lb="+params["max_depth_lb"]+""
                " colsample_bytree_ub="+params["colsample_bytree_ub"]+" colsample_bytree_lb="+params["colsample_bytree_lb"]+""
                " gamma_ub="+params["gamma_ub"]+" gamma_lb="+params["gamma_lb"]+""
                " learning_rate_ub="+params["learning_rate_ub"]+" learning_rate_lb="+params["learning_rate_lb"]+""
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
                " C_ub="+params["C_ub"]+" C_lb="+params["C_lb"]+""
                " gamma_ub="+params["gamma_ub"]+" gamma_lb="+params["gamma_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "Logistic"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " C_ub="+params["C_ub"]+" C_lb="+params["C_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            elif (classifier == "KNN"):
                script_text = ("cat phase1_models_tuning.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" trial=${SLURM_ARRAY_TASK_ID}"
                " n_neighbors_ub="+params["n_neighbors_ub"]+" n_neighbors_lb="+params["n_neighbors_lb"]+""
                " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_${SLURM_ARRAY_TASK_ID}_5w.ipynb")

            runscript  = open("slurm_run","w")
            runscript.write(header)
            runscript.write(script_text)
            runscript.close()
            os.system("sbatch --array=0-"+str(trials_num-1)+" slurm_run")


