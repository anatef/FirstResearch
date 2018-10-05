import time
from subprocess import call

#Params to run on
ligands = ["sm", "metabolite", "peptide", "ion", "dna", "rna"]
ligands_names = ["sm", "met", "pep", "ion", "dna", "rna"]
classifiers = ["RF"]
classifiers_names = ["RF"]

#Hyperparameters ranges
n_estimators_ub = "1500"
n_estimators_lb = "100"
max_depth_ub = "20"
max_depth_lb = "2"
min_samples_leaf_ub = "50"
min_samples_leaf_lb = "1"
min_samples_split_ub = "50:
min_samples_split_lb = "1"

#Looping over the jobs
for j in range(len(classifiers)):
    classifier = classifiers[j]
    for i in range(len(ligands)):
        ligand = ligands[i]
        for k in range(1,6):
            fold = str(k)
    
            header ="""#!/bin/bash
#SBATCH --mem=40960
#SBATCH --time=1day
#SBATCH --gres=gpu:1
#SBATCH --job-name={0}_{1}_{2}
#SBATCH --mail-user=anatf@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(classifiers_names[j],ligands_names[i], fold)
            script_text = "cat classification_CV-per-fold-Phase1.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+
            " n_estimators_ub="+n_estimators_ub+" n_estimators_lb= "+n_estimators_lb+
            " max_depth_ub="+max_depth_ub+" max_depth_lb="+max_depth_lb+
            " min_samples_leaf_ub="+min_samples_leaf_ub+" min_samples_leaf_lb="+min_samples_leaf_lb+
            " min_samples_split_ub="+min_samples_split_ub+" min_samples_split_lb="+min_samples_split_lb+
            " runipy --stdout > reports/"+ligand+"_"+classifier+"_"+fold+"_5w.ipynb"

            runscript  = open("slurm_run","w") 
            runscript.write(header)
            runscript.write(script_text)
            runscript.close()
            call(["sbatch","slurm_run"])
    #time.sleep(30)


