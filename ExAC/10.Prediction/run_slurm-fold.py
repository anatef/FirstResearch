import time
from subprocess import call

#ligands = ["metabolite", "peptide", "ion", "dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone"]
#ligands_names = ["met", "pep", "ion", "dna", "dbs", "dbc", "rna", "rbs", "rbc"]
ligands = ["dna"]
ligands_names = ["dna"]

#classifiers = ["RF", "KNN", "SVM", "ADA", "Logistic"]
#classifiers = ["XGB", "ADA-RF", "RF", "SVM", "KNN", "Logistic"]
classifiers = ["XGB"]
classifiers_names = ["XB"]


#down_names = ["RUS", "ND", "IHT", "ENN", "RENN", "NCR", "Tomek", "CC", "NM3", "NM2", "NM1"]
#down_names = ["ND", "IHT", "ENN", "REN", "NCR", "TL", "RUS"]
down_names = ["ND"]


for j in range(len(classifiers)):
    classifier = classifiers[j]
    for i in range(len(ligands)):
        ligand = ligands[i]
        for k in range(1,11):
            fold = str(k)
    
            header ="""#!/bin/bash
#SBATCH --mem=40960
#SBATCH --qos=1day
#SBATCH --job-name={0}_{1}_{2}
#SBATCH --mail-user=anatf@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(classifiers_names[j],ligands_names[i], fold)

            script_text = "cat classification_CV-per-fold.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" runipy --stdout > reports/reports_per_fold/"+ligand+"_"+classifier+"_"+fold+"_10w.ipynb"

            runscript  = open("slurm_run","w") 
            runscript.write(header)
            runscript.write(script_text)
            runscript.close()
            call(["sbatch","slurm_run"])

