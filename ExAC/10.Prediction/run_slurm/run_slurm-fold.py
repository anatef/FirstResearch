import time
from subprocess import call

# ligands = ["metabolite", "peptide", "ion", "sm", "dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone"]
# ligands_names = ["met", "pep", "ion", "sm", "dna", "dbs", "dbc", "rna", "rbs", "rbc"]

# ligands = ["metabolite", "peptide", "ion", "sm", "dna", "rna"]
# ligands_names = ["met", "pep", "ion", "sm", "dna", "rna"]

# classifiers = ["XGB", "ADA", "SVM", "RF", "KNN", "Logistic"]
# classifiers_names = ["XB", "AD", "SV", "RF", "KN", "LG"]

#one run params
ligands = ["metabolite", "peptide", "ion", "sm", "rna"]
ligands_names = ["met", "pep", "ion", "sm", "rna"]
classifiers = ["KNN"]
classifiers_names = ["KN"]

for j in range(len(classifiers)):
    classifier = classifiers[j]
    for i in range(len(ligands)):
        ligand = ligands[i]
        for k in range(1,6):
            fold = str(k)
    
            header ="""#!/bin/bash
#SBATCH --mem=40960
#SBATCH --qos=1day
#SBATCH --job-name={0}_{1}_{2}
#SBATCH --mail-user=anatf@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(classifiers_names[j],ligands_names[i], fold)

            script_text = "cat classification_CV-per-fold.ipynb | ligand="+ligand+" fold="+fold+" classifier="+classifier+" runipy --stdout > reports/reports_per_fold/"+ligand+"_"+classifier+"_"+fold+"_combined.ipynb"

            runscript  = open("slurm_run","w") 
            runscript.write(header)
            runscript.write(script_text)
            runscript.close()
            call(["sbatch","slurm_run"])
    #time.sleep(30)
