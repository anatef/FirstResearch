import time
from subprocess import call

#ligands = ["all_ligands", "metabolite", "peptide", "ion", "dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone"]
ligands = ["metabolite", "peptide", "ion", "dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone"]
#ligands = ["dnabase"]
#classifiers = ["RF", "KNN", "SVM", "ADA", "Logistic"]
#classifiers = ["XGB", "ADA-RF", "RF", "SVM", "KNN", "Logistic"]
classifiers = ["XGB"]

#down_methods = ["RandomUnderSampler", "NoDown", "InstanceHardnessThreshold", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "NeighbourhoodCleaningRule", "TomekLinks", "ClusterCentroids", "NearMiss3", "NearMiss2", "NearMiss1"]
#down_methods = ["NoDown", "InstanceHardnessThreshold", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "NeighbourhoodCleaningRule", "TomekLinks", "RandomUnderSampler"]
down_methods = ["NoDown"]
#down_names = ["RUS", "ND", "IHT", "ENN", "RENN", "NCR", "Tomek", "CC", "NM3", "NM2", "NM1"]
#down_names = ["ND", "IHT", "ENN", "REN", "NCR", "TL", "RUS"]
down_names = ["ND"]


for ligand in ligands:
    for i in range(len(down_methods)):
        down_method = down_methods[i]
        for classifier in classifiers:
    
            header ="""#!/bin/bash
#SBATCH --mem=40960
#SBATCH --qos=1day
#SBATCH --job-name=3{0}_{1}_{2}
#SBATCH --mail-user=anatf@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(ligand,down_names[i], classifier, )

            script_text = "cat classification_generic-CV-parallele.ipynb | ligand="+ligand+" down="+down_method+" classifier="+classifier+" runipy --stdout > reports/reports_windowed/"+ligand+"_"+down_method+"_"+classifier+"3w.ipynb"

            runscript  = open("slurm_run","w") 
            runscript.write(header)
            runscript.write(script_text)
            runscript.close()
            call(["sbatch","slurm_run"])

