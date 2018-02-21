import time
from subprocess import call

ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "all_ligands"]
classifiers = ["RF", "KNN", "SVM", "ADA", "Logistic"]
down_methods = ["RandomUnderSampler", "NoDown", "ClusterCentroids", "NearMiss3", "NearMiss2", "NearMiss1", "TomekLinks", "EditedNearestNeighbours", "RepeatedEditedNearestNeighbours", "NeighbourhoodCleaningRule", "InstanceHardnessThreshold"]
down_names = ["RUS", "ND", "CC", "NM3", "NM2", "NM1", "Tomek", "ENN", "RENN", "NCR", "IHT"]

for ligand in ligands:
    for i in range(len(down_methods)):
        down_method = down_methods[i]
        for classifier in classifiers:
    
            header ="""#!/bin/bash
#SBATCH --mem=20480
#SBATCH --qos=1wk
#SBATCH --job-name={0}_{1}_{2}
#SBATCH --mail-user=anatf@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(down_names[i], classifier, ligand)

            script_text = "cat classification_generic-CV-parallele.ipynb | ligand="+ligand+" down="+down_method+" classifier="+classifier+" runipy --stdout > reports/"+ligand+"_"+down_method+"_"+classifier+".ipynb"

            runscript  = open("slurm_run","w")  
            runscript.write(header)
            runscript.write(script_text)
            runscript.close()
            call(["sbatch","slurm_run"])

