import time
from subprocess import call
import datetime

ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "all_ligands"]

curr_dt = datetime.datetime.now().strftime("%m-%d-%y_%H-%M-%S")

#ligands = ["dnabase"]

for ligand in ligands:
            header ="""#!/bin/bash
#SBATCH --mem=204800
#SBATCH --qos=1day
#SBATCH --job-name={0}
#SBATCH --mail-user=rsharan@princeton.edu
#SBATCH --mail-type=end,fail,time_limit
#SBATCH -n 1 --cpus-per-task=20\n\n""".format(ligand)
            folder_stuff = "CURRDATE={0}\nmkdir reports/reports_{0}\n".format(curr_dt,curr_dt)
            script_text = "cat classification_generic-CV-XGB-Hyperparameter-Tuning-Ran_by_script_threaded.ipynb | ligand="+ligand + " runipy --stdout > reports/reports_"+ curr_dt +"/"+ligand+".ipynb"

            runscript  = open("slurm_run","w")  
            runscript.write(header)
            runscript.write(folder_stuff)
            runscript.write(script_text)
            runscript.close()
            call(["sbatch","slurm_run"])

