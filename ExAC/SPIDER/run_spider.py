from subprocess import call
import pickle

# Directories
files_dir = '/Genomics/grid/users/dtodd/SPIDER2'

SEQ_PER_JOB = 35

# Load genes to aa sequence mapping for each domain
with open(files_dir+"/sequence_dict.pik", 'rb') as handle:
    sequence_dict = pickle.load(handle)

# Break up jobs to be no larger than 1 gene
i = 0
while i < len(sequence_dict.keys()):
  hi = min(i+SEQ_PER_JOB,len(sequence_dict.keys()))

  # Set memory and time params
  header ="""#!/bin/bash
#SBATCH --mem=40480
#SBATCH --qos=1day
#SBATCH --job-name=i{0}_{1}
#SBATCH --mail-user=dtodd@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(i,hi)

  script_text = "python run_job.py "+str(i)+" "+str(hi)

  runscript  = open("i{0}_{1}".format(i,hi),"w")
  runscript.write(header)
  runscript.write(script_text)
  runscript.close()
  call(["sbatch","i{0}_{1}".format(i,hi)])

  i += SEQ_PER_JOB
