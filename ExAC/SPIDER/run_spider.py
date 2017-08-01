from subprocess import call
import pickle

# Directories
files_dir = '/Genomics/grid/users/dtodd/SPIDER2'

# Load genes to aa sequence mapping for each domain
with open(files_dir+"/gene_dict.pik", 'rb') as handle:
    gene_dict = pickle.load(handle)

# Break up jobs to be no larger than 40 genes
for i in range(0,703):
    domain = gene_dict.keys()[i]
    j = 0
    while j < len(gene_dict[domain].keys()):
      hi = min(j+40,len(gene_dict[domain].keys()))

      # Set memory and time params
      header ="""#!/bin/bash
#SBATCH --mem=20480
#SBATCH --qos=1wk
#SBATCH --job-name=i{0}_{1}_{2}
#SBATCH --mail-user=dtodd@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(i,j,hi)

      script_text = "python spider2.py "+str(i)+" "+str(j)+" "+str(hi)

      runscript  = open("idx{0}_{1}_{2}".format(i,j,hi),"w")
      runscript.write(header)
      runscript.write(script_text)
      runscript.close()
      call(["sbatch","idx{0}_{1}_{2}".format(i,j,hi)])

      j += 40
