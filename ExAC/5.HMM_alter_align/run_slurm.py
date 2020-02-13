import time
from subprocess import call

#time.sleep(60*8)

for index in range(0,1):
		
    header ="""#!/bin/bash
#SBATCH --mem=20480
#SBATCH --qos=1day
#SBATCH --job-name=idx{0}
#SBATCH --mail-user=anatf@princeton.edu
#SBATCH --mail-type=fail,time_limit\n\n""".format(index)

    script_text = "cat alteration_to_hmm_state-parallele.ipynb | idx="+str(index)+" runipy --stdout > domains_states_dicts/pfam-v31/reports/idx"+str(index)+".ipynb"
    runscript  = open("slurm_run","w")  
    runscript.write(header)
    runscript.write(script_text)
    runscript.close()
    call(["sbatch","slurm_run"])

    #time.sleep(60)
