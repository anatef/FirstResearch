import time
import subprocess
import multiprocessing as mp


#one run params
ligands = ["dna"]
classifiers = ["KNN"]
heldout_folds = range(1,6)
test_folds = range(1,6)
       
def run_folds(classifier, ligand, heldout_fold, test_fold, sleep_flag, sleep_counter, NUM_CPUS):
    
    script_text = "ligand="+ligand+" heldout_fold="+str(heldout_fold)+" test_fold="+str(test_fold)+" classifier="+classifier+" runipy"
    
    if (sleep_flag):
        time.sleep((sleep_counter % NUM_CPUS)*30)
    print "starting "+ligand+" "+classifier+" heldout_fold = "+str(heldout_fold)+" test fold = "+str(test_fold)
    myinput = open("/home/ubuntu/ExAC/10.Prediction/stacking/stacking-1st-level-oof.ipynb", 'r')
    subprocess.call(script_text, stdin=myinput, shell=True)
            
#Create a list of all trials input params 
NUM_CPUS = 10
folds_list = []
sleep_counter = 0 #Sleep only at the beginning of pool
for classifier in classifiers:
    for ligand in ligands:
        for heldout_fold in heldout_folds:
            for test_fold in test_folds:
                if (test_fold == heldout_fold):
                    continue
                #Update the sleep flag just for the first NUM_CPUS trials
                if (sleep_counter <= NUM_CPUS):
                    sleep_flag = True
                else:
                    sleep_flag = False

                #Create the input tuple
                input_params = (classifier, ligand, heldout_fold, test_fold, sleep_flag, sleep_counter, NUM_CPUS)
                folds_list.append(input_params)
                sleep_counter += 1
                
#Call a pool of threads with all the requested trials                
pool = mp.Pool(processes = NUM_CPUS)
jobs = [pool.apply_async(run_folds, args = (params)) for params in folds_list]
results = [l.get() for l in jobs]
