#Run HHalign version 2.0.15 (June 2012) like this: hhalign -i a1.hmm -t a2.hmm

import subprocess
import pandas as pd

header = "hhalign -i "
path = "../../2.parse_Pfam/v30/pfam_hmms_v30/*"
ext = ".hmm"
flag = " -t "

sim_pairs_df = pd.read_csv("domains_pairs_for_hhalign_filtered10.csv", sep=',', index_col=0)

scores = []
pvals = []

for index, pair in sim_pairs_df.iterrows():
    dom1 =  pair["sim_dom1"]
    dom2 =  pair["sim_dom2"]
    script = header+path+dom1+ext+flag+path+dom2+ext
    runscript = open("hhlign_local_run.sh", "w")
    runscript.write(script)
    runscript.close()
    query = subprocess.check_output("chmod 777 hhlign_local_run.sh", shell=True)
    query = subprocess.check_output("./hhlign_local_run.sh", shell=True)
    end_of_query = query[query.find("Score")+8:]
    score = end_of_query[:end_of_query.find(" ")]
    try:
        float(score)
    except: 
        print index
        print "score isn't a number"
    scores.append(score)
    pval = end_of_query[end_of_query.find("P-value")+10:end_of_query.find("\n")]
    pvals.append(pval)
    if (index % 100 == 0):
        print index
    if (index % 1000 == 0):
        #Saving tmp results
        scores_pvals_alone = pd.DataFrame({"scores": scores, "pvals": pvals})
        scores_pvals_alone.to_csv("scores_pvals_alone_"+str(index)+".csv", sep='\t')
    
sim_pairs_df["scores"] = scores
sim_pairs_df["p-values"] = pvals

#Save the file with all the results
sim_pairs_df.to_csv("domains_hhlign_scores.csv", sep='\t')
    