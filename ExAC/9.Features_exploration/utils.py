import math
import numpy as np
from collections import defaultdict
from dnds_func import calculate_ns, seq_ns

# Calculates a normalized Shannon entropy (from Miller et al, 2015)
def entropy(a):
    a = float(np.asarray(a)) / sum(a)
    entropy = 0
    for val in a:
        if val == 0 or np.isnan(val):
            continue
        entropy += val * math.log(val)
    return(-entropy / math.log(len(a)))

# Measure density of vector with a sliding window
def density(a,window):
    norm = []
    for i in range(0,len(a)):
        norm.append(float(a[i])/sum(a))
    sums = np.empty(0)
    for i in range(0,len(a)+window-1):
        sums = np.append(sums,sum(norm[max(0,i-window+1):min(len(norm),i+1)]))
    return(entropy(sums))

# Calculate the dN/dS ratio
def calc_dNdS(ref_seq,Nd,Sd):
    (N,S) = seq_ns(ref_seq) #Reference expected syn/nonsyn per site
    if N == 0:
        PN = 0
    else:
        PN = Nd/float(N) #Proportion of nonsyn
    if S == 0:
        PS = 0
    else:
        PS = Sd/float(S) #Proportion of syn

    #num of nonsyn substitutions per syn site
    dN = -0.75 * (np.log(1-4*PN/float(3)))
    #num of syn substitutions per nonsyn site
    dS = -0.75 * (np.log(1-4*PS/float(3)))

    if dN == 0 or dS == 0:
        dN_dS = 1 #There isn't enough information to calculate dN/dS
    else:
        dN_dS = dN/dS

    return(dN_dS)

#Find the most recent file
def find_recent(files_list):
    recent_priority = -1
    recent_filename = ""
    for f in files_list:
        tokens = f.split("_")
        date = tokens[len(tokens)-1].split(".")
        month = int(date[0])
        day = int(date[1])
        #Not all files have years, but those that do are the most recent
        if date[2] != "pik":
            year = int(date[2])
        else:
            year = 0
        priority = year*1000 + month*50 + day
        if priority > recent_priority:
            recent_priority = priority
            recent_filename = f
    return(recent_filename)

#Creates a new default dict with given keys and values set at zero
def zeroes_dict(keys):
    ret = defaultdict(int)
    for key in keys:
        ret[key] = 0
    return(ret)
