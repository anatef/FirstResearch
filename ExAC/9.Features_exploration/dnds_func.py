import sys
import pickle
sys.path.append('/home/anat/Research/ExAC/5.HMM_alter_align') 
from calc_exac_freq_func import codon_table

#Read the substitutions table (for the dN/dS calculation)
with open("/home/anat/Research/ExAC/9.Features_exploration/codon_ns_table.pik", 'rb') as handle:
    codon_ns_table = pickle.load(handle)

def calculate_ns(codon):
    """Given a codon (string of len=3 of chars from {A,T,G,C}
    Calculate n = number of nonsynonymous sites possible for this codon,
    and s = number of synnonymous sites possible for this codon.
    n+s=3, since this is a site measurment"""
    
    ref_aa = codon_table[codon]
    bp1 = codon[:1]
    bp2 = codon[1:2]
    bp3 = codon[2:]
    
    syn = 0
    nonsyn = 0
    nucletoides = ["A", "T", "G", "C"]
    
    #Mutating bp1
    for n in nucletoides:
        if (bp1 == n):
            continue  
        alt_codon = n+bp2+bp3
        alt_aa = codon_table[alt_codon]
        if (alt_aa == ref_aa):
            syn += 1
        else:
            nonsyn += 1
    
    #Mutating bp2
    for n in nucletoides:
        if (bp2 == n):
            continue    
        alt_codon = bp1+n+bp3
        alt_aa = codon_table[alt_codon]
        if (alt_aa == ref_aa):
            syn += 1
        else:
            nonsyn += 1
            
    #Mutating bp3
    for n in nucletoides:
        if (bp3 == n):
            continue    
        alt_codon = bp1+bp2+n
        alt_aa = codon_table[alt_codon]
        if (alt_aa == ref_aa):
            syn += 1
        else:
            nonsyn += 1

    n = float('{:.5e}'.format(float(nonsyn/float(3))))
    s = float('{:.5e}'.format(float(syn/float(3))))

    return (n,s)
#========================================================================#

def seq_ns(sequence):
    """Given a sequence of nucletides that comprise full codons triplets
    calculate and return N = the total number of nonsynnonymous sites,
    and S = the total number of synnonymous sites
    """
    
    N = 0
    S = 0
    #af_i = 0
    
    for i in range(0,len(sequence),3):
        codon = sequence[i:i+3]
        N += codon_ns_table[codon]["N"]
        S += codon_ns_table[codon]["S"]
        #af_i += 1
        
    return (N,S)