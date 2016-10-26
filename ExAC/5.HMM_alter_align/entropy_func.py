import math
from enum import Enum

#Amino acids constant
amino_acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S',
          'T','W','Y','V','*'] # 20 amino acids, and * for termination codon

def KLdiv(po, qo):
    """Compute the Kullback-Leibler divergence of two vectors, KL(p||q)"""
    assert len(po) == len(qo),            "Cannot compute KL divergence of vectors of unequal length."
    p = [float(i) for i in po]
    q = [float(i) for i in qo]
    
    return sum([p[x]*math.log(p[x]/q[x],2) for x in range(len(p))                if p[x] != 0. and p[x] != 0 and q[x] != 0. and q[x] != 0])
#-------------------------------------------------------------------------------------------#

def JSdiv(po, qo, wt=0.5):
    """Compute the Jensen-Shannon divergence of two vectors, JS(p||q)"""
    assert len(po) == len(qo),            "Cannot compute JS divergence of vectors of unequal length."
    p = [float(i) for i in po]
    q = [float(i) for i in qo]
    
    # Take weighted average of KL divergence
    av = [wt*p[x] + (1-wt)*q[x] for x in xrange(len(p))]
    return wt*KLdiv(p, av) + (1-wt)*KLdiv(q, av)
#-------------------------------------------------------------------------------------------#

def cons(residues, rand=False): 
    """Compute the Jensen-Shannon divergence of vector r with background
    frequency. Default is Blosum62 background frequencies, but can use 
    random if specified"""
        
    residues = [r for r in residues if r != '-'] # Remove gaps distribution
    aa = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S',
          'T','W','Y','V'] # 20 amino acids, not including gaps...
 
    if len(residues) < 2: return -1 # Too few seqs in this particular column
 
    # Background frequencies of amino acids (random or BLOSUM62 matrix): 
    q = [1./len(aa)]*len(aa) if rand else         [0.074, 0.052, 0.045, 0.054, 0.025, 0.034, 0.054, 0.074,
         0.026, 0.068, 0.099, 0.058, 0.025, 0.047, 0.039, 0.057,
         0.051, 0.013, 0.032, 0.073]
 
    fqs = [0.00001 + float(residues.count(s))/len(residues) for s in aa]
    freqs = [f/sum(fqs) for f in fqs]
    
    assert str(sum(q))=='1.0' and str(sum(freqs))=='1.0',            "Prob. vectors do not sum to 1"
 
    return JSdiv(freqs, q)
#-------------------------------------------------------------------------------------------#

#Constants to represent JSD background types
class JSD_background(Enum):
    PFAM_PROB = 2
    MAJOR_ALLELE = 1
    BLOSUM62 = 0

#-------------------------------------------------------------------------------------------#

def JSD(alterations_af_dict, aa_ref, maf, hmm_state, background):
    """
    Compute the Jensen-Shannon divergence of the aa frequencies, as described in the dictionary
    with background frequency. 
    """
    
    pfam_aa_order = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    if (background == JSD_background.BLOSUM62):
        #Background frequencies of amino acids (random or BLOSUM62 matrix): 
        q = [0.074, 0.052, 0.045, 0.054, 0.025, 0.034, 0.054, 0.074,
             0.026, 0.068, 0.099, 0.058, 0.025, 0.047, 0.039, 0.057,
             0.051, 0.013, 0.032, 0.073, 0]
        
    elif (background == JSD_background.MAJOR_ALLELE):
        #Create the background frequency vector according to the major allele
        q = [0.00001]*len(amino_acids)
        q[amino_acids.index(aa_ref)] += 1     
        
    elif (background == JSD_background.PFAM_PROB):
        #Get the Pfam HMM match emission prob. for this domain
        with open(curr_dir[0]+"/../2.parse_Pfam/zinc_match_state_prob.pik", 'rb') as handle:
            pfam_match_prob_dict = pickle.load(handle) 
        
        #Create the background frequency vector from the Pfam HMM prob.
        state_prob = pfam_match_prob_dict[hmm_state]
        q = []
        for aa in amino_acids:
            if (aa == "*"):
                q.append(0.00001)
            else:
                q.append(state_prob[pfam_aa_order.index(aa)])
        
    else:
        print "Wrong background input"
        return -1
    
    #Make sure it all sums to 1
    q = [f/sum(q) for f in q]
    
    #Create the frequency vector according to the alterations dictionary
    feqs_vector = []
    for aa in amino_acids:
        if (aa in alterations_af_dict.keys()):
            feqs_vector.append(0.00001 + sum(alterations_af_dict[aa]))
        elif (aa == aa_ref):
            feqs_vector.append(0.00001 + (1 - maf))
        else:
            feqs_vector.append(0.00001)
    
    p = [f/sum(feqs_vector) for f in feqs_vector]
    
    assert str(sum(q))=='1.0' and str(sum(p))=='1.0', "Prob. vectors do not sum to 1"
    
    return JSdiv(p, q)
#-------------------------------------------------------------------------------------------#

def SE(alterations_af_dict, aa_ref,  maf):
    """
    Calculate the Shannon Entropy based on the alterations af dictionary and retrieve the result.
    """
    
    #Create the frequency vector according to the alterations dictionary
    feqs_vector = []
    for aa in amino_acids:
        if (aa in alterations_af_dict.keys()):
            feqs_vector.append(0.00001 + sum(alterations_af_dict[aa]))
        elif (aa == aa_ref):
            feqs_vector.append(0.00001 + (1 - maf))
        else:
            feqs_vector.append(0.00001)
    
    p = [f/sum(feqs_vector) for f in feqs_vector]
    
    assert str(sum(p))=='1.0', "Prob. vector do not sum to 1"
    
    #Compute the Shannon Entropy of residues
    shannon_entropy = - sum(aa_prob*np.log2(aa_prob) for aa_prob in p)                  
    
    return shannon_entropy