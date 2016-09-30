
import math

def KLdiv(po, qo):
    """Compute the Kullback-Leibler divergence of two vectors, KL(p||q)"""
    assert len(po) == len(qo),            "Cannot compute KL divergence of vectors of unequal length."
    p = [float(i) for i in po]
    q = [float(i) for i in qo]
    
    return sum([p[x]*math.log(p[x]/q[x],2) for x in range(len(p))                if p[x] != 0. and p[x] != 0 and q[x] != 0. and q[x] != 0])


def JSdiv(po, qo, wt=0.5):
    """Compute the Jensen-Shannon divergence of two vectors, JS(p||q)"""
    assert len(po) == len(qo),            "Cannot compute JS divergence of vectors of unequal length."
    p = [float(i) for i in po]
    q = [float(i) for i in qo]
    
    # Take weighted average of KL divergence
    av = [wt*p[x] + (1-wt)*q[x] for x in xrange(len(p))]
    return wt*KLdiv(p, av) + (1-wt)*KLdiv(q, av)


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

