import numpy as np
from pyemd import emd
from sklearn.metrics.pairwise import euclidean_distances

###EMD functions###

#Installed pymed-0.2.0 from: https://github.com/garydoranjr/pyemd

def compute_pymed_emd(dict_vals, bins_num=1000):
    """
    pymed function for computing EMD between a dictionary of distributions, and an all 0s distribution.
    Accepts a flat dictionary, each key points to a list of values comprising a distribution.
    compute the EMD for each key seperately and returns a dict of the EMD value for each key.
    """
    emd_dict = {}
    
    for state in dict_vals.keys():
        #Create the histograms
        (first_hist, first_bin_edges) = np.histogram(dict_vals[state], bins=bins_num, range=(0,1))
        second_hist = np.array([0] * bins_num)
        second_hist[0] = len(dict_vals[state])
        
        #Convert to float to fit the pymed function
        first_signature = np.array(dict_vals[state], dtype=float)
        second_signature = np.array([0] * len(dict_vals[state]), dtype=float)
        
        #Create a distance matrix and compute the EMD
        distance_matrix = euclidean_distances(first_signature.reshape(-1, 1), second_signature.reshape(-1, 1))
        emd_dict[state] = emd(first_signature, second_signature, distance_matrix)
        
    return emd_dict
#-------------------------------------------------------------------------------------------#

#Computing the EMD for one-dimensional array of bins:
#https://en.wikipedia.org/wiki/Earth_mover%27s_distance#Computing_the_EMD

def calc_emd_bins(first_dist, second_dist, bins_num):
    """
    A helper function for "compute_bins_emd" that compute the EMD between two distributions.
    Returns the EMD value
    """
    
    #Create a histogram for the two distributions
    (first_hist, first_bin_edges) = np.histogram(first_dist, bins=bins_num, range=(0,1))
    (second_hist, second_bin_edges) = np.histogram(second_dist, bins=bins_num, range=(0,1))
    
    #Compute EMD
    total_distance = 0
    last_dirt = 0
    for i in range(bins_num):
        new_dirt = (first_hist[i] + last_dirt) - second_hist[i]
        last_dirt = new_dirt
        total_distance += abs(new_dirt)
    
    return total_distance
#-------------------------------------------------------------------------------------------#
def compute_bins_emd(dict_vals, bins_num=1000):
    """
    Computing EMD between a dictionary of distributions, and an all 0s distribution.
    EMD implemented here for one dimensional array of bins.
    Accepts a flat dictionary, each key points to a list of values comprising a distribution.
    compute the EMD for each key seperately and returns a dict of the EMD value for each key.
    """

    bins_emd_dict = {}

    for state in states_af_dict.keys():
        bins_emd_dict[state] = calc_emd_bins(states_af_dict[state], [0] * len(states_af_dict[state]), bins_num)
        
    return bins_emd_dict
