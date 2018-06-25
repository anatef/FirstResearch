#General functions and constants to use in 10.Prediction code files
import pandas as pd

ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "sm"]
score_cols_suffix = ["_propensity", "_prop_th_0.1", "_prop_th_0.25", "_prop_th_0.5", "_prop_th_0.75"]


def get_features_cols(features_all):
    """
    Returning a list of features column names
    """
    
    features_cols = features_all.columns.tolist()
    #removing binding scores and domain name
    for ligand in ligands:
        for suffix in score_cols_suffix:
            features_cols.remove(ligand+suffix)
    features_cols.remove("domain_name")
    
    return features_cols