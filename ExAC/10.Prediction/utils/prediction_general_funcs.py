#General functions and constants to use in 10.Prediction code files
import pandas as pd

ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "druglike", "sm", "all"]
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
#====================================================================================================================#

def remove_unimportant_features(features_all, features_cols, additional_removal_features = []):
    """
    Removing features that aren't useful for the prediction.
    """
    
    GO_BEG1 = 425
    GO_END1 = 432
    
    #Remove domain id feature that was added just as a sanity check
    features_for_removal = ["domain_id"]
    
    #Remove GO terms as these features are missing got most domains and incomplete
    GO_features = features_cols[GO_BEG1:GO_END1]
    features_for_removal.extend(GO_features)
    
    #Removing also features from the input
    features_for_removal.extend(additional_removal_features)
    
    for feature in features_for_removal:
        del features_all[feature]   
    
    #Remove the features from the featues_cols list
    for feature in features_for_removal:
        features_cols.remove(feature)