import pandas as pd
import numpy as np
import pickle
import os
ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite"]

def filter_to_ligand_binding_domains(use_max_binding_score, features_all, features_cols):
    """
    Return groups of negatives only from domains that has positives for the ligand.
    
    @use_max_binding_score: if True, returning only absolute negatives. else: returning the ligand negatives.
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    
    return: dictionary of negatives by ligand
    """
    
    cwd = os.getcwd()
    #ligand binding domains dictionary
    with open(cwd+"/../ligands_negatives_domains_dict.pik", 'rb') as handle:
        negatives_dict = pickle.load(handle)
    
    ligands_negatives_df = {}
    for ligand in ligands:
        
        ligands_negatives_df[ligand] = pd.DataFrame()
        for domain in negatives_dict[ligand].keys():
            if domain == 'negatives' or domain == 'domains':
                continue
            domain_all = features_all.loc[features_all.loc[:,"domain_name"] == domain,:]
            
            #In case this domain was previously filtered
            if len(domain_all) == 0:
                continue
            
            if (use_max_binding_score):
                ligands_negatives_df[ligand] = pd.concat([ligands_negatives_df[ligand],domain_all.loc[domain_all.loc[:,"max_binding_score"] == 0,:]])
            else:
                ligand_bind_str = ligand+"_binding_score"
                ligands_negatives_df[ligand] = pd.concat([ligands_negatives_df[ligand],domain_all.loc[domain_all.loc[:,ligand_bind_str] == 0,:]])
        
    #Handeling the ligand "all_ligands"
    all_ligands_negatives_df = pd.concat([ligands_negatives_df["dna"], ligands_negatives_df["dnabase"], ligands_negatives_df["dnabackbone"], ligands_negatives_df["rna"], ligands_negatives_df["rnabase"], 
                                 ligands_negatives_df["rnabackbone"], ligands_negatives_df["ion"], ligands_negatives_df["peptide"], ligands_negatives_df["metabolite"]])
    all_ligands_negatives_df = all_ligands_negatives_df.drop_duplicates()
    #Filter to just positions with max. binding score = 0
    all_ligands_negatives_df = all_ligands_negatives_df[all_ligands_negatives_df["max_binding_score"] == 0]
    ligands_negatives_df["all_ligands"] = all_ligands_negatives_df
    
    #Leaving just the features columns
    for ligand in ligands_negatives_df.keys():   
        ligands_negatives_df[ligand] = ligands_negatives_df[ligand][features_cols]
        print(ligand+" non-binding #:"+str(len(ligands_negatives_df[ligand])))
    
    return ligands_negatives_df
#====================================================================================================================#

def negatives_by_binding_score(use_max_binding_score, filter_max_score_zero, features_all, features_cols):
    """
    Return groups of negatives according to ligand binding score.
    
    @use_max_binding_score: if True, returning only absolute negatives. else: returning the ligand negatives.
    @filter_max_score_zero: if True, remove the absolute negatives.
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    
    return: dictionary of negatives by ligand
    """
    
    ligands_negatives_df = {}
    for ligand in ligands:
        
        if use_max_binding_score:
            ligand_bind_str = "max_binding_score"
        else:
            ligand_bind_str = ligand+"_binding_score"
        
        if (filter_max_score_zero):
            ligands_negatives_df[ligand] = features_all[features_all[ligand_bind_str] == 0][features_all["max_binding_score"] != 0]
        else:
            ligands_negatives_df[ligand] = features_all[features_all[ligand_bind_str] == 0]
        ligands_negatives_df[ligand] = ligands_negatives_df[ligand].loc[:,features_cols]
        print(ligand+" non-binding #:"+str(len(ligands_negatives_df[ligand])))
        
    #Handeling the ligand "all_ligands"
    ligands_negatives_df["all_ligands"] = features_all[features_all["max_binding_score"] == 0]
    ligands_negatives_df["all_ligands"] = ligands_negatives_df["all_ligands"].loc[:,features_cols]
    print("all_ligands non-binding #:"+str(len(ligands_negatives_df["all_ligands"])))
    
    return ligands_negatives_df
#====================================================================================================================#

def create_negatives_datasets(FILTER_DOMAIN, ABSOLUTE_NEGATIVES, FILTER_MAX_SCORE_ZERO, features_all, features_cols):
    """
    Create the ligands negatives datasets according to a few boolean flags.
    
    @FILTER_DOMAIN: filter to only domains that contain positives of the ligand
    @ABSOLUTE_NEGATIVES: filter to just absolute negatives
    @FILTER_MAX_SCORE_ZERO: remove the absolute negatives
    @features_all: the table of all the positions and their features
    @features_cols: column names of the features.
    
    return: dictionary of negatives by ligand
    """
                              
    if FILTER_DOMAIN:
        if ABSOLUTE_NEGATIVES:
            ligands_negatives_df = filter_to_ligand_binding_domains(True, features_all, features_cols)
        else:
            ligands_negatives_df = filter_to_ligand_binding_domains(False, features_all, features_cols)
    else:
        if ABSOLUTE_NEGATIVES:
            ligands_negatives_df = negatives_by_binding_score(True, FILTER_MAX_SCORE_ZERO, features_all, features_cols)
        else:
            ligands_negatives_df = negatives_by_binding_score(False, FILTER_MAX_SCORE_ZERO, features_all, features_cols)
    
    return ligands_negatives_df
#====================================================================================================================#

def create_positives_datasets(bind_th, features_all, features_cols):
    
    ligands_features_df = {}
    
    for ligand in ligands:
        score_col_str = ligand+"_binding_score"
        ligand_binding_df = features_all[features_all[score_col_str] >= bind_th]
        print ligand+" #: "+str(ligand_binding_df.shape[0])
        ligands_features_df[ligand] = ligand_binding_df.loc[:,features_cols]
    
    return ligands_features_df
#====================================================================================================================#
