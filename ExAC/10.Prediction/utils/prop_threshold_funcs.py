import pandas as pd
import numpy as np
import pickle
import os
ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "sm"]


def create_negatives_datasets(zero_prop, no_prop, features_all, features_cols, all_ligands):
    """
    Return groups of negatives according to ligand propensity.
    
    @zero_prop: if True, include positions with propensity = 0
    @no_prop: if True, include positions with undefined propensity.
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    @all_ligands: if True, creates a table of postivies from all ligands
    
    return: dictionary of negatives by ligand
    """
    
    ligands_negatives_df = {}
    for ligand in ligands:
        
        score_col_str = ligand+"_propensity"
        
        if (zero_prop == True and no_prop == False):
            ligands_negatives_df[ligand] = features_all[features_all[score_col_str] == 0]
            
        elif (zero_prop == False and no_prop == True):
            ligands_negatives_df[ligand] = features_all[features_all[score_col_str] == -1]
        
        elif (zero_prop == True and no_prop == True):
            neg_zero_df = features_all[features_all[score_col_str] == 0]
            neg_und_df = features_all[features_all[score_col_str] == -1]
            ligands_negatives_df[ligand] = pd.concat([neg_zero_df, neg_und_df], join="outer")
        else:
            print "No Negatives group chosen. exiting(-1)"
            return -1
        
        print(ligand+" non-binding #:"+str(len(ligands_negatives_df[ligand])))
        
        #Update the negatives  table for this ligand with just the features
        ligands_negatives_df[ligand] = ligands_negatives_df[ligand].loc[:,features_cols]
        
    #Handeling the ligand "all_ligands"
    if (all_ligands):
        ligands_negatives_df["all_ligands"] = ligands_negatives_df[ligand]
        for ligand in ligands:
            ligands_negatives_df["all_ligands"] = pd.merge(ligands_negatives_df["all_ligands"], ligands_negatives_df[ligand], how="inner")
        print("all_ligands non binding#:"+str(len(ligands_negatives_df["all_ligands"])))
    
    return ligands_negatives_df
#====================================================================================================================#

def create_positives_datasets(prec_th, features_all, features_cols, all_ligands):
    """
    Create the ligands positives datasets.
    
    @prec_th: the InteracDome precision. This precisoon dtermines the per-domain propensity threshold
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    @all_ligands: if True, creates a table of postivies from all ligands
    
    return: dictionary of positives by ligand
    """
    
    ligands_positives_df = {}
    
    for ligand in ligands:
        
        score_col_str = ligand+"_propensity"
        prop_th_str = ligand+"_prop_th_"+str(prec_th)
        
        ligand_binding_df = features_all[features_all[prop_th_str] != -1][features_all[score_col_str] >= features_all[prop_th_str]]
        print ligand+" #: "+str(ligand_binding_df.shape[0])
        
        #Update the positives table for this ligand with just the features
        ligands_positives_df[ligand] = ligand_binding_df.loc[:,features_cols]
        
    #Handeling the ligand "all_ligands"
    if (all_ligands):
        ligands_positives_df["all_ligands"] = ligands_positives_df[ligand]
        for ligand in ligands:
            ligands_positives_df["all_ligands"] = pd.merge(ligands_positives_df["all_ligands"], ligands_positives_df[ligand], how='outer')

        print("all_ligands #:"+str(len(ligands_positives_df["all_ligands"])))
    
    return ligands_positives_df