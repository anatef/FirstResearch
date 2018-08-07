import pandas as pd
import numpy as np
import pickle
import os

ligands = ["dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "druglike", "sm"]


def create_negatives_datasets(zero_prop, no_prop, features_all, features_cols, all_ligands, verbose=True):
    """
    Return groups of negatives according to ligand propensity.
    
    @zero_prop: if True, include positions with propensity = 0
    @no_prop: if True, include positions with undefined propensity.
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    @all_ligands: if True, creates a table of postivies from all ligands
    @verbose: if True, print number of positions found for each ligand
    
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
        
        if (verbose): print(ligand+" non-binding #:"+str(len(ligands_negatives_df[ligand])))
        
        #Update the negatives  table for this ligand with just the features
        ligands_negatives_df[ligand] = ligands_negatives_df[ligand].loc[:,features_cols]
        
    #Handeling the ligand "all_ligands"
    if (all_ligands):
        ligands_negatives_df["all_ligands"] = ligands_negatives_df[ligand]
        for ligand in ligands:
            ligands_negatives_df["all_ligands"] = pd.merge(ligands_negatives_df["all_ligands"], ligands_negatives_df[ligand], how="inner")
        if (verbose): print("all_ligands non binding#:"+str(len(ligands_negatives_df["all_ligands"])))
    
    return ligands_negatives_df
#====================================================================================================================#

def create_positives_datasets(prec_th, features_all, features_cols, all_ligands, verbose=True):
    """
    Create the ligands positives datasets.
    
    @prec_th: the InteracDome precision. This precisoon dtermines the per-domain propensity threshold
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    @all_ligands: if True, creates a table of postivies from all ligands
    @verbose: if True, print number of positions found for each ligand
    
    return: dictionary of positives by ligand
    """
    
    ligands_positives_df = {}
    
    for ligand in ligands:
        
        score_col_str = ligand+"_propensity"
        prop_th_str = ligand+"_prop_th_"+str(prec_th)
        
        ligand_binding_df = features_all[features_all[prop_th_str] != -1][features_all[score_col_str] >= features_all[prop_th_str]]
        if (verbose): print ligand+" #: "+str(ligand_binding_df.shape[0])
        
        #Update the positives table for this ligand with just the features
        ligands_positives_df[ligand] = ligand_binding_df.loc[:,features_cols]
        
    #Handeling the ligand "all_ligands"
    if (all_ligands):
        ligands_positives_df["all_ligands"] = ligands_positives_df[ligand]
        for ligand in ligands:
            ligands_positives_df["all_ligands"] = pd.merge(ligands_positives_df["all_ligands"], ligands_positives_df[ligand], how='outer')

        if (verbose): print("all_ligands #:"+str(len(ligands_positives_df["all_ligands"])))
    
    return ligands_positives_df
#====================================================================================================================#

def create_negatives_datasets_combined(zero_prop, no_prop, features_all, features_cols, all_ligands, verbose=True):
    """
    Return groups of negatives according to ligand propensity. Also, combining all dna and rna to one label
    
    @zero_prop: if True, include positions with propensity = 0
    @no_prop: if True, include positions with undefined propensity.
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    @all_ligands: if True, creates a table of postivies from all ligands
    @verbose: if True, print number of positions found for each ligand
    
    return: dictionary of negatives by ligand
    """
    
    ligands_negatives_df = {}
    ligands_negatives_dna = {}
    ligands_negatives_rna = {}
    
    #Create the combined dna negatives
    for ligand in ["dna", "dnabase", "dnabackbone"]:
        
        score_col_str = ligand+"_propensity"
        
        if (zero_prop == True and no_prop == False):
            ligands_negatives_dna[ligand] = features_all[features_all[score_col_str] == 0]
            
        elif (zero_prop == False and no_prop == True):
            ligands_negatives_dna[ligand] = features_all[features_all[score_col_str] == -1]
        
        elif (zero_prop == True and no_prop == True):
            neg_zero_df = features_all[features_all[score_col_str] == 0]
            neg_und_df = features_all[features_all[score_col_str] == -1]
            ligands_negatives_dna[ligand] = pd.concat([neg_zero_df, neg_und_df], join="outer")
        else:
            print "No Negatives group chosen. exiting(-1)"
            return -1
        if (verbose): print(ligand+" non-binding #:"+str(len(ligands_negatives_dna[ligand])))
            
    #Combine all the dnas together
    all_dna_idx = set(ligands_negatives_dna["dna"].index) & set(ligands_negatives_dna["dnabase"].index) & set(ligands_negatives_dna["dnabackbone"].index)
    all_dna_idx_list = list(all_dna_idx)
    all_dna_idx_list.sort()
    ligands_negatives_df["dna"] = features_all.loc[all_dna_idx_list,features_cols]
    if (verbose): print "dna combined non binding #: "+str(ligands_negatives_df["dna"].shape[0])
        
    #Create the combined rna negatives
    for ligand in ["rna", "rnabase", "rnabackbone"]:
        
        score_col_str = ligand+"_propensity"
        
        if (zero_prop == True and no_prop == False):
            ligands_negatives_rna[ligand] = features_all[features_all[score_col_str] == 0]
            
        elif (zero_prop == False and no_prop == True):
            ligands_negatives_rna[ligand] = features_all[features_all[score_col_str] == -1]
        
        elif (zero_prop == True and no_prop == True):
            neg_zero_df = features_all[features_all[score_col_str] == 0]
            neg_und_df = features_all[features_all[score_col_str] == -1]
            ligands_negatives_rna[ligand] = pd.concat([neg_zero_df, neg_und_df], join="outer")
        else:
            print "No Negatives group chosen. exiting(-1)"
            return -1
        if (verbose): print(ligand+" non-binding #:"+str(len(ligands_negatives_rna[ligand])))
            
    #Combine all the rnas together
    all_rna_idx = set(ligands_negatives_rna["rna"].index) & set(ligands_negatives_rna["rnabase"].index) & set(ligands_negatives_rna["rnabackbone"].index)
    all_rna_idx_list = list(all_rna_idx)
    all_rna_idx_list.sort()
    ligands_negatives_df["rna"] = features_all.loc[all_rna_idx_list,features_cols]
    if (verbose): print "rna combined non binding #: "+str(ligands_negatives_df["rna"].shape[0])
    
    for ligand in ["peptide", "ion", "metabolite", "druglike", "sm"]:
        
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
        if (verbose): print(ligand+" non-binding #:"+str(len(ligands_negatives_df[ligand])))
        #Update the negatives  table for this ligand with just the features
        ligands_negatives_df[ligand] = ligands_negatives_df[ligand].loc[:,features_cols]
        
    #Handeling the ligand "all_ligands"
    if (all_ligands):
        ligands_negatives_df["all_ligands"] = ligands_negatives_df[ligand]
        for ligand in ligands:
            ligands_negatives_df["all_ligands"] = pd.merge(ligands_negatives_df["all_ligands"], ligands_negatives_df[ligand], how="inner")
        if (verbose): print("all_ligands non binding#:"+str(len(ligands_negatives_df["all_ligands"])))
    
    return ligands_negatives_df
#====================================================================================================================#

def create_positives_datasets_combined(features_all, features_cols, all_ligands, verbose=True):
    """
    Create the ligands positives datasets in a specific format:
    
    1. Combining all dna (0.5), dnabase (0.5), dnabackbone (0.5) to one label using logical or
    2. Combining all rna (0.5), rnabase (0.5), rnabackbone (0.75) to one label using logical or
    3. ion (0.75)
    4. peptide (0.5)
    5. metabolite (0.5)
    6. sm (0.5)
    
    @features_all: the table of all the positions and their features.
    @features_cols: column names of the features.
    @all_ligands: if True, creates a table of postivies from all ligands
    @verbose: if True, print number of positions found for each ligand
    
    return: dictionary of positives by ligand
    """
    
    ligands_prop_th = {"dna": 0.5,
                      "dnabase": 0.5,
                      "dnabackbone": 0.5,
                      "rna": 0.25,
                      "rnabase": 0.25,
                      "rnabackbone": 0.25,
                      "ion": 0.75,
                      "peptide": 0.5,
                      "metabolite": 0.5,
                      "druglike": 0.5,
                      "sm": 0.5}
    
    ligands_positives_df = {}
    ligands_positives_dna = {}
    ligands_positives_rna = {}
    
    #Create the combined dna positives
    for ligand in ["dna", "dnabase", "dnabackbone"]:

        score_col_str = ligand+"_propensity"
        prop_th_str = ligand+"_prop_th_"+str(ligands_prop_th[ligand])

        ligands_positives_dna[ligand] = features_all[features_all[prop_th_str] != -1][features_all[score_col_str] >= features_all[prop_th_str]]
        if (verbose): print ligand+" #: "+str(ligands_positives_dna[ligand].shape[0])

    #Combine all the dnas together
    all_dna_idx = set(ligands_positives_dna["dna"].index) | set(ligands_positives_dna["dnabase"].index) | set(ligands_positives_dna["dnabackbone"].index)
    all_dna_idx_list = list(all_dna_idx)
    all_dna_idx_list.sort()
    ligands_positives_df["dna"] = features_all.loc[all_dna_idx_list,features_cols]
    if (verbose): print "dna combined #: "+str(ligands_positives_df["dna"].shape[0])
        
    #Create the combined rna positives
    for ligand in ["rna", "rnabase", "rnabackbone"]:

        score_col_str = ligand+"_propensity"
        prop_th_str = ligand+"_prop_th_"+str(ligands_prop_th[ligand])

        ligands_positives_rna[ligand] = features_all[features_all[prop_th_str] != -1][features_all[score_col_str] >= features_all[prop_th_str]]
        if (verbose): print ligand+" #: "+str(ligands_positives_rna[ligand].shape[0])

    #Combine all the rna together
    all_rna_idx = set(ligands_positives_rna["rna"].index) | set(ligands_positives_rna["rnabase"].index) | set(ligands_positives_rna["rnabackbone"].index)
    all_rna_idx_list = list(all_rna_idx)
    all_rna_idx_list.sort()
    ligands_positives_df["rna"] = features_all.loc[all_rna_idx_list,features_cols]
    if (verbose): print "rna combined #: "+str(ligands_positives_df["rna"].shape[0])
    
    
    for ligand in ["peptide", "ion", "metabolite", "druglike", "sm"]:
        
        score_col_str = ligand+"_propensity"
        prop_th_str = ligand+"_prop_th_"+str(ligands_prop_th[ligand])
        
        ligand_binding_df = features_all[features_all[prop_th_str] != -1][features_all[score_col_str] >= features_all[prop_th_str]]
        if (verbose): print ligand+" #: "+str(ligand_binding_df.shape[0])
        
        #Update the positives table for this ligand with just the features
        ligands_positives_df[ligand] = ligand_binding_df.loc[:,features_cols]
    
    #Handeling the ligand "all_ligands"
    if (all_ligands):
        ligands_positives_df["all_ligands"] = ligands_positives_df[ligand]
        for ligand in ligands:
            ligands_positives_df["all_ligands"] = pd.merge(ligands_positives_df["all_ligands"], ligands_positives_df[ligand], how='outer')

        if (verbose): print("all_ligands #:"+str(len(ligands_positives_df["all_ligands"])))
    
    return ligands_positives_df