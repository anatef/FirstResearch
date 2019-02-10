import pandas as pd
import numpy as np

def create_stacked_dataset(stacking_path, stacking_ligands, stacking_models, features_positives, features_negatives, all_models_list, keep_original_features=True):
    "Combine level-1 probs with features toe create a stacked dataset"
    
    positives_df_stacking_ligands = pd.DataFrame()
    negatives_df_stacking_ligands = pd.DataFrame()
    
    for stack_ligand in stacking_ligands:
    
        #Read the stacking-1st level probs of all the ligands
        staking1_filename = stack_ligand+"_stacking1_probs.csv"
        stacking1_ligand_fold_df = pd.read_csv(stacking_path+staking1_filename, sep='\t', index_col=0)
        stacking1_ligand_fold_df.index = stacking1_ligand_fold_df["idx"]

        #removing models not in "stacking_models"
        for model in all_models_list:
            if (model not in stacking_models):
                del stacking1_ligand_fold_df[model+"_"+stack_ligand+"_prob"]

        #Subsetting to the positive indices
        pos_idx = features_positives.index
        positives_df_ligand = stacking1_ligand_fold_df.loc[pos_idx]

        #Subsetting to the negative indices
        neg_idx = features_negatives.index
        negatives_df_ligand = stacking1_ligand_fold_df.loc[neg_idx]

        #Add to the combined df tables
        if (positives_df_stacking_ligands.shape[0] == 0):
            positives_df_stacking_ligands = positives_df_ligand
            negatives_df_stacking_ligands = negatives_df_ligand
        else:
            positives_df_stacking_ligands = pd.merge(positives_df_stacking_ligands, positives_df_ligand, on="idx")
            negatives_df_stacking_ligands = pd.merge(negatives_df_stacking_ligands, negatives_df_ligand, on="idx")

    positives_df_stacking_ligands.index = positives_df_stacking_ligands["idx"]
    del positives_df_stacking_ligands["idx"]
    negatives_df_stacking_ligands.index = negatives_df_stacking_ligands["idx"]
    del negatives_df_stacking_ligands["idx"]

    #Adding the original features
    if (keep_original_features):
        positives_df_stacking_ligands = pd.concat([positives_df_stacking_ligands, features_positives], axis=1)
        negatives_df_stacking_ligands = pd.concat([negatives_df_stacking_ligands, features_negatives], axis=1)
    
    print "#(features) = "+str(positives_df_stacking_ligands.shape[1])
    
    return (positives_df_stacking_ligands, negatives_df_stacking_ligands)