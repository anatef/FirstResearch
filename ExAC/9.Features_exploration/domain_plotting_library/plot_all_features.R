# Plots violin plots of different groups for the most interesting/non-correlated features
#
# thresh: Instance cutoff for domain inclusion
# groups: The vectors of GO terms (numbers or descriptions) corresponding to the two groups
# labels: Optionally replaces default group names
plot_all_features <- function(thresh,names,labels=list()) {
  # Get packages
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(Hmisc)
  
  # Get GO term file
  freq_counts <- read.csv(paste("freq_counts_expanded",toString(thresh),".csv",sep=""),row.names=1,stringsAsFactors=FALSE)
  
  # Get features file
  domains_table <- read.csv(paste("domains_features_df_filtered", toString(thresh), ".csv", sep=""), header = TRUE, sep = '\t', row.names = 1)
  
  # Check user input and throw appropriate errors
  check_input(table=domains_table,counts=freq_counts,groups=names,labels=labels)
  
  # Convert GO descriptions to numbers
  groups <- groups_to_numbers(names,freq_counts)

  # Restrict plot to relevant groups
  restrict_list <- restrict_groups(domains_table,groups,labels=labels)
  domains_table_agg <- restrict_list[[1]]
  counts <- restrict_list[[2]]
  
  # Define features and remove outliers
  vars <- c("avg_maf_nonsyn","sites_aa_alter_num","frac_alter_aa","frac_poly_several",
            "frac_rare_005","BLOSUM_avg","pseudo_dNdS","SIFT_avg","PolyPhen_avg",
            "entropy_nonsyn_pos","entropy_nonsyn_gene","phyloP_avg","clustering_90",
            "pfam_avg_max","charge_frac_negative","charge_frac_positive","net_charge",
            "func_frac_polar","func_frac_aromatic","func_frac_aliphatic","aa_volume_avg",
            "hydrophobicity_avg","flexibility_avg","hsa2_cn","frac_coil","frac_helix",
            "frac_sheet","asa")
  domains_table_agg <- remove_outliers(vars,domains_table_agg)
  
  # Format plots
  colors = c("red","blue","green","purple","orange")
  
  # Average MAF of non-synonomous mutations
  plot_list <- plot_feature_groups_violin(domains_table_agg,"avg_maf_nonsyn","Average MAF")
  gt1 <- plot_list[[1]]
  legend <- plot_list[[2]]
  
  # alter_num_aa
  gt2 <- plot_feature_groups_violin(domains_table_agg,"sites_aa_alter_num","Number Non-synonomous")[[1]]
  
  # frac_alter_aa
  gt3 <- plot_feature_groups_violin(domains_table_agg,"frac_alter_aa","Fraction Non-synonomous")[[1]]
  
  # frac_poly_several
  gt4 <- plot_feature_groups_violin(domains_table_agg,"frac_poly_several","Multiple Variants")[[1]]
  
  # rare_poly_0.005.
  gt5 <- plot_feature_groups_violin(domains_table_agg,"frac_rare_005","Rare SNP")[[1]]
  
  # BLOSUM_avg
  gt6 <- plot_feature_groups_violin(domains_table_agg,"BLOSUM_avg","Average BLOSUM62")[[1]]
  
  # pseudo_dNdS
  gt7 <- plot_feature_groups_violin(domains_table_agg,"pseudo_dNdS","Pseudo dNdS")[[1]]
  
  # SIFT
  gt8 <- plot_feature_groups_violin(domains_table_agg,"SIFT_avg","SIFT")[[1]]
  
  # PolyPhen
  gt9 <- plot_feature_groups_violin(domains_table_agg,"PolyPhen_avg","PolyPhen")[[1]]
  
  # entropy_pos_alter
  gt10 <- plot_feature_groups_violin(domains_table_agg,"entropy_nonsyn_pos","Entropy by Position")[[1]]
  
  # entropy_gene_alter
  gt11 <- plot_feature_groups_violin(domains_table_agg,"entropy_nonsyn_gene","Entropy by Gene")[[1]]
  
  # avg_phyloP
  gt12 <- plot_feature_groups_violin(domains_table_agg,"phyloP_avg","Average phyloP")[[1]]
  
  # Clustering
  gt13 <- plot_feature_groups_violin(domains_table_agg,"clustering_90","Clustering")[[1]]
  
  # pfam emissions prob
  gt14 <- plot_feature_groups_violin(domains_table_agg,"pfam_avg_max","pfam Emissions Probability")[[1]]
  
  # Table to display domain counts in each group
  table <- ftable(counts,size=0.9,xshift=0.3,yshift=0.3)
  
  lay <- rbind(c(1,2,3,4),
               c(5,6,7,8),
               c(9,10,11,15),
               c(12,13,14,16))
  
  grid.arrange(gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,gt9,gt10,gt11,gt12,gt13,gt14,legend,table,layout_matrix=lay)
  
  # frac_negative
  gt15 <- plot_feature_groups_violin(domains_table_agg,"charge_frac_negative","Fraction Negative AA")[[1]]
  
  # Fraction positive
  gt16 <- plot_feature_groups_violin(domains_table_agg,"charge_frac_positive","Fraction Positive AA")[[1]]
  
  # Net charge
  gt17 <- plot_feature_groups_violin(domains_table_agg,"net_charge","Net Charge")[[1]]
  
  # Fraction polar
  gt18 <- plot_feature_groups_violin(domains_table_agg,"func_frac_polar","Fraction Polar AA")[[1]]
  
  # Fraction aromatic
  gt19 <- plot_feature_groups_violin(domains_table_agg,"func_frac_aromatic","Fraction Aromatic AA")[[1]]
  
  # Fraction aliphatic
  gt20 <- plot_feature_groups_violin(domains_table_agg,"func_frac_aliphatic","Fraction Aliphatic AA")[[1]]
  
  # AA volume average
  gt21 <- plot_feature_groups_violin(domains_table_agg,"aa_volume_avg","Average Volume")[[1]]
  
  # Hydrophobicity average
  gt22 <- plot_feature_groups_violin(domains_table_agg,"hydrophobicity_avg","Average Hydrophobicity")[[1]]
  
  # Flexibility average
  gt23 <- plot_feature_groups_violin(domains_table_agg,"flexibility_avg","Average Flexibility")[[1]]
  
  # Average contact number
  gt24 <- plot_feature_groups_violin(domains_table_agg,"hsa2_cn","Average Contact Number")[[1]]
  
  # Fraction coil
  gt25 <- plot_feature_groups_violin(domains_table_agg,"frac_coil","Fraction Coil")[[1]]
  
  # Fraction helix
  gt26 <- plot_feature_groups_violin(domains_table_agg,"frac_helix","Fraction Helix")[[1]]
  
  # Fraction sheet
  gt27 <- plot_feature_groups_violin(domains_table_agg,"frac_sheet","Fraction Sheet")[[1]]
  
  # Average exposed surface area
  gt28 <- plot_feature_groups_violin(domains_table_agg,"asa","Average Available SA")[[1]]
  
  lay <- rbind(c(1,2,3,4),
               c(5,6,7,8),
               c(9,10,11,12),
               c(13,14,15,16))
  
  grid.arrange(gt15,gt16,gt17,gt18,gt19,gt20,gt21,gt22,gt23,gt24,gt25,gt26,gt27,gt28,legend,table,layout_matrix=lay)
}