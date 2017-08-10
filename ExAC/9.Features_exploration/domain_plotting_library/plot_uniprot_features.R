# Draws barplots for the fractions of domains in the groups specified with each uniprot annotation
#
# thresh: Instance cutoff for domains
# names: A list of groups of interest
plot_uniprot_features <- function(thresh,names) {
  # Get packages
  library(ggplot2)
  library(grid)
  library(gridExtra)
  
  # Get GO terms file
  freq_counts <- read.csv(paste("freq_counts_expanded",toString(thresh),".csv",sep=""),row.names=1,stringsAsFactors=FALSE)
  
  # Find features file
  domains_table <- read.csv(paste("domains_features_df_filtered", toString(thresh), ".csv", sep=""), header = TRUE, sep = '\t', row.names = 1)
  
  # Check user input
  check_input(table=domains_table,counts=freq_counts,groups=names)
  
  # Convert GO descriptions to numbers
  groups <- groups_to_numbers(names,freq_counts)
  
  # Set features
  features = c("disulfide","disulfide_interchain","binding","calcium",
               "metal","dna","nucleotide","crosslink")
  
  # Restrict plot to groups
  domains_table_agg <- restrict_groups(domains_table,groups)[[1]]
  
  # Compute fraction of domains with feature for each group
  ratios <- data.frame(matrix(ncol=length(features)+1,nrow=length(names)))
  colnames(ratios) <- c(features,"Group")
  row.names(ratios) <- names
  ratios$Group <- names
  for (var in features) {
    for (n in names) {
      capitalized = paste(toupper(substring(n, 1,1)), substring(n, 2),sep="")
      index <- domains_table_agg$Group == capitalized
      vals <- domains_table_agg[index,var]
      ratios[n,var] <- sum(vals) / length(vals)
      ratios[n,"Group"] <- capitalized
    }
  }
  
  # Format plots
  plot_list <- plot_feature_groups_barplot(ratios,"disulfide","Disulfide Bridge")
  gt1 <- plot_list[[1]]
  legend <- plot_list[[2]]
  gt2 <- plot_feature_groups_barplot(ratios,"disulfide_interchain","Interchain disulfide Bridge")[[1]]
  gt3 <- plot_feature_groups_barplot(ratios,"binding","Binding")[[1]]
  gt4 <- plot_feature_groups_barplot(ratios,"calcium","Calcium Binding")[[1]]
  gt5 <- plot_feature_groups_barplot(ratios,"metal","Metal Binding")[[1]]
  gt6 <- plot_feature_groups_barplot(ratios,"dna","DNA Binding")[[1]]
  gt7 <- plot_feature_groups_barplot(ratios,"nucleotide","Nucleotide Binding")[[1]]
  gt8 <- plot_feature_groups_barplot(ratios,"crosslink","Crosslink")[[1]]
  
  # Display in grid
  lay <- rbind(c(1,2,3),
               c(4,5,6),
               c(7,8,9))
  grid.arrange(gt1,gt2,gt3,gt4,gt5,gt6,gt7,gt8,legend,layout_matrix=lay,left="Fraction Domains")
}