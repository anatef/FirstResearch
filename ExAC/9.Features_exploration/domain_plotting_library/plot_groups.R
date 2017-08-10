# Draws a density plot for a feature with the groups specified
#
# thresh: Instance cutoff for domain inclusion
# groups: The vectors of GO terms (numbers or descriptions) corresponding to the two groups
# xvar: The two features plotted
# xlabel: Optionally override default label for x-axis
plot_groups <- function(thresh,groups,xvar,xlabel="") {
  # Get packages
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(Hmisc)
  
  # Get GO terms file
  freq_counts <- read.csv(paste("freq_counts_expanded",toString(thresh),".csv",sep=""),row.names=1,stringsAsFactors=FALSE)
  
  # Get features file
  domains_table <- read.csv(paste("domains_features_df_filtered", toString(thresh), ".csv", sep=""), header = TRUE, sep = '\t', row.names = 1)
  
  # Check user input
  check_input(table=domains_table,counts=freq_counts,groups=groups,features=c(xvar))

  # Convert GO descriptions to numbers
  groups <- groups_to_numbers(groups,freq_counts)
  
  # Restrict plot to groups
  domains_table_agg <- restrict_groups(domains_table,groups)[[1]]
  
  # Remove outliers
  domains_table_agg <- remove_outliers(c(xvar),domains_table_agg)
  
  # Label for x-axis
  if (nchar(xlabel) == 0) {
    xlabel = paste(toupper(substring(xvar, 1,1)), substring(xvar, 2),sep="")
  }
  
  # Format plot
  colors = c("red","blue","green","purple","orange")
  ggplot(domains_table_agg, aes(domains_table_agg[[xvar]], fill=Group)) +
    geom_density(alpha = 0.5, position = 'identity', adjust=1/2) +
    theme_bw() +
    scale_fill_manual(values=colors[1:length(groups)]) +
    xlab(xlabel) +
    ylab("Domain Density") +
    theme(axis.line = element_line(color = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          #panel.border = element_blank(),
          panel.background = element_blank())
}