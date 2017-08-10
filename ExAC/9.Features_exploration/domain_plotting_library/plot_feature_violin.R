# Draws a violin plot for the feature specified
#
# thresh: Instance cutoff for domains
# feature: The feature to be plotted
# label: Optionally override default label for the y-axis
plot_feature_violin <- function(thresh,feature,label="") {
  # Get feaatures file
  domains_table <- read.csv(paste("domains_features_df_filtered", toString(thresh), ".csv", sep=""), header = TRUE, sep = '\t', row.names = 1)
  
  # Check inputted columns
  check_input(table=domains_table,features=c(feature))
  
  # Label for y-axis
  if (nchar(label) == 0) {
    label = paste(toupper(substring(feature, 1,1)), substring(feature, 2),sep="")
  }
  
  # Format plot
  ggplot(domains_table, aes(x=1,y=domains_table[[feature]])) +
    geom_violin(fill="grey90") +
    theme_bw() +
    xlab("") +
    ylab(label) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          #panel.border = element_blank(),
          panel.background = element_blank())
}