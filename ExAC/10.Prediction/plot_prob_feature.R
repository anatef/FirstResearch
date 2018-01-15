# Plots a stacked histogram of the fraction of positions with each value of a feature
# at each binding probability.
#
# filepath: The file path of the prediction table
# feature: The feature of interest

plot_prob_feature <- function(filepath,feature) {
  # Load table specified by filepath
  dat <- read.csv(filepath, header = TRUE, sep = '\t', row.names = 1)
  
  # Check input
  if (!(feature %in% colnames(dat))) {
    stop("Feature not found")
  }
  
  # Plot
  ggplot(dat) +
    geom_histogram(aes(x=prob,fill=factor(dat[[feature]])),position="fill",binwidth=0.025) +
    xlab("Probability") +
    ylab("Fraction in Bin") +
    scale_fill_discrete(name=feature) +
    theme(axis.line = element_line(colour="black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank())
  
}