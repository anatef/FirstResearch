# Draws a scatterplot for two features with only domains in the groups specified
#
# thresh: Instance cutoff for domains
# groups: A list of groups of interest
# xvar: Feature to be plotted on x-axis
# yvar: Feature to be plotted on y-axis
plot_groups_features <- function(thresh,groups,xvar,yvar) {
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
  check_input(table=domains_table,counts=freq_counts,groups=groups,features=c(xvar,yvar))
  
  # Convert GO descriptions to numbers
  groups <- groups_to_numbers(groups,freq_counts)
  
  # Restrict plot to two groups
  domains_table_agg <- restrict_groups(domains_table,groups)[[1]]
  
  # Remove outliers
  domains_table_agg <- remove_outliers(c(xvar,yvar),domains_table_agg)
  
  # Compute pearson correlation
  data <- cbind(domains_table_agg[[xvar]], domains_table_agg[[yvar]])
  pearson <- rcorr(data, type="pearson")
  
  # Positioning of text
  x_max <- max(domains_table_agg[[xvar]],na.rm=TRUE)
  x_range <- x_max-min(domains_table_agg[[xvar]],na.rm=TRUE)
  y_max <-max(domains_table_agg[[yvar]],na.rm=TRUE)
  
  # Colors for the color gradient guide
  colfunc<-colorRampPalette( c("red","blue","green","purple","orange","yellow","brown","pink"))
  
  # Format plot
  ggplot(domains_table_agg, aes(x=domains_table_agg[[xvar]], y=domains_table_agg[[yvar]], color=Group)) +
    geom_point(alpha=0.5, na.rm = TRUE) +
    geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4, na.rm=TRUE) +
    annotate("text", x = x_max-x_range*1.2/10, y = y_max, label = sprintf("Pearson Correlation:\n%.3f",pearson[[1]][1,2])) +
    theme(legend.title = element_text(vjust = 1)) +
    theme_bw() +
    xlab(xvar) +
    ylab(yvar) +
    theme(axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          #panel.border = element_blank(),
          panel.background = element_blank())
}
