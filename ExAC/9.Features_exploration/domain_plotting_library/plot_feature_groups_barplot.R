# Draws a barplot for the feature and groups specified
#
# table: Features data frame with formed groups
# feature: The feature to be plotted
# label: Feature name to be used as the label for the x-axis
plot_feature_groups_barplot <- function(table,feature,label) {
  # Check groups have been formed
  if (!("Group") %in% names(table)) {
    stop("Groups must be made first")
  }
  
  # Get number of groups
  ngroups = length(unique(table$Group))
  
  # Colors for plot
  colors = c("red","blue","green","purple","orange")
  
  # Format plot
  p <- ggplot(table,aes(Group,table[[feature]],fill=Group)) +
    geom_col(color='black',alpha=0.5) +
    xlab(label) +
    scale_y_continuous(limits=c(0,1)) +
    scale_fill_manual(values=colors[1:ngroups]) +
    theme_bw() +
    theme(axis.line = element_line(color = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          #panel.border = element_blank(),
          panel.background = element_blank(),
          axis.title.y=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())
  
  # Extract legend
  legend <- get_legend(p)
  p <- p + theme(legend.position="none")
  gt <- ggplot_gtable(ggplot_build(p))
  return(list(gt,legend))
}