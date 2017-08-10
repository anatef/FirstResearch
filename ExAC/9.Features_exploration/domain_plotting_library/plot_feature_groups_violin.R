# Plots a violin plot for one feature and different groups
#
# table: The features table
# feature: The feature of interest
# label: The feature name used as the x-axis label
plot_feature_groups_violin <- function(table,feature,label) {
  # Check groups have been formed
  if (!("Group") %in% names(table)) {
    stop("Groups must be made first")
  }
  
  # Get number of groups
  ngroups = length(unique(table$Group))
  
  # Colors for plot
  colors = c("red","blue","green","purple","orange")
  
  # Format plot
  p <- ggplot(table, aes(Group, table[[feature]], fill=Group)) +
    geom_violin(alpha = 0.5, position = 'identity', adjust=1/2) +
    theme_bw() +
    scale_fill_manual(values=colors[1:ngroups]) +
    ylab(label) +
    theme(axis.line = element_line(color = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          #panel.border = element_blank(),
          panel.background = element_blank(),
          axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank())
  
  # Extract legend
  legend <- get_legend(p)
  p <- p + theme(legend.position="none")
  gt <- ggplot_gtable(ggplot_build(p))
  
  return(list(gt,legend))
}