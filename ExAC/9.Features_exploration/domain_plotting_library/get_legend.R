# Function found at https://github.com/hadley/ggplot2/wiki/Share-a-legend-between-two-ggplot2-graphs
# Extracts legend from a plot
#
# myggplot: The ggplot that contains the legend
get_legend <- function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}