# Displays a scatter plot of two features with Pearson correlation and best fit line
#
# thresh: Domain number cutoff (10, 50, or 100)
# xvar: Feature to be plotted on the x-axis
# yvar: Feature to be plotted on the y-axis
test_plotting <- function(thresh, xvar, yvar) {
  # Get packages
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(Hmisc)
  
  # Get features file
  domains_table <- read.csv(paste("domains_features_df_filtered", toString(thresh), ".csv", sep=""), header = TRUE, sep = '\t', row.names = 1)
  
  # Check user input
  check_input(table=domains_table,features=c(xvar,yvar))
  
  # Remove outliers
  domains_table <- remove_outliers(c(xvar,yvar),domains_table)
  
  # Plot settings
  size_trans <- cut(domains_table[,"sites_snp_alter_num"], breaks=c(100,1000,10000,100000), labels=FALSE)
  domains_table <- cbind(domains_table, size_trans)
  
  # The last break for the colors scale
  max_break <- floor(max(domains_table[,"num_instances_log2"]))
  
  # Colors for the color gradient guide
  colfunc<-colorRampPalette(c("dark blue", "blue", "cyan", "green", "yellow", "orange","orangered" ,"red", "dark red"))
  
  # Positioning of text
  x_max <- max(domains_table[[xvar]],na.rm=TRUE)
  x_range <- x_max-min(domains_table[[xvar]],na.rm=TRUE)
  y_max <-max(domains_table[[yvar]],na.rm=TRUE)
  
  # Compute pearson and spearman correlations
  data <- cbind(domains_table[[xvar]], domains_table[[yvar]])
  pearson <- rcorr(data, type="pearson")
  spearman <- rcorr(data, type="spearman")

  # Format plot
  ggplot(domains_table, aes(x=domains_table[[xvar]], y=domains_table[[yvar]], size=size_trans, color=num_instances_log2)) +
    geom_point(alpha=0.5, na.rm = TRUE) +
    scale_color_gradientn(expression(atop(log[2]*"(number of", "instances in domain)")),
                          limits = c(1,12), breaks=1:12, colours = colfunc(13), guide = "colourbar") +
    geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4, na.rm=TRUE) +
    annotate("text", x = x_max-x_range*1.2/10, y = y_max, label = sprintf("Pearson Correlation:\n%.3f",pearson[[1]][1,2])) +
    theme(legend.title = element_text(vjust = 1)) +
    guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
    scale_size("Number of non-\nsyn alterations\n in domain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
    #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
    theme_bw() +
    xlab(xvar) +
    ylab(yvar) +
    theme(axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          #panel.border = element_blank(),
          panel.background = element_blank())
}