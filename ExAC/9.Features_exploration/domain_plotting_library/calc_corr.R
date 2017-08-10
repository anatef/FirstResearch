# Calculates and plots correlations for each pair of features
#
# thresh: Domain instance cutoff (10, 50, or 100)
calc_corr <- function(thresh) {
  # Get libraries
  library(reshape2)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(Hmisc)
  
  # Get features file
  domains_table <- read.csv(paste("domains_features_df_filtered", toString(thresh), ".csv", sep=""), header = TRUE, sep = '\t', row.names = 1)
  
  # Store values in matrix
  len <- length(names(domains_table))
  pearson_vals <- matrix(0, len, len)
  spearman_vals <- matrix(0, len, len)

  for (i in 1:len) {
    for (j in i:len) {
      # Remove outliers greater than 3 SD from the mean
      x <- domains_table[[i]]
      y <- domains_table[[j]]
      x_sd <- sd(x,na.rm=TRUE)
      x_mean <- mean(x,na.rm=TRUE)
      y_sd <- sd(y,na.rm=TRUE)
      y_mean <- mean(y,na.rm=TRUE)
      xvar <- rep(NA,2000)
      yvar <- rep(NA,2000)
      counter <- 1
      for (k in 1:length(x)) {
        if (!is.na(x[k]) && x[k] < x_mean + 3*x_sd && x[k] > x_mean - 3*x_sd
            && !is.na(y[k]) && y[k] < y_mean + 3*y_sd && y[k] > y_mean - 3*y_sd) {
          xvar[counter] <- x[k]
          yvar[counter] <- y[k]
          counter <- counter + 1
        }
      }
      
      # Compute pearson and spearman
      data <- cbind(xvar[!is.na(xvar)], yvar[!is.na(yvar)])
      
      # Correlations
      pearson <- rcorr(data, type="pearson")
      spearman <- rcorr(data, type="spearman")
      
      # Save results
      pearson_vals[i,j] <- pearson[[1]][1,2]
      pearson_vals[j,i] <- pearson[[1]][1,2]
      spearman_vals[i,j] <- spearman[[1]][1,2]
      spearman_vals[j,i] <- spearman[[1]][1,2]
    }
  }
  
  # Format plot
  p <- ggplot(melt(pearson_vals), aes(Var1, Var2, fill=value)) + geom_raster() +
  scale_fill_gradientn(name="  Pearson\nCorrelation",colors=c("#0571b0","#92c5de","#f7f7f7","#f4a582","#ca0020"))
  for (i in 1:length(names(domains_table)))  {
    p <- p + annotation_custom(
      grob = textGrob(label = names(domains_table)[i], hjust = 0, gp = gpar(cex = 0.4), rot = -90),
      ymin = 0,
      ymax = 0,
      xmin = i,
      xmax = i)
    p <- p + annotation_custom(
      grob = textGrob(label = names(domains_table)[i], hjust = 1, gp = gpar(cex = 0.4)),
      ymin = i,
      ymax = i,
      xmin = 0,
      xmax = 0)
  }
  p <- p + theme(axis.title.x=element_blank(),
                 axis.text.x=element_blank(),
                 axis.ticks.x=element_blank(),
                 axis.title.y=element_blank(),
                 axis.text.y=element_blank(),
                 axis.ticks.y=element_blank(),
                 panel.grid.major = element_blank(),
                 panel.grid.minor = element_blank(),
                 panel.border = element_blank(),
                 panel.background = element_blank(),
                 plot.margin=unit(c(0.75,0.75,2.5,2.5), "cm"),
                 panel.spacing=unit(c(1,1,1,1), "cm"))
  
  # Turn off clipping
  gt <- ggplot_gtable(ggplot_build(p))
  gt$layout$clip[gt$layout$name == "panel"] <- "off"
  grid.draw(gt)
}