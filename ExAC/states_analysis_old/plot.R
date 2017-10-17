install.packages("ggplot2")
install.packages("viridis")
install.packages("wesanderson")
install.packages("qqman")
library(viridis)
library(ggplot2)

library(scales)
library(wesanderson)
library(RColorBrewer)
library(plyr)
library(lattice)
library(qqman)
library(grid)
library(gridExtra)

data_path = "/home/anat/Research/ExAC/8.states_analysis/data-frames/avg_JSD/"
domain_names <- c("zf-C2H2", "SH3_1", "WW", "homeobox", "PUF")
zinc_emd_jsd_filename = "zf-C2H2_EMD_JSD_adj_df.csv"
zinc_emd_maf_filename = "zf-C2H2_EMD_MAF_adj_df.csv"
homeobox_emd_maf_filename = "homeobox_EMD_MAF_adj_df.csv"
ww_emd_maf_filename = "WW_EMD_MAF_adj_df.csv"


zinc_jsd_filename = "zf-C2H2_JSD_adj_chrom_pos_df.csv"
zinc_maf_filename = "zf-C2H2_MAF_adj_chrom_pos_df.csv"
sh3_maf_filename = "SH3_1_MAF_adj_chrom_pos_df.csv"
ww_maf_filename = "WW_MAF_adj_chrom_pos_df.csv"
homeobox_maf_filename = "homeobox_MAF_adj_chrom_pos_df.csv"
puf_maf_filename = "PUF_MAF_adj_chrom_pos_df.csv"

#maf_adj_scaled
ig_maf <- "ig_avg_JSD_df.csv"
ig_jsd_filename <- "ig_MAF_JSD_chrom_pos_df.csv"
ig_noncon <- "ig_noncon_avg_JSD_df.csv"


#===Import the data===#
#EMD
jsd_emd_adj_df <- read.csv(paste0(data_path, zinc_emd_jsd_filename), header = TRUE, sep = '\t', row.names = 1)
zinc_emd_maf_df <-  read.csv(paste0(data_path, zinc_emd_maf_filename), header = TRUE, sep = '\t', row.names = 1)
homeobox_emd_maf_df <- read.csv(paste0(data_path, homeobox_emd_maf_filename), header = TRUE, sep = '\t', row.names = 1)
ww_emd_maf_df <- read.csv(paste0(data_path, ww_emd_maf_filename), header = TRUE, sep = '\t', row.names = 1)

#JSD
jsd_adj_df <- read.csv(paste0(data_path, zinc_jsd_filename), header = TRUE, sep = '\t', row.names = 1)

#MAF dfs
zinc_maf_adj_df <- read.csv(paste0(data_path, zinc_maf_filename), header = TRUE, sep = '\t', row.names = 1)
sh3_maf_adj_df <- read.csv(paste0(data_path, sh3_maf_filename), header = TRUE, sep = '\t', row.names = 1)
ww_maf_adj_df <- read.csv(paste0(data_path, ww_maf_filename), header = TRUE, sep = '\t', row.names = 1)
homeobox_maf_adj_df <- read.csv(paste0(data_path, homeobox_maf_filename), header = TRUE, sep = '\t', row.names = 1)
puf_maf_adj_df <- read.csv(paste0(data_path, puf_maf_filename), header = TRUE, sep = '\t', row.names = 1)

ig_df <- read.csv(paste0(data_path, ig_maf), header = TRUE, sep = '\t', row.names = 1)
ig_df_non_scaled <- read.csv(paste0(data_path, ig_jsd_filename), header = TRUE, sep = '\t', row.names = 1)

#JSD dfs
domain_name <- "KRAB"

domain_jsd_avg <- read.csv(paste0(data_path, domain_name, "_avg_JSD_df.csv"), header = TRUE, sep = ',', row.names = 1)
domain_jsd_noncon <- read.csv(paste0(data_path, domain_name, "_noncon_avg_JSD_df.csv"), header = TRUE, sep = ',', row.names = 1)


#Rare poly
zinc_rare_poly_df <- read.csv(paste0(data_path, "rare_poly/","zf-C2H2_poly.csv"), header = TRUE, sep = '\t', row.names = 1)
homeobox_rare_poly_df <- read.csv(paste0(data_path, "rare_poly/","Homeobox_poly.csv"), header = TRUE, sep = '\t', row.names = 1)
sh3_rare_poly_df <- read.csv(paste0(data_path, "rare_poly/","SH3_1_poly.csv"), header = TRUE, sep = '\t', row.names = 1)
ww_rare_poly_df <- read.csv(paste0(data_path, "rare_poly/","WW_poly.csv"), header = TRUE, sep = '\t', row.names = 1)
puf_rare_poly_df <- read.csv(paste0(data_path, "rare_poly/","PUF_poly.csv"), header = TRUE, sep = '\t', row.names = 1)

#===Plot EMD===#
hlim <- max(ig_jsd_avg$avg_JSD)
JSD_rev <- vector(length = nrow(ig_jsd_avg))
i <- 1
for (val in ig_jsd_avg$avg_JSD){
  JSD_rev[i] <- (hlim - val)
  i <- i +1
}
ig_jsd_avg <- cbind(ig_jsd_avg, JSD_rev)


reverselog_trans <- function(base = exp(1)) {
  trans <- function(x) -log(x, base)
  inv <- function(x) base^(-x)
  trans_new(paste0("reverselog-", format(base)), trans, inv, 
            log_breaks(base = base), 
            domain = c(1e-100, Inf))
}

colfunc<-colorRampPalette(c("dark blue", "blue", "cyan", "green", "yellow", "orange", "orangered", "red"))
mypal <- colorRampPalette( brewer.pal( 5 , "YlGnBu" ) )

color.gradient <- function(x, colors=c("red","yellow","green"), colsteps=100) {
  return( colorRampPalette(colors) (colsteps) [ findInterval(x, seq(min(x),max(x), length.out=colsteps)) ] )
}

ggplot(data=ank_jsd_avg, aes(x=factor(state), y=avg_JSD, fill=avg_JSD)) + 
  scale_fill_gradientn("Average JSD", colors = rev(mypal(5)))+
  #scale_fill_viridis(option="magma") +
  geom_bar(stat='identity') + 
  xlab("") +
  ylab("") +
  #ggtitle("ig domain") +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        #panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Plot sorted by value
ggplot(data=ank_jsd_noncon, aes(x=reorder(factor(state), avg_JSD), y=avg_JSD, fill=avg_JSD)) + 
  scale_fill_gradientn("Average JSD", colors = rev(mypal(5)))+
  #scale_fill_viridis(option="magma") +
  geom_bar(stat='identity') + 
  xlab("") +
  ylab("") +
  ggtitle("") +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        #panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#===Plot JSD Adj disributions===#

jsd_adj_df <- ddply(jsd_adj_df, .(state), transform, ecd = ecdf(JSD)(JSD))

#CDF
ggplot(jsd_adj_df, aes(x = JSD, y = ecd, group = state, color = factor(state))) + 
  geom_line(size=0.25) +
  scale_fill_manual(values=colfunc(23)) +
  xlim(0,0.1) +
  ylim(0.985,1) +
  theme_bw()

#Jitter the data
stripplot(jsd_adj_df$JSD ~ jsd_adj_df$state, jitter=T)

#Side-by-side histograms
histogram(~ JSD | state, data=jsd_adj_df)

#Manhattan plot
jsd_adj_gwas <- data.frame(jsd_adj_df$state, jsd_adj_df$JSD, jsd_adj_df$chrom, jsd_adj_df$chrom_pos)
colnames(jsd_adj_gwas) <- c("CHR", "P", "SNP", "BP")
jsd_adj_gwas <- jsd_adj_gwas[jsd_adj_gwas$P >0,] #Remove all 0s (basically change the distribution)
manhattan(jsd_adj_gwas)


ggplot (jsd_adj_df, aes(x=JSD, group=state, color=factor(state))) +
  geom_histogram()


#===Plot MAF Adj disributions===#
histogram(~ MAF | state, data=maf_adj_df)

stripplot(maf_adj_df$MAF ~ maf_adj_df$state, jitter=T)

ggplot (maf_adj_df, aes(x=MAF, group=state, color=factor(state))) +
  geom_histogram()

maf_adj_df <- ddply(maf_adj_df, .(state), transform, ecd = ecdf(MAF)(MAF))

#CDF
ggplot(maf_adj_df, aes(x = MAF, y = ecd, group = state, color = factor(state))) + 
  geom_line(size=0.25) +
  scale_fill_manual(values=colfunc(23)) +
  xlim(0,0.1) +
  ylim(0.985,1) +
  theme_bw()


#Boxplot
mixed_colors = c("#89C5DA", "#DA5724", "#74D944", "#CE50CA", "#3F4921", "#C0717C", "#CBD588", "#5F7FC7", 
                 "#673770", "#D3D93E", "#38333E", "#508578", "#D7C1B1", "#689030", "#AD6F3B", "#CD9BCD", 
                 "#D14285", "#6DDE88", "#652926", "#7FDCC0", "#C84248", "#8569D5", "#5E738F", "#D1A33D", 
                 "#8A7C64", "#599861", "#89C5DA", "#DA5724", "#74D944", "#CE50CA", "#3F4921", "#C0717C", "#CBD588", "#5F7FC7",
                 "#89C5DA", "#DA5724", "#74D944", "#CE50CA", "#3F4921", "#C0717C", "#CBD588", "#5F7FC7",
                 "#673770", "#D3D93E", "#38333E", "#508578", "#D7C1B1", "#689030", "#AD6F3B", "#CD9BCD", 
                 "#D14285", "#6DDE88", "#652926", "#7FDCC0", "#C84248", "#8569D5", "#5E738F", "#D1A33D", 
                 "#8A7C64", "#599861", "#89C5DA", "#DA5724", "#74D944", "#CE50CA", "#3F4921", "#C0717C", "#CBD588", "#5F7FC7",
                 "#89C5DA", "#DA5724", "#74D944", "#CE50CA", "#3F4921", "#C0717C", "#CBD588", "#5F7FC7",
                 "#673770", "#D3D93E", "#38333E", "#508578", "#D7C1B1", "#689030", "#AD6F3B", "#CD9BCD", 
                 "#D14285", "#6DDE88", "#652926", "#7FDCC0", "#C84248", "#8569D5", "#5E738F", "#D1A33D", 
                 "#8A7C64", "#599861", "#89C5DA", "#DA5724", "#74D944", "#CE50CA", "#3F4921", "#C0717C", "#CBD588", "#5F7FC7")

ggplot (ig_df_non_scaled, aes(x=state, y=JSD, group = state, color = factor(state))) +
  geom_boxplot() +
  scale_color_manual(values=mixed_colors) +
  theme_bw()


#Rare polymorphism
curr_poly_df <- ww_rare_poly_df

p1 <- ggplot(data=curr_poly_df, aes(x=factor(state), y=X0.05, fill=X0.05)) + 
  scale_fill_gradientn("poly frac", colors = mypal(5), guide = FALSE) +
  #scale_fill_viridis(option="magma") +
  geom_bar(stat='identity') + 
  xlab("") +
  ylab("") +
  ggtitle("5%") +
  coord_cartesian(ylim = c(0.985, 1)) +
  theme_bw()
  
p2 <- ggplot(data=curr_poly_df, aes(x=factor(state), y=X0.005, fill=X0.005)) + 
  scale_fill_gradientn("poly frac", colors = mypal(5), guide = FALSE) +
  geom_bar(stat='identity') + 
  geom_text(aes(label=number), vjust=-0.25) + 
  xlab("") +
  ylab("") +
  ggtitle("zf-C2H2 0.5%") +
  coord_cartesian(ylim = c(0.95, 1)) +
  theme_bw()
 
p3 <- ggplot(data=curr_poly_df, aes(x=factor(state), y=X0.0005, fill=X0.0005)) + 
  scale_fill_gradientn("poly frac", colors = mypal(5), guide = FALSE) +
  geom_bar(stat='identity') + 
  xlab("") +
  ggtitle("0.05%") +
  coord_cartesian(ylim = c(0.9, 1)) +
  theme_bw()
  
p4 <- ggplot(data=curr_poly_df, aes(x=factor(state), y=X0.00005, fill=X0.00005)) + 
  scale_fill_gradientn("poly frac", colors = mypal(5), guide = FALSE) +
  geom_bar(stat='identity') + 
  geom_text(aes(label=number), vjust=-0.25) + 
  xlab("") +
  ylab("") +
  ggtitle("WW 0.005%") +
  theme(plot.title=element_text(family="Times", face="bold", size=8)) +
  coord_cartesian(ylim = c(0.4, 1)) +
  theme_bw()

grid.arrange(p1, p2, p3, p4, ncol = 2, top="zf-C2H2", bottom="state", left="Fraction of rare SNPs")
#multiplot(p1, p2, p3, p4, cols=2)


# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}
