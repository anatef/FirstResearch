# Individual plots can be copy-pasted after reading in the needed files

# Get libraries
library(reshape2)
library(ggplot2)
library(grid)
library(gridExtra)
library(Hmisc)

# Domain instance cutoff
thresh = 50

# Get features file
domains_table <- read.csv(paste("domains_features_df_filtered", toString(thresh), ".csv", sep=""), header = TRUE, sep = '\t', row.names = 1)

# Load files
clustering_table <- read.csv("clustering_fixedThresh.csv", header = TRUE, sep = ',', row.names = 1)
clustering_percentile <- read.csv("clustering_percentile.csv", header = TRUE, sep = ',', row.names = 1)
clustering_binom <- read.csv("clustering_binom.csv", header = TRUE, sep = ',', row.names = 1)

## Format plots
# Clustering violin
ggplot(clustering_table) +
  geom_violin(aes(x=1,y=clustering_table$X0.05,fill="5%")) +
  geom_violin(aes(x=1,y=-clustering_table$X0.05.ratio,fill="5%")) +
  geom_violin(aes(x=2,y=clustering_table$X0.005,fill="0.5%")) +
  geom_violin(aes(x=2,y=-clustering_table$X0.005.ratio,fill="0.5%")) +
  geom_violin(aes(x=3,y=clustering_table$X0.0005,fill="0.05%")) +
  geom_violin(aes(x=3,y=-clustering_table$X0.0005.ratio,fill="0.05%")) +
  geom_violin(aes(x=4,y=clustering_table$X5e.05,fill="0.005%")) +
  geom_violin(aes(x=4,y=-clustering_table$X5e.05.ratio,fill="0.005%")) +
  scale_fill_manual(name="Threshold",values=c("#66c2a5","#fc8d62","#8da0cb","#e78ac3")) +
  ylab("<-- Fraction Positions Binned | Clustering Score -->              ") +
  scale_y_continuous(breaks=seq(-1,1,by=0.25),labels=abs(seq(-1,1,by=0.25)),limits=c(-1,1)) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.line = element_line(colour="black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

# Clustering score vs. fraction binned — fixed threshold
br <- c(seq(0,1,by=0.25),seq(1.25,2.25,by=0.25),seq(2.5,3.5,by=0.25),
        seq(3.75,4.75,by=0.25))
ggplot(clustering_table) +
  geom_point(aes(x=X0.05,y=X0.05.ratio,color="5%")) +
  geom_point(aes(x=1.25+X0.005,y=X0.005.ratio,color="0.5%")) +
  geom_point(aes(x=2.5+X0.0005,y=X0.0005.ratio,color="0.05%")) +
  geom_point(aes(x=3.75+X5e.05,y=X5e.05.ratio,color="0.005%")) +
  scale_color_manual(name="Threshold",values=c("#66c2a5","#fc8d62","#8da0cb","#e78ac3")) +
  xlab("Clustering Score") +
  ylab("Fraction Positions Binned") +
  scale_x_continuous(breaks=br,labels=rep(seq(0,1,by=0.25),4),limits=c(0,5)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

# Clustering score vs. fraction binned — percentile
br <- c(seq(0,1,by=0.25),seq(1.25,2.25,by=0.25),seq(2.5,3.5,by=0.25),
        seq(3.75,4.75,by=0.25))
ggplot(clustering_percentile) +
  geom_point(aes(x=X95,y=X95.ratio,color="95")) +
  geom_point(aes(x=1.25+X90,y=X90.ratio,color="90")) +
  geom_point(aes(x=2.5+X85,y=X85.ratio,color="85")) +
  geom_point(aes(x=3.75+X80,y=X80.ratio,color="80")) +
  scale_color_manual(name="Percentile",values=c("#66c2a5","#fc8d62","#8da0cb","#e78ac3")) +
  xlab("Clustering Score") +
  ylab("Fraction Positions Binned") +
  scale_x_continuous(breaks=br,labels=rep(seq(0,1,by=0.25),4),limits=c(0,5)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

# Clustering score vs. fraction binned — binomial cutoff
br <- c(seq(0,1,by=0.25),seq(1.25,2.25,by=0.25),seq(2.5,3.5,by=0.25),
        seq(3.75,4.75,by=0.25))
ggplot(clustering_binom) +
  geom_point(aes(x=X0.95,y=X0.95.ratio,color="0.05")) +
  geom_point(aes(x=1.25+X0.99,y=X0.99.ratio,color="0.01")) +
  geom_point(aes(x=2.5+X0.995,y=X0.995.ratio,color="0.005")) +
  geom_point(aes(x=3.75+X0.999,y=X0.999.ratio,color="0.001")) +
  scale_color_manual(name="Binomial Cutoff",values=c("#66c2a5","#fc8d62","#8da0cb","#e78ac3")) +
  scale_x_continuous(breaks=br,labels=rep(seq(0,1,by=0.25),4),limits=c(0,5)) +
  xlab("Clustering Score") +
  ylab("Fraction Positions Binned") +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

# Compare threshold and percentile
combined <- cbind(clustering_table,clustering_percentile)
ggplot(combined,aes(x=X5e.05,y=X85)) +
  geom_point() +
  xlab("0.005% Threshold") +
  ylab("85th percentile") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

# Histogram for pfam hmm probs
probs <- read.csv(paste0(data_path, "hmm_dict_probs.csv"), header = TRUE, sep = ',', row.names = 1)
ggplot(probs,aes(max)) +
  geom_histogram(bins=50) +
  xlab("Max probability") +
  ylab("Number of positions") +
  theme_bw() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())