library(ggplot2)
library(reshape)

data_path = "/home/anat/Research/ExAC/9.Features_exploration/BayeScan_input/"
filename = "domains10_features_rm1.csv"

domains_positions <- read.csv(paste0(data_path, filename), header = TRUE, sep = '\t', row.names = 1)

domain_melt <- melt(domains_positions)
colnames(domain_melt) <- c("position_type", "value")

ggplot(domain_melt, aes(x=value, fill=position_type)) +
  geom_density(alpha=0.33) +
  #geom_histogram(binwidth=250, alpha=0.33, position="identity") +
  coord_cartesian(xlim = c(0, 10000)) +
  #scale_y_continuous(breaks=seq()) + 
  xlab("Number of positions") +
  ylab("Density") +
  ggtitle("Distribution of domains' total number of positions") +
  theme_bw()

#The histogram by number of domains
ggplot(domains_positions, aes(altered_positions)) +
  geom_histogram(binwidth=500) +
  stat_bin(binwidth=500, geom="text", aes(label=..count..), vjust=-1.5) +
  #scale_x_log10("Number of instances in human - log scaled", breaks=c(1,2,3,5,10,20,30,50,100,200,300,500,1000,2000,4000)) +
  coord_cartesian(xlim = c(0, 10000)) +
  #scale_y_continuous("Number of domains families") +
  ggtitle("Histogram of protein domains number of different instances") +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

ggplot(domains_positions, aes(non.altered_positions)) +
  geom_histogram(binwidth=500) +
  stat_bin(binwidth=500, geom="text", aes(label=..count..), vjust=-1.5) +
  #scale_x_log10("Number of instances in human - log scaled", breaks=c(1,2,3,5,10,20,30,50,100,200,300,500,1000,2000,4000)) +
  coord_cartesian(xlim = c(0, 15000)) +
  #scale_y_continuous("Number of domains families") +
  ggtitle("Histogram of protein domains number of different instances") +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Dsitribution of number of altered loci
ggplot(domains_positions, aes(non.altered_positions)) +
  geom_point() +
  theme_bw()