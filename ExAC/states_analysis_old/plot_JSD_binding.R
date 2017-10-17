library(viridis)
library(ggplot2)
library(RColorBrewer)
library(Hmisc)
library(reshape)

data_path = "/home/anat/Research/ExAC/8.states_analysis/data-frames/avg_JSD/"

domain_name <- "zf-C2H2"

domain_jsd_avg <- read.csv(paste0(data_path, domain_name, "_avg_JSD_df.csv"), header = TRUE, sep = ',', row.names = 1)
domain_jsd_noncon <- read.csv(paste0(data_path, domain_name, "_noncon_avg_JSD_df.csv"), header = TRUE, sep = ',', row.names = 1)

mypal <- colorRampPalette( brewer.pal( 9 , "OrRd" ) )

ggplot(data=domain_jsd_avg, aes(x=factor(state), y=avg_JSD, fill=bind_score)) + 
  scale_fill_gradientn("Binding\nscore", colors = c(mypal(10)[4],mypal(10)[7:8],mypal(10)[10]), limits=c(0,1))+
  geom_bar(stat='identity') + 
  xlab("") +
  ylab("") +
  #ggtitle("ig domain") +
  theme_bw() +
  theme(
    axis.text.x=element_text(size=14),
    axis.text.y=element_text(size=14)
  ) +
  theme(axis.line = element_line(colour = "black"),
        #panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Plot sorted by value
ggplot(data=domain_jsd_noncon, aes(x=reorder(factor(state), avg_JSD), y=avg_JSD, fill=bind_score)) + 
  scale_fill_gradientn("Binding score", colors = c(mypal(10)[4],mypal(10)[8:10]), limits=c(0,1))+
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

domain_name <- "EGF_CA"

domain_jsd_avg <- read.csv(paste0(data_path, domain_name, "_avg_JSD_df.csv"), header = TRUE, sep = ',', row.names = 1)
domain_jsd_noncon <- read.csv(paste0(data_path, domain_name, "_noncon_avg_JSD_df.csv"), header = TRUE, sep = ',', row.names = 1)

#Compute JSD and binding scores correlation
p1_data <- cbind(domain_jsd_avg$avg_JSD, domain_jsd_avg$bind_score)

#Corrlations
p1_pearson <- rcorr(p1_data, type="pearson")
p1_spearman <- rcorr(p1_data, type="spearman")

p2_data <- cbind(domain_jsd_noncon$avg_JSD, domain_jsd_noncon$bind_score)

#Corrlations
p2_pearson <- rcorr(p2_data, type="pearson")
p2_spearman <- rcorr(p2_data, type="spearman")


#Trying to align them together
domain_jsd_avg_melted <- melt(domain_jsd_avg, id=c("state"))
colnames(domain_jsd_avg_melted) <- c("state", "type", "value")

ggplot(data=domain_jsd_avg_melted, aes(x=factor(state), y=value, fill=type)) +
  geom_bar(stat="identity", position=position_dodge(width = 0), width=0.76) +
  ylim(-1,1)