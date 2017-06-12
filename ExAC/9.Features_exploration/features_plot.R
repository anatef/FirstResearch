library(ggplot2)
library(grid)
library(gridExtra)
library(Hmisc)

data_path <- "/home/anat/Research/ExAC/9.Features_exploration/"
filename100 <- "domains_features_df_filtered100.csv"
filename50 <- "domains_features_df_filtered50.csv"
filename10 <- "domains_features_df_filtered10.csv"

domains_table100 <- read.csv(paste0(data_path, filename100), header = TRUE, sep = '\t', row.names = 1)
domains_table50 <- read.csv(paste0(data_path, filename50), header = TRUE, sep = '\t', row.names = 1)
domains_table10 <- read.csv(paste0(data_path, filename10), header = TRUE, sep = '\t', row.names = 1)

#Plot average MAF Vs. number of alterations in DNA level

#===Creating the plot===#
#Size transformation for the size scale
size_trans <- cut(domains_table10[,"alter_num_dna"], breaks=c(100,1000,10000,100000), labels=FALSE)
domains_table10 <- cbind(domains_table10, size_trans)

size_trans <- cut(domains_table50[,"alter_num_dna"], breaks=c(100,1000,10000,100000), labels=FALSE)
domains_table50 <- cbind(domains_table50, size_trans)

size_trans <- cut(domains_table100[,"alter_num_dna"], breaks=c(100,1000,10000,100000), labels=FALSE)
domains_table100 <- cbind(domains_table100, size_trans)

#The last break for the colors scale
max_break10 <- floor(max(domains_table10[,"num_instances_log2"]))
max_break50 <- floor(max(domains_table50[,"num_instances_log2"]))
max_break100 <- floor(max(domains_table100[,"num_instances_log2"]))

#Colors for the color gradient guide
colfunc<-colorRampPalette(c("dark blue", "blue", "cyan", "green", "yellow", "orange","orangered" ,"red", "dark red"))

#Fraction of alterations Vs. average of alterations at each site
ggplot(domains_table10, aes(x=frac_alter, y=avg_poly, size=size_trans, color=num_instances_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of", "instances in domain)")),
                        limits = c(1,12), breaks=1:12, colours = colfunc(13), guide = "colourbar") +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of non-\nsyn alterations\n in domain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  xlab("Fraction of altered position") +
  ylab("Avgrage number of different alterations") +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Compute pearson and spearman
p1_data_10 <- cbind(domains_table10$frac_alter, domains_table10$avg_poly)
p1_data_50 <- cbind(domains_table50$frac_alter, domains_table50$avg_poly)
p1_data_100 <- cbind(domains_table100$frac_alter, domains_table100$avg_poly)

#Corrlations
p1_pearson_10 <- rcorr(p1_data_10, type="pearson")
p1_pearson_50 <- rcorr(p1_data_50, type="pearson")
p1_pearson_100 <- rcorr(p1_data_100, type="pearson")

p1_spearman_10 <- rcorr(p1_data_10, type="spearman")
p1_spearman_50 <- rcorr(p1_data_50, type="spearman")
p1_spearman_100 <- rcorr(p1_data_100, type="spearman")

#Fraction of alterations Vs. fraction of rare altered sites
ggplot(domains_table10, aes(x=frac_alter, y=rare_poly_0.05., size=size_trans, color=num_instances_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of", "instances in domain)")),
                        limits = c(1,12), breaks=1:12, colours = colfunc(13), guide = "colourbar") +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of non-\nsyn alterations\n in domain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Compute pearson and spearman
p2_data_10 <- cbind(domains_table10$frac_alter, domains_table10$rare_poly_0.05.)
p2_data_50 <- cbind(domains_table50$frac_alter, domains_table50$rare_poly_0.05.)
p2_data_100 <- cbind(domains_table100$frac_alter, domains_table100$rare_poly_0.05.)

#Corrlations
p2_pearson_10 <- rcorr(p2_data_10, type="pearson")
p2_pearson_50 <- rcorr(p2_data_50, type="pearson")
p2_pearson_100 <- rcorr(p2_data_100, type="pearson")

p2_spearman_10 <- rcorr(p2_data_10, type="spearman")
p2_spearman_50 <- rcorr(p2_data_50, type="spearman")
p2_spearman_100 <- rcorr(p2_data_100, type="spearman")

#Fraction of rare altered sites Vs. fraction of polymorphic sites (sites with more than one alteration)
ggplot(domains_table, aes(x=rare_poly_0.05., y=frac_poly_several, size=size_trans, color=num_genes_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of genes", "in domain family)")),colours = colfunc(50), guide = "colourbar", breaks=0:max_break) +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  scale_x_continuous("Fraction of rare ploymorphic sites <0.05% (out of all altered)") +
  scale_y_continuous("Fraction of polymorphic sites (out of all altered)") +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of\nnon-syn alterations in\ndomain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Average MAF (of the altered sites) Vs. number of aa alterations (normalized by length)
ggplot(domains_table, aes(x=avg_maf_altered, y=alter_num_aa_norm, size=size_trans, color=num_genes_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of genes", "in domain family)")),colours = colfunc(50), guide = "colourbar", breaks=0:max_break) +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of\nnon-syn alterations in\ndomain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())




#Num of sym mut Vs. domain length
ggplot(domains_table10, aes(x=length, y=syn_mut_num, size=size_trans, color=num_instances_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of", "instances in domain)")),
                        limits = c(1,12), breaks=1:12, colours = colfunc(13), guide = "colourbar") +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of non-\nsyn alterations\n in domain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  xlab("Domain length") +
  ylab("Number of synnonymous mutations") +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

p4_data_10 <- cbind(domains_table10$length, domains_table10$syn_mut_num)
p4_data_50 <- cbind(domains_table50$length, domains_table50$syn_mut_num)
p4_data_100 <- cbind(domains_table100$length, domains_table100$syn_mut_num)

#Corrlations
p4_pearson_10 <- rcorr(p4_data_10, type="pearson")
p4_pearson_50 <- rcorr(p4_data_50, type="pearson")
p4_pearson_100 <- rcorr(p4_data_100, type="pearson")

p4_spearman_10 <- rcorr(p4_data_10, type="spearman")
p4_spearman_50 <- rcorr(p4_data_50, type="spearman")
p4_spearman_100 <- rcorr(p4_data_100, type="spearman")

format.pval(p4_spearman_10$P, digits = 7)

pvalues:

#pn/ps Vs. BLOSUM Avg
ggplot(domains_table10, aes(x=pn.ps, y=BLOSUM_avg, size=size_trans, color=num_instances_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of", "instances in domain)")),
                        limits = c(1,12), breaks=1:12, colours = colfunc(13), guide = "colourbar") +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of non-\nsyn alterations\n in domain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

p3_data_10 <- cbind(domains_table10$pn.ps, domains_table10$BLOSUM_avg)
p3_data_50 <- cbind(domains_table50$pn.ps, domains_table50$BLOSUM_avg)
p3_data_100 <- cbind(domains_table100$pn.ps, domains_table100$BLOSUM_avg)

#Corrlations
p3_pearson_10 <- rcorr(p3_data_10, type="pearson")
p3_pearson_50 <- rcorr(p3_data_50, type="pearson")
p3_pearson_100 <- rcorr(p3_data_100, type="pearson")

p3_spearman_10 <- rcorr(p3_data_10, type="spearman")
p3_spearman_50 <- rcorr(p3_data_50, type="spearman")
p3_spearman_100 <- rcorr(p3_data_100, type="spearman")


#Average poly Vs. BLOSUM Avg
ggplot(domains_table10, aes(x=BLOSUM_avg, y=avg_poly, size=size_trans, color=num_instances_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of", "instances in domain)")),
                        limits = c(1,12), breaks=1:12, colours = colfunc(13), guide = "colourbar") +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of non-\nsyn alterations\n in domain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  xlab("Avergae BLOSUM62 score") +
  ylab("Avgrage number of different alterations") +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

p6_data_10 <- cbind(domains_table10$BLOSUM_avg, domains_table10$avg_poly)
p6_data_50 <- cbind(domains_table50$BLOSUM_avg, domains_table50$avg_poly)
p6_data_100 <- cbind(domains_table100$BLOSUM_avg, domains_table100$avg_poly)

#Corrlations
p6_pearson_10 <- rcorr(p6_data_10, type="pearson")
p6_pearson_50 <- rcorr(p6_data_50, type="pearson")
p6_pearson_100 <- rcorr(p6_data_100, type="pearson")

p6_spearman_10 <- rcorr(p6_data_10, type="spearman")
p6_spearman_50 <- rcorr(p6_data_50, type="spearman")
p6_spearman_100 <- rcorr(p6_data_100, type="spearman")

#Rare alter frac 0.005 Vs. BLOSUM Avg
ggplot(domains_table10, aes(x=BLOSUM_avg, y=rare_poly_0.005., size=size_trans, color=num_instances_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of", "instances in domain)")),
                        limits = c(1,12), breaks=1:12, colours = colfunc(13), guide = "colourbar") +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of non-\nsyn alterations\n in domain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  xlab("Avergae BLOSUM62 score") +
  ylab("Fraction of rare (<0.005%) alterations") +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

p7_data_10 <- cbind(domains_table10$BLOSUM_avg, domains_table10$rare_poly_0.005.)
p7_data_50 <- cbind(domains_table50$BLOSUM_avg, domains_table50$rare_poly_0.005.)
p7_data_100 <- cbind(domains_table100$BLOSUM_avg, domains_table100$rare_poly_0.005.)

#Corrlations
p7_pearson_10 <- rcorr(p7_data_10, type="pearson")
p7_pearson_50 <- rcorr(p7_data_50, type="pearson")
p7_pearson_100 <- rcorr(p7_data_100, type="pearson")

p7_spearman_10 <- rcorr(p7_data_10, type="spearman")
p7_spearman_50 <- rcorr(p7_data_50, type="spearman")
p7_spearman_100 <- rcorr(p7_data_100, type="spearman")

#Correlating everything together
all_data_10 <- as.matrix(domains_table10[1:19])
all_data_50 <- as.matrix(domains_table50[1:19])
all_data_100 <- as.matrix(domains_table100[1:19])

all_pearson_10 <- rcorr(all_data_10, type="pearson")
all_pearson_50 <- rcorr(all_data_50, type="pearson")
all_pearson_100 <- rcorr(all_data_100, type="pearson")

all_spearman_10 <- rcorr(all_data_10, type="spearman")
all_spearman_50 <- rcorr(all_data_50, type="spearman")
all_spearman_100 <- rcorr(all_data_100, type="spearman")