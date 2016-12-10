library(ggplot2)
library(grid)
library(gridExtra)

data_path = "/home/anat/Research/ExAC/9.Features_exploration/"
filename100 = "domains_features_df_filtered100.csv"
filename50 = "domains_features_df_filtered50.csv"

domains_table <- read.csv(paste0(data_path, filename50), header = TRUE, sep = '\t', row.names = 1)

#Plot average MAF Vs. number of alterations in DNA level

#===Creating the plot===#
#Size transformation for the size scale
size_trans <- cut(domains_table[,"alter_num_dna"], breaks=c(100,1000,10000,100000), labels=FALSE)
domains_table <- cbind(domains_table, size_trans)

#The last break for the colors scale
max_break <- floor(max(domains_table[,"num_genes_log2"]))

#Colors for the color gradient guide
colfunc<-colorRampPalette(c("dark blue", "blue", "cyan", "green", "yellow", "orange","orangered" ,"red", "dark red"))

#Fraction of alterations Vs. fraction of polymorphic sites (sites with more than one alteration)
ggplot(domains_table, aes(x=frac_alter, y=frac_poly_several, size=size_trans, color=num_genes_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of genes", "in domain family)")),colours = colfunc(50), guide = "colourbar", breaks=0:max_break) +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  scale_x_continuous("Fraction of sites with alteration") +
  scale_y_continuous("Fraction of polymorphic sites (out of all altered)") +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of\nmutations in\ndomain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Fraction of alterations Vs. fraction of rare altered sites
ggplot(domains_table, aes(x=frac_alter, y=rare_poly_0.05., size=size_trans, color=num_genes_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of genes", "in domain family)")),colours = colfunc(50), guide = "colourbar", breaks=0:max_break) +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  scale_x_continuous("Fraction of sites with alteration") +
  scale_y_continuous("Fraction of rare ploymorphic sites <0.05% (out of all altered)") +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of\nmutations in\ndomain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())

#Fraction of rare altered sites Vs. fraction of polymorphic sites (sites with more than one alteration)
ggplot(domains_table, aes(x=rare_poly_0.05., y=frac_poly_several, size=size_trans, color=num_genes_log2)) +
  geom_point(alpha=0.5, na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of genes", "in domain family)")),colours = colfunc(50), guide = "colourbar", breaks=0:max_break) +
  geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  scale_x_continuous("Fraction of rare ploymorphic sites <0.05% (out of all altered)") +
  scale_y_continuous("Fraction of polymorphic sites (out of all altered)") +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of\nmutations in\ndomain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
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
  scale_size("Number of\nmutations in\ndomain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())