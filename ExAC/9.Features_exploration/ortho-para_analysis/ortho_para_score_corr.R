library(ggplot2)
library(reshape2)
library(ggpmisc)
library(plyr)
library(Hmisc)
library(grid)
library(gridExtra)

#Read input
data_path <- "/home/anat/Research/ExAC/9.Features_exploration/ortho-para_analysis/"
ortho_para_table <- read.csv(paste0(data_path, "positions_ortho-para_01.25.18.csv"), header = TRUE, sep = '\t', row.names = 1)

#Constants
para_measure = c("instances_change_frac", "aa_ref_overlap", "aa_ref_SE", "aa_ref_jsd")
ortho_measurments = c("med_jsd_100way_blosum", "avg_maf_all")
scores = c("dna_binding_score", "dnabase_binding_score", "dnabackbone_binding_score", "rna_binding_score", "rnabase_binding_score",
		   "rnabackbone_binding_score", "ion_binding_score", "peptide_binding_score", "metabolite_binding_score", "max_binding_score")

#Scale measurments to 0-1
ortho_para_scores_scaled <- ortho_para_table
ortho_para_scores_scaled["aa_ref_overlap"] <- as.vector(sapply(ortho_para_table["aa_ref_overlap"], function(x) x/20))
ortho_para_scores_scaled["aa_ref_SE"] <- as.vector(sapply(ortho_para_table["aa_ref_SE"], function(x) x/max(ortho_para_table["aa_ref_SE"])))

#Para table
para_table <- cbind(ortho_para_scores_scaled[para_measure], ortho_para_scores_scaled[scores])
para_melted <- melt(para_table, id=scores)
para_colors <- c("#d7191c", "#fdae61", "#abdda4", "#2b83ba")

#Ortho table
ortho_table <- cbind(ortho_para_scores_scaled[ortho_measurments], ortho_para_scores_scaled[scores])
ortho_melted <- melt(ortho_table, id=scores)
ortho_colors <- c("#d7191c", "#2b83ba")

#Para Corrlations
p1_spearman <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$instances_change_frac, method = "spearman")
p2_spearman <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$aa_ref_overlap, method = "spearman")
p3_spearman <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$aa_ref_SE, method = "spearman")
p4_spearman <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$aa_ref_jsd, method = "spearman")

p1_pearson <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$instances_change_frac, method = "pearson")
p2_pearson <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$aa_ref_overlap, method = "pearson")
p3_pearson <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$aa_ref_SE, method = "pearson")
p4_pearson <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$aa_ref_jsd, method = "pearson")

#Ortho Corrlations
p5_spearman <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$med_jsd_100way_blosum, method = "spearman")
p6_spearman <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$avg_maf_all, method = "spearman")
p7_spearman <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$avg_jsd_100way_blosum, method = "spearman")

p5_pearson <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$med_jsd_100way_blosum, method = "pearson")
p6_pearson <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$avg_maf_all, method = "pearson")
p7_pearson <- cor.test(ortho_para_table$max_binding_score, ortho_para_table$avg_jsd_100way_blosum, method = "pearson")

#Para scatter
ggplot(para_melted[sample(nrow(para_melted)),], aes(x=rnabackbone_binding_score, y=value, group=variable, color=variable)) +
	geom_point(shape=1, alpha=0.25) +
	theme_bw() +
	scale_color_manual("measurment" ,values=para_colors) +
	geom_smooth(method = "lm", se = FALSE)

#Ortho scatter
ggplot(ortho_melted[sample(nrow(ortho_melted)),], aes(x=metabolite_binding_score, y=value, group=variable, color=variable)) +
	geom_point(shape=1, alpha=0.25) +
	theme_bw() +
	scale_color_manual("measurment" ,values=ortho_colors) +
	geom_smooth(method = "lm", se = FALSE)

#Ortho histogram
ggplot(ortho_melted, aes(x=value, group=variable, color=variable)) +
	geom_density() +
	theme_bw() +
	scale_color_manual("measurment" ,values=ortho_colors) +
	coord_cartesian(ylim = c(0,10), xlim=c(0,1), expand = TRUE)

	