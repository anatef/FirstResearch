library(ggplot2)
library(ggpubr)
library(reshape)

data_path <- "/home/anat/Research/ExAC/9.Features_exploration/ortho-para_analysis/"
ortho_para_table <- read.csv(paste0(data_path, "positions_ortho-para_01.25.18.csv"), header = TRUE, sep = '\t', row.names = 1)

methods_col_names <- c("instances_individuals_change_ratio", "aa_ref_overlap_individuals_change_ratio", "jsd_100way_instances_major_ratio", "jsd_100way_aa_not_used_ratio", "jsd_mul_aa_ref_SE",
								"jsd_SE_diff_ratio", "SE_jsd_diff_ratio", "jsd_SE_sum", "jsds_ratio", "jsds_subtraction")

ligands <- c("dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "max")
non_binding <- subset(ortho_para_table, max_binding_score == 0)
non_binding["label"] <- "Neutral"

plots_list <- list()
methods_data <- list()
plots_idx <- 1
pv_list <- list()
pv_idx <- 1

for (i in 1:length(ligands)) {
	ligand <- ligands[[i]]
	ligand_str <- paste0(ligand,"_binding_score")
	
	col_names <- append(ligand_str, methods_col_names)
	ligand_binding <- ortho_para_table[ortho_para_table[[ligand_str]] >= 0.1, col_names]
	ligand_binding["label"] <- "Binding"
	
	col_names <- append(col_names, "label")
	ligand_non_binding <- non_binding[, col_names]
	
	combined_table <- rbind(ligand_binding, ligand_non_binding)
	melted_ids <-c(ligand_str, "label")
	combined_melted <- melt(combined_table, id.vars=melted_ids)
	
	methods_data[[i]] <- split(combined_melted, combined_melted$variable)
	for (j in 1: length(methods_data[[i]])) {
		method_table <- methods_data[[i]][[j]]
		method_name <- toString(method_table$variable[1])
		binding_num <- nrow(ligand_binding)
		
		w_test <- wilcox.test(ligand_binding[[method_name]], ligand_non_binding[[method_name]], alternative="greater")
		pv_list[[pv_idx]] <- signif(w_test$p.value,2)
		
		plots_list[[plots_idx]] <- ggplot(method_table, aes(x=label, y=value, group=label, fill = factor(label))) +
			geom_boxplot(outlier.colour="black", outlier.shape=16, outlier.size=0.5, notch=TRUE) +
			coord_cartesian(ylim = quantile(method_table$value, c(0, 0.9))) +
			#scale_y_log10() +
			ggtitle(paste0(ligand, ", ", method_name, "\n#(binding)=",binding_num, "\np=",pv_list[[pv_idx]])) +
			xlab("") +
			ylab("") +
			theme_bw() +
			theme(plot.title = element_text(hjust = 0.5, size=8), 
				  axis.title.y = element_text(size=8))
		plots_idx <- plots_idx + 1
		pv_idx <- pv_idx + 1
	}
}

#Plotting a grid for each method
#1)
instances_individuals_change_ratio_plots <- c(plots_list[1], plots_list[11], plots_list[21], plots_list[31], plots_list[41],
												 plots_list[51], plots_list[61], plots_list[71], plots_list[81], plots_list[91])
ggarrange(plotlist=instances_individuals_change_ratio_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#2)
aa_ref_overlap_individuals_change_ratio_plots <- c(plots_list[2], plots_list[12], plots_list[22], plots_list[32], plots_list[42],
											 plots_list[52], plots_list[62], plots_list[72], plots_list[82], plots_list[92])
ggarrange(plotlist=aa_ref_overlap_individuals_change_ratio_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#3)
jsd_100way_instances_major_ratio_plots <- c(plots_list[3], plots_list[13], plots_list[23], plots_list[33], plots_list[43],
											plots_list[53], plots_list[63], plots_list[73], plots_list[83], plots_list[93])
ggarrange(plotlist=jsd_100way_instances_major_ratio_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#4)
jsd_100way_aa_not_used_ratio_plots <- c(plots_list[4], plots_list[14], plots_list[24], plots_list[34], plots_list[44],
										plots_list[54], plots_list[64], plots_list[74], plots_list[84], plots_list[94])
ggarrange(plotlist=jsd_100way_aa_not_used_ratio_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#5)
jsd_mul_aa_ref_SE_plots <- c(plots_list[5], plots_list[15], plots_list[25], plots_list[35], plots_list[45],
							 plots_list[55], plots_list[65], plots_list[75], plots_list[85], plots_list[95])
ggarrange(plotlist=jsd_mul_aa_ref_SE_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#6)
jsd_SE_diff_ratio_plots <- c(plots_list[6], plots_list[16], plots_list[26], plots_list[36], plots_list[46],
							 plots_list[56], plots_list[66], plots_list[76], plots_list[86], plots_list[96])
ggarrange(plotlist=jsd_SE_diff_ratio_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#7)
jsd_SE_sum_plots <- c(plots_list[7], plots_list[17], plots_list[27], plots_list[37], plots_list[47],
					  plots_list[57], plots_list[67], plots_list[77], plots_list[87], plots_list[97])
ggarrange(plotlist=jsd_SE_sum_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#8)
SE_jsd_diff_ratio_plots <- c(plots_list[8], plots_list[18], plots_list[28], plots_list[38], plots_list[48],
							 plots_list[58], plots_list[68], plots_list[78], plots_list[88], plots_list[98])
ggarrange(plotlist=SE_jsd_diff_ratio_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#9)
jsds_ratio_plos <- c(plots_list[9], plots_list[19], plots_list[29], plots_list[39], plots_list[49],
					 plots_list[59], plots_list[69], plots_list[79], plots_list[89], plots_list[99])
ggarrange(plotlist=jsds_ratio_plos, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#10)
jsds_subtraction_plots <- c(plots_list[10], plots_list[20], plots_list[30], plots_list[40], plots_list[50],
							 plots_list[60], plots_list[70], plots_list[80], plots_list[90], plots_list[100])
ggarrange(plotlist=jsds_subtraction_plots, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")


#Best pvals
best_pval_for_each_ligand <- c(plots_list[4], plots_list[14], plots_list[24], plots_list[37], plots_list[47],
							   plots_list[57], plots_list[64], plots_list[74], plots_list[84], plots_list[94])
ggarrange(plotlist=best_pval_for_each_ligand, ncol=5, nrow=2, common.legend = TRUE, legend="bottom")

#draft
w_PolyPhen <- wilcox.test(neutral_table$PolyPhen, binding_table$PolyPhen, alternative="less")
pv_PolyPhen <- signif(w_PolyPhen$p.value,2)
p14 <- ggplot(no_strtuct_table10, aes(x=label, y=PolyPhen, group = label, fill = factor(label))) +
	geom_boxplot(outlier.colour="black", outlier.shape=16, outlier.size=0.5, notch=TRUE) +
	coord_cartesian(ylim = quantile(no_strtuct_table10$PolyPhen, c(0.1, 1))) +
	#scale_y_continuous(limits = quantile(states_table50$pseudo_dNdS, c(0.1, 0.9))) +
	scale_fill_manual(values=colorder) +
	guides(fill=FALSE) +
	ggtitle(paste0("PolyPhen Functional scores\n\np=",pv_PolyPhen)) +
	xlab("") +
	ylab("PolyPhen scores") +
	theme_bw() +
	theme(plot.title = element_text(hjust = 0.5, size=8), 
		  axis.title.y = element_text(size=8))