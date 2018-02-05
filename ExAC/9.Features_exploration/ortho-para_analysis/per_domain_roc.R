library(ROCR)
library(ggpubr)

data_path = "/home/anat/Research/ExAC/9.Features_exploration/ortho-para_analysis/"

positions_filename = "positions_ortho-para_01.25.18.csv"

positions_table <- read.csv(paste0(data_path, positions_filename), header = TRUE, sep = '\t', row.names = 1)

#Removing all conserved positions (pfam >= 0.5)
non_con_positions_table <- positions_table[positions_table["pfam_prob_max"] <= 0.5, ]


ligands = c("dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "max")
methods = c("instances_individuals_change_ratio", "aa_ref_overlap_individuals_change_ratio", "jsd_100way_instances_major_ratio", "jsd_100way_aa_not_used_ratio", "jsd_mul_aa_ref_SE", "jsd_SE_diff_ratio", "SE_jsd_diff_ratio", "jsd_SE_sum", "jsds_ratio", "jsds_subtraction")
domains_names <- unique(non_con_positions_table["domain_name"])
domains_names <- domains_names$domain_name

#Calculate AUCs for each ligand-method pair
all_auc_list <- list()
skipped_domains <- list()
bad_domains <- list()
bad_aucs <- list()

for (i in 1:length(ligands)) {
	ligand <- ligands[[i]]
	
	all_auc_list[[i]] <- list()
	skipped_domains[[i]] <- vector()
	bad_domains[[i]] <- list()
	bad_aucs[[i]] <- list()
	
	for (j in 1:length(methods)) {
		method <- methods[[j]]
		all_auc_list[[i]][[j]] <- vector()
		bad_domains[[i]][[j]] <- vector()
		bad_aucs[[i]][[j]] <- vector()
		
		for (domain in domains_names) {
			
			domain_table <- non_con_positions_table[non_con_positions_table["domain_name"] == domain, ]
			score_str <- paste0(ligand, "_binding_score")
			scores_table <- cbind(domain_table[method], domain_table[[score_str]])
			colnames(scores_table) <- c(method, score_str)
			
			#Create "labels" binary vector according to the ligand binding score
			labels <- unlist(lapply(scores_table[[score_str]], function(x) if (x >= 0.1) {1} else {0}))
			#Skipping domain if this ligand has no binding positions in this domain
			if (sum(labels) == 0) {
				skipped_domains[[i]] <- append(skipped_domains[[i]], domain)
				next
			}
			
			#Calculate AUC
			predictions <- domain_table[[method]]
			pred_obj <- prediction(predictions, labels)
			auc_obj <- performance(pred_obj, "auc")
			auc_val <- round(mean(as.numeric(auc_obj@y.values)), 4)
			if (auc_val < 0.5) {
				bad_domains[[i]][[j]] <- append(bad_domains[[i]][[j]], domain)
				bad_aucs[[i]][[j]] <- append(bad_aucs[[i]][[j]], auc_val)
			}
			all_auc_list[[i]][[j]] <- append(all_auc_list[[i]][[j]], auc_val)
			
			#Plot ROC
			#roc_perf_obj <- performance(pred_obj,"tpr","fpr")
			#plot(roc_perf_obj)
			#lines(x = c(0,1), y = c(0,1), col="grey38")
			
		}
		print(paste0("Finished method ", method,"\n"))
		
	}
	print(paste0("finished ligand ", ligand,"\n"))
}

#Print AUCs boxplots per ligand
color_codes <- c("#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2", "#bc80bd")
plots_list <- list()
for (i in 1:length(ligands)) {
	ligand <- ligands[[i]]
	ligand_auc_table <- data.frame(all_auc_list[[i]])
	colnames(ligand_auc_table) <- methods
	
	#melt table before ploting
	ligand_auc_table_melted <- melt(ligand_auc_table)
	
	#plot
	domain_num <- nrow(ligand_auc_table) 
	plots_list[[i]] <- ggplot(ligand_auc_table_melted, aes(x=variable, y=value, group=variable, fill = factor(variable))) +
		  	geom_boxplot(outlier.colour="black", outlier.shape=16, outlier.size=0.5, notch=TRUE) +
		  	geom_point(position=position_jitterdodge(jitter.width = 1.0), size =.25) +
			scale_fill_manual(values=color_codes)+
		  	#coord_cartesian(ylim = quantile(ligand_auc_table_melted$value, c(0, 0.9))) +
		  	#scale_y_log10() +
		  	ggtitle(paste0(ligand, " \n#(domains)=",domain_num)) +
		  	xlab("") +
		  	ylab("") +
		  	theme_bw() +
		  	theme(plot.title = element_text(hjust = 0.5, size=8), 
		  		  axis.text.x = element_blank(),
		  		  axis.title.y = element_text(size=8))
	
}
ggarrange(plotlist=plots_list[1:6], ncol=3, nrow=2, common.legend = TRUE, legend="bottom")
ggarrange(plotlist=plots_list[7:10], ncol=2, nrow=2, common.legend = TRUE, legend="bottom")


#Test for intersection
intersection_list <- list()
for (i in 1:length(ligands)) {
	intersection_list[[i]] <- bad_domains[[i]][[1]]
	
	for (j in 1:length(methods)) {
		intersection_list[[i]] <- intersect(intersection_list[[i]], bad_domains[[i]][[j]])
	}
}


