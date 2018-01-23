library(ggplot2)

data_path <- "/home/anat/Research/ExAC/9.Features_exploration/ortho-para_analysis/topX_res_tables/"

top_frac_table <- read.csv(paste0(data_path, "top_frac.csv"), header = TRUE, sep = '\t', row.names = 1)
skipped_domains_X_table <- read.csv(paste0(data_path, "skipped_domains_X.csv"), header = TRUE, sep = '\t', row.names = 1)
skipped_domains_no_bind_table <- read.csv(paste0(data_path, "skipped_domains_no_bind.csv"), header = TRUE, sep = '\t', row.names = 1)


ligands <- c("dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "peptide", "ion", "metabolite", "max")
top_columns = c(10, 20, 30)
color_codes <- c("#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2")

for (i in 1:length(ligands)) {
	ligand <- ligands[[i]]
	
	ligand_table <- top_frac_table[top_frac_table["ligand"] == ligand]
}
ligands_data <- list()

for (i in 1:length(top_columns)) {
	
	X_data <- subset(top_frac_table, X==top_columns[[i]])
	ligands_data[[i]] <- split(X_data, X_data$ligand)
	
	for (j in 1:length(ligands)) {
		curr_table <- ligands_data[[i]][[j]]
		curr_ligand <- curr_table$ligand[[1]]
		title_str <- paste0("Number of binding positions in top ",top_columns[[i]]," predicted - ", curr_ligand)
		print(ggplot(data=curr_table, aes(x=method, y=hits_frac, fill=method)) +
			geom_boxplot() +
			scale_fill_manual(values=color_codes)+
			xlab("measurment") +
			ylab("hits fraction in top predicted") +
			ggtitle(title_str) +
			theme_bw() +
			theme(
				axis.text.y=element_text(size=14),
				axis.text.x=element_blank(),
				axis.ticks.x=element_blank()
			) +
			theme(axis.line = element_line(colour = "black"),
				  #panel.grid.major = element_blank(),
				  panel.grid.minor = element_blank(),
				  #panel.border = element_blank(),
				  panel.background = element_blank()))
	}
}