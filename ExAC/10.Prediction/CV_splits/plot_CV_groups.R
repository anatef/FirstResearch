library(ggplot2)
library(reshape)

data_path = "/home/anat/Research/ExAC/10.Prediction/CV_splits/"


ligands_list <- c("dna", "dnabase", "dnabackbone", "rna", "rnabase", "rnabackbone", "ion", "peptide", "metabolite")

#Read table and flat it for plotting
filename <- "group_stats_df.csv"
pos_neg_table <- read.csv(paste0(data_path, filename), header = TRUE, sep = '\t', row.names = 1)
pos_neg_table["ligand"] <- rownames(pos_neg_table)
pos_table <- subset(pos_neg_table, endsWith(ligand, "pos"))
neg_table <- subset(pos_neg_table, endsWith(ligand, "neg"))

pos_neg_table_melted <- melt(pos_neg_table, id="ligand")
colnames(pos_neg_table_melted) <- c("ligand", "group", "num")
pos_table_melted <- melt(pos_table, id="ligand")
colnames(pos_table_melted) <- c("ligand", "group", "num")
neg_table_melted <- melt(neg_table, id="ligand")
colnames(neg_table_melted) <- c("ligand", "group", "num")

ggplot(neg_table_melted, aes(x=ligand, y=num)) +
	geom_bar(stat="identity", aes(fill = group)) +
	scale_fill_brewer(type="qual", palette = 3) +
	#ggtitle("groups positives in the CV splits") +
	ggtitle("ligands negatives in the CV splits") +
	theme_bw() +
	theme(plot.title = element_text(hjust = 0.5))


