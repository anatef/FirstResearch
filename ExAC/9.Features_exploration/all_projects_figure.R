library(ggplot2)

data_path = "~/Documents/Courses/COS551/project/code/"

#===import the data===#
pvalues_data <- read.csv(paste0(data_path,"pvalues/pvalues_csv/pvals_corrected_all_projects.csv"), header = TRUE)
enrtopy_data <- read.csv(paste0(data_path,"entropy/all_entropies.csv"), header = TRUE)

#===Combine data (add the entropies to the domains in pvalues, which are less)===#
#Exclude domains that were removed in the pvalue analysis
pvals_domains <- unique(as.vector(pvalues_data[,"Domain"]))
entropy_for_pvals <- enrtopy_data[enrtopy_data$domains %in% pvals_domains,]

#Sort entropy data according to domain
entropy_sorted <- entropy_for_pvals[with(entropy_for_pvals, order(domains)), ]

#Create a combined data frame
plot_df <- NULL
data.frame(plot_df)
entropy_vals <- entropy_sorted[,"entropies3"]
plot_df <- cbind(pvalues_data, entropy_vals)


#===Creating the plot===#
#Size transformation for the size scale
size_trans <- cut(plot_df[,"Mutations_num"], breaks=c(10,100,1000,10000,100000), labels=FALSE)
plot_df <- cbind(plot_df, size_trans)

#The last break for the colors scale
max_break <- floor(max(plot_df[,"Genes_num_log2"]))

#Colors for the color gradient guide
colfunc<-colorRampPalette(c("dark blue", "blue", "cyan", "green", "yellow", "orange","orangered" ,"red", "dark red"))

#The value fot the dashed line
significant_pval <- -log10(0.05)

#Adding domain name for interesting domains
domain_text <- vector(mode = "character", length = nrow(plot_df))
interesting_domains <- c(15, 189, 196, 301, 328, 574, 567, 588, 725, 746, 754, 790, 873, 1027, 1124, 1111, 1171, 1179, 1134, 1209, 1212, 1329, 1349, 1488, 1494)
for (i in interesting_domains) {
  domain_text[i] <- as.vector(plot_df[i,"Domain"])
}
plot_df <- cbind(plot_df, domain_text)

#Creating the plot
ggplot(plot_df, aes(x=entropy_vals, y=log10_mul_corrected, size=size_trans, color=Genes_num_log2)) +
  geom_point(na.rm = TRUE) +
  scale_color_gradientn(expression(atop(log[2]*"(number of genes", "in domain family)")),colours = colfunc(50), guide = "colourbar", breaks=0:max_break) +
  scale_x_continuous(expression("Normalized entropy score "*(bar(S))), breaks=seq(0,1,0.2)) +
  scale_y_continuous(expression(atop("Domain mutation burden test",-log[10]*"(corrected p-value)")), breaks=seq(-4,3,1)) +
  theme(legend.title = element_text(vjust = 1)) +
  guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  scale_size("Number of\nmutations in\ndomain", breaks=c(4,3,2,1), labels=c("10,000","1,000","100","10"), range=c(1,4)) +
  geom_segment(aes(x = 0, y = significant_pval, xend = 1, yend = significant_pval), colour="black", size=0.05, linetype="longdash") +
  geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

#To view only the domains with low pvalue along with the entropy:
plot_df_by_pval <- plot_df[with(plot_df, order(-log10_mul_corrected)), ]
plot_df_by_pval <- plot_df_by_pval[,c("Domain", "log10_mul_corrected", "entropy_vals")]

#To view only the domains with low pvalue along with the entropy:
plot_df_by_log2genes <- plot_df[with(plot_df, order(-Genes_num_log2)), ]
plot_df_by_log2genes <- plot_df_by_log2genes[,c("Domain", "log10_mul_corrected", "entropy_vals", "Genes_num_log2")]

#Getting a list of interesting domains
max_pval <- max(plot_df$log10_mul_corrected)
highest_pvals_indices <- which(plot_df$log10_mul_corrected == max_pval)
highest_pval_domains <- plot_df[highest_pvals_indices, ]
sort_highest_pval_by_entropy <- highest_pval_domains[with(highest_pval_domains, order(-entropy_vals)),]
go_selected <- sort_highest_pval_by_entropy[sort_highest_pval_by_entropy$entropy_vals >=0.95, ]
go_domains <- as.vector(go_selected[,"Domain"])
write.csv(go_domains, paste0(data_path,"GO/domains_0.95.csv"))

