library(ggplot2)
library(ggfortify)

data_path = "/home/anat/Research/ExAC/9.Features_exploration/"

#states_filename = "domains_states_features_df.csv"
states_filename = "binding_50instances_states_features_df.csv"

states_table <- read.csv(paste0(data_path, states_filename), header = TRUE, sep = '\t', row.names = 1)

ggplot(states_table, aes(x=EMD, y=rare_poly_0.05.)) +
  geom_point(aes(shape=as.factor(state_type), color=as.factor(state_type)), alpha=0.5, na.rm = TRUE, size=3) +
  #scale_shape_identity() +
  scale_shape_manual(values = c(19 ,18, 17)) +
  coord_cartesian(xlim = c(0, 0.0000001), ylim = c(0.9, 1)) +
  #scale_color_gradientn(expression(atop(log[2]*"(number of genes", "in domain family)")),colours = colfunc(50), guide = "colourbar", breaks=0:max_break) +
  #geom_smooth(method=lm, se=FALSE, show.legend=FALSE, color="black", size=0.4) +
  #scale_x_continuous("Fraction of sites with alteration") +
  #scale_y_continuous("Fraction of polymorphic sites (out of all altered)") +
  #theme(legend.title = element_text(vjust = 1)) +
  #guides(size = guide_legend(order = 1), colour = guide_colourbar(order = 2)) +
  #scale_size("Number of\nmutations in\ndomain", breaks=c(3,2,1), labels=c("10,000","1,000","100"), range=c(1,3)) +
  #geom_text(aes(label=domain_text),hjust=0, vjust=0, color="black", size=3, angle=45) +
  theme_bw() +
  theme(axis.line = element_line(colour = "black"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        #panel.border = element_blank(),
        panel.background = element_blank())


pca_df <- states_table[c(1,2,3,4,5,6,7,8,9,10,11,12)]
z_scoring <- function(table) {
  z_table <- table[, FALSE]
  for (i in names(table)) {
    z_col <- scale(table[i], center = TRUE, scale = TRUE)
    z_table <- cbind(z_table, z_col)
  }
  colnames(z_table) <- names(table)
  return(z_table)
}
scaled_pca_df <- z_scoring(pca_df)

prin_comp <- prcomp(pca_df)
autoplot(prcomp(pca_df), data = states_table, colour = 'state_type', loadings = TRUE, loadings.label = TRUE)