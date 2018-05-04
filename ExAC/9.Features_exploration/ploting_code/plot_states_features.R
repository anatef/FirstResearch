library(ggplot2)
library(ggfortify)
library(rpart)

data_path = "/home/anat/Research/ExAC/9.Features_exploration/"

#states_filename = "domains_states_features_df.csv"
states_filename10 = "binding10_states_SB.csv"
states_filename50 = "binding_50instances_states_features_df.csv"
states_filename100 = "binding_100instances_states_features_df.csv"

train100_filename <- "rand_binding_100_train_df.csv"
test100_filename <- "rand_binding_100_test_df.csv"

rand_states_filename100 = "rand_binding_100instances_states_features_df_5.csv"
rand_states_filename50 = "rand_binding_50instances_states_features_df.csv"
rand_states_filename10 = "rand_binding_10instances_states_features_df.csv"
rand_2class_filename100 = "rand_binding_100instances_states_features_df_2classes_2.csv"

states_table10 <- read.csv(paste0(data_path, states_filename10), header = TRUE, sep = '\t', row.names = 1)
states_table50 <- read.csv(paste0(data_path, states_filename50), header = TRUE, sep = '\t', row.names = 1)
states_table100 <- read.csv(paste0(data_path, states_filename100), header = TRUE, sep = '\t', row.names = 1)

rand_table100 <- read.csv(paste0(data_path, rand_states_filename100), header = TRUE, sep = '\t', row.names = 1)
rand_table50 <- read.csv(paste0(data_path, rand_states_filename50), header = TRUE, sep = '\t', row.names = 1)
rand_table10 <- read.csv(paste0(data_path, rand_states_filename10), header = TRUE, sep = '\t', row.names = 1)
rand_2class <- read.csv(paste0(data_path, rand_2class_filename100), header = TRUE, sep = '\t', row.names = 1)

train100 <- read.csv(paste0(data_path, train100_filename), header = TRUE, sep = '\t', row.names = 1)
test100 <- read.csv(paste0(data_path, test100_filename), header = TRUE, sep = '\t', row.names = 1)

ggplot(rand_table100, aes(x=BLOSUM_avg, y=alter_num_aa_norm)) +
  geom_point(aes(shape=as.factor(state_type), color=as.factor(state_type)), alpha=0.5, na.rm = TRUE, size=3) +
  #scale_shape_identity() +
  scale_shape_manual(values = c(19 ,18, 17)) +
  #coord_cartesian(xlim = c(0, 0.0000001), ylim = c(0.9, 1)) +
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


#PCA
pca_df100 <- states_table100[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)]



#A function to scale features
z_scoring <- function(table) {
  z_table <- table[, FALSE]
  for (i in names(table)) {
    z_col <- scale(table[i], center = TRUE, scale = TRUE)
    z_table <- cbind(z_table, z_col)
  }
  colnames(z_table) <- names(table)
  return(z_table)
}
scaled_pca_df100 <- z_scoring(pca_df100)


prin_comp100 <- prcomp(scaled_pca_df100)
autoplot(prcomp(scaled_pca_df100), data = states_table100, colour = 'state_type', loadings = TRUE, loadings.label = TRUE)

pca_rand_100 <- rand_table100[c(1,2,3,4,5,6,7,12,14,15)]
rand_scaled_pca_df100 <- z_scoring(pca_rand_100)
prin_comp100 <- prcomp(rand_scaled_pca_df100)
autoplot(prcomp(rand_scaled_pca_df100), data = rand_table100, colour = 'state_type', alpha=0.5, frame = TRUE)

pca_2class <- rand_2class[c(1,2,3,4,5,6,7,12,14,15)]
rand_scaled_2class <- z_scoring(pca_2class)
prin_comp_2class <- prcomp(rand_scaled_2class)
autoplot(prcomp(rand_scaled_2class), data = rand_2class, colour = 'state_type', alpha=0.5)

pca_rand_50 <- rand_table50[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)]
rand_scaled_pca_df50 <- z_scoring(pca_rand_50)
prin_comp50 <- prcomp(rand_scaled_pca_df50)
autoplot(prcomp(rand_scaled_pca_df50), data = rand_table50, colour = 'state_type', alpha=0.5)

pca_rand_10 <- rand_table10[c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)]
rand_scaled_pca_df10 <- z_scoring(pca_rand_10)
prin_comp10 <- prcomp(rand_scaled_pca_df10)
autoplot(prcomp(rand_scaled_pca_df10), data = rand_table10, colour = 'state_type', alpha=0.5)

###Density functions of PC1 and PC2###:

#Adding PC1, PC2 values to the big data-frame
PCs_vals <- prin_comp50$x
pc1_vals <- PCs_vals[,1]
pc2_vals <- PCs_vals[,2]
rand_table50_pc1pc2 <- cbind(rand_table50, pc1_vals, pc2_vals)

#Plot density - 3 classes
ggplot(rand_table50_pc1pc2, aes(pc2_vals, colour = state_type)) +
  geom_density(alpha = 0.1) +
  theme_bw() +
  theme(axis.line = element_line(colour = "gray"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

ggplot(rand_table50_pc1pc2, aes(pc2_vals, colour = state_type)) +
  geom_density(alpha = 0.1) +
  guides(fill=FALSE) +
  scale_x_reverse()+
  theme_bw() +
  theme(axis.line = element_line(colour = "gray"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        panel.background = element_blank())

#Adding PC1, PC2 values to the big data-frame
PCs_vals_2class <- prin_comp_2class$x
pc1_vals_2class <- PCs_vals_2class[,1]
pc2_vals_2class <- PCs_vals_2class[,2]
rand_table100_pc1pc2_2class <- cbind(rand_2class, pc1_vals_2class, pc2_vals_2class)






#Plot density - 2 classes
ggplot(rand_table100_pc1pc2_2class, aes(pc1_vals_2class, colour = state_type)) +
  geom_density(alpha = 0.1) +

ggplot(rand_table100_pc1pc2_2class, aes(pc2_vals_2class, colour = state_type)) +
  geom_density(alpha = 0.1) +
  theme_bw()

#compute standard deviation of each principal component
std_dev <- prin_comp50$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
pr_var

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex

#scree plot
plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")

#compute standard deviation of each principal component
std_dev <- prin_comp_2class$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
pr_var

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
prop_varex

#scree plot
plot(prop_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")


#Transform state_type to numbers
type_trans <- as.numeric(train100[,"state_type"])

#Predict

train_pca <- data.frame(type_trans, prin_comp100$x)
rpart.model <- rpart(type_trans ~ .,data = train_pca, method = "anova")

#transform test into PCA
test.data <- predict(prin_comp100, newdata = test100)
test.data <- as.data.frame(test.data)

#make prediction on test data
rpart.prediction <- predict(rpart.model, test.data)

#For fun, finally check your score of leaderboard
sample <- read.csv("SampleSubmission_TmnO39y.csv")
final.sub <- data.frame(Item_Identifier = sample$Item_Identifier, Outlet_Identifier = sample$Outlet_Identifier, Item_Outlet_Sales = rpart.prediction)
write.csv(final.sub, "pca.csv",row.names = F)
