library(ROCR)
library(PRROC)

data_path = "/home/anat/Research/ExAC/10.Prediction/ligand_df/"

#ligands
ligands_list <- c("dna", "rna", "ion", "peptide", "metabolite", "all_ligands", "dna_not_con", "rna_not_con", "ion_not_con", "peptide_not_con", "metabolite_not_con", "all_ligands_not_con")

ligand_data <- list()
#Read the ligands datasets
for (i in 1:length(ligands_list)) {
	ligand <- ligands_list[[i]]
  	ligand_filename <- paste0(ligand, "_0.1.csv")
  	ligand_data[[i]] <- read.csv(paste0(data_path, ligand_filename), header = TRUE, sep = '\t', row.names = 1)
}

auprc_data <- list()
#Read the AUPRC tables
for (i in 1:length(ligands_list)) {
	ligand <- ligands_list[[i]]
	auprc_filename <- paste0(ligand, "_0.1_auprc.csv")
	auprc_data[[i]] <- read.csv(paste0(data_path, auprc_filename), header = TRUE, sep = '\t', row.names = 1)
}

#Arrange the data according to models and folds
models_data <- list()
pred_folds <- list()
labels_folds <- list()
pred_obj <- list()
roc_perf_obj <- list()
auc_res <- list()
ac_obj <- list()
legend_str <- list()
colors_list <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")
for (i in 1:length(ligands_list)) {
  	#Split by the different models
  	models_data[[i]] <- split(ligand_data[[i]], ligand_data[[i]]$model)
  	models_names <- attributes(models_data[[i]])$name
  
  	#init model lists
  	pred_folds[[i]] <- list()
  	labels_folds[[i]] <- list()
  	pred_obj[[i]] <- list()
  	roc_perf_obj[[i]] <- list()
  	auc_res[[i]] <- list()

  	#For each model, split by different folds
  	for (j in 1:length(models_names)) {
  		
  		model_name <- models_names[j]
    
    	pred_folds[[i]][[j]] <- split(models_data[[i]][[model_name]]$prob, models_data[[i]][[model_name]]$fold)
    	labels_folds[[i]][[j]] <- split(models_data[[i]][[model_name]]$obs, models_data[[i]][[model_name]]$fold)
    
    	pred_obj[[i]][[j]] <- prediction(pred_folds[[i]][[j]], labels_folds[[i]][[j]])
    	roc_perf_obj[[i]][[j]] <- performance(pred_obj[[i]][[j]],"tpr","fpr")
    
    	ac_obj[[i]] <- performance(pred_obj[[i]][[j]], "auc")
    	auc_res[[i]][[j]] <- round(mean(as.numeric(ac_obj[[i]]@y.values)), 4)
    
    	#ROC curve
    	if (j == 1) {
      		#plot(perf_obj[[i]][[j]],col="grey82",lty=3)
      		plot(roc_perf_obj[[i]][[j]],col=colors_list[j], lwd=3,avg="vertical", main=ligands_list[[i]])
    		#plot(roc_perf_obj[[i]][[j]],col=colors_list[j], lwd=3,avg="vertical", main="small-molecule")
      		lines(x = c(0,1), y = c(0,1), col="grey38")
    	} else {
      		#plot(perf_obj[[i]][[j]],col="grey82",lty=3, add=TRUE)
      		plot(roc_perf_obj[[i]][[j]],col=colors_list[j], lwd=3,avg="vertical", add=TRUE)
    	}
  	}
  	legend_str[[i]] <-  paste0(models_names,", AUC=", auc_res[[i]])
  	legend("bottomright", title= "Model", legend_str[[i]], lty=1, col=colors_list)
  	grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")
}

models_data <- list()
pred_folds <- list()
labels_folds <- list()
pred_obj <- list()
pr_perf_obj <- list()
auprc_res <- list()
ac_obj <- list()
legend_str <- list()
positives_num <- list()
total_num <- list()
random_val <- list()
postivies_prob <- list()
negatives_prob <- list()
colors_list <- c("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02")
for (i in 1:length(ligands_list)) {
	
	#Split by the different models
	models_data[[i]] <- split(ligand_data[[i]], ligand_data[[i]]$model)
	models_names <- attributes(models_data[[i]])$name
	
	#init model lists
	pred_folds[[i]] <- list()
	labels_folds[[i]] <- list()
	pred_obj[[i]] <- list()
	pr_perf_obj[[i]] <- list()
	auprc_res[[i]] <- list()
	postivies_prob[[i]] <- list()
	negatives_prob[[i]] <- list()
	
	#For each model, split by different folds
	for (j in 1:length(models_names)) {
		
		model_name <- models_names[j]
		
		pred_folds[[i]][[j]] <- split(models_data[[i]][[model_name]]$prob, models_data[[i]][[model_name]]$fold)
		labels_folds[[i]][[j]] <- split(models_data[[i]][[model_name]]$obs, models_data[[i]][[model_name]]$fold)
		
		pred_obj[[i]][[j]] <- prediction(pred_folds[[i]][[j]], labels_folds[[i]][[j]])
		pr_perf_obj[[i]][[j]] <- performance(pred_obj[[i]][[j]],"prec","rec")
		
		#parse AUPRC
		auprc_res[[i]][[j]] <- round(sum(auprc_data[[i]][[model_name]])/10.0, 4)
		
		#PR curve
		if (j == 1) {
			#plot(perf_obj[[i]][[j]],col="grey82",lty=3)
			plot(pr_perf_obj[[i]][[j]],col=colors_list[j], lwd=3,avg="vertical", main=ligands_list[[i]], xlim=c(0, 1), ylim=c(0,1))
			#plot(roc_perf_obj[[i]][[j]],col=colors_list[j], lwd=3,avg="vertical", main="small-molecule")
			#
		} else {
			#plot(perf_obj[[i]][[j]],col="grey82",lty=3, add=TRUE)
			plot(pr_perf_obj[[i]][[j]],col=colors_list[j], lwd=3,avg="vertical", xlim=c(0, 1), ylim=c(0,1), add=TRUE)
		}
	}
	#Compute random line
	postivies_prob[[i]][[j]] <- subset(models_data[[i]][[model_name]], obs==1)$prob
	negatives_prob[[i]][[j]] <- subset(models_data[[i]][[model_name]], obs==0)$prob
	positives_num[[i]] <- length(postivies_prob[[i]][[j]])
	total_num[[i]] <- length(negatives_prob[[i]][[j]]) + positives_num[[i]]
	random_val[[i]] <- positives_num[[i]]/total_num[[i]]
	abline(h=random_val[[i]], col="grey38")
	
	#Add legend
	legend_str[[i]] <-  paste0(models_names,", AUC=", auprc_res[[i]])
	legend("topright", title= "Model", legend_str[[i]], lty=1, col=colors_list)
	grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")
}

### From this point: Testing code
model_dfs_list <- split(ridge_data , f = ridge_data$model)
auc_vec = c()
pr_list = list()
for (i in 1:length(model_dfs_list)) {
  fg <- subset(model_dfs_list[[i]], obs==1)$prob
  bg <- subset(model_dfs_list[[i]], obs==0)$prob
  pr<-pr.curve(scores.class0 = fg, scores.class1 = bg, curve=TRUE,  rand.compute = T, max.compute = T, min.compute = T)
  pr_list[[i]] <- pr
  auc_vec[i] <- pr$auc.integral
  
  if (i == 1) {
    plot(pr_list[[i]], col=i, main = "Ridge PR", auc.main = FALSE,  rand.plot = TRUE,  max.plot = TRUE, min.plot = TRUE, legend = TRUE)
  } else {
    plot(pr_list[[i]], add=TRUE, col=i, main = "Ridge PR", auc.main = FALSE, legend = TRUE)
  }
}

models_strs = unique(logistic_data$model)
legend_strs <- paste(models_strs, ", AUC=", format(auc_vec, digits=4))
legend("topright", title="model", legend=legend_strs,
       col=seq(6), lty=1, cex=0.8)

plot(pr_list[[1]], col=colors_vec[1], auc.main = FALSE,  rand.plot = TRUE,  max.plot = TRUE, min.plot = TRUE, fill.area = T)