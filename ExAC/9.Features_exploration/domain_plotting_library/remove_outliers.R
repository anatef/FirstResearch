# Removes outliers from features
#
# features: A list of features
# domains_table: The features data frame
remove_outliers <- function(features,domains_table) {
  for (var in features) {
    x <- domains_table[[var]]
    x_sd <- sd(x,na.rm=TRUE)
    x_mean <- mean(x,na.rm=TRUE)
    for (i in 1:length(x)) {
      # Remove points greater than 3 SD from the mean
      if (is.na(x[i]) || x[i] > x_mean + 3*x_sd || x[i] < x_mean - 3*x_sd) {
        domains_table[[var]][i] = NA
      }
    }
  }
  return(domains_table)
}