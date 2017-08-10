# Check user input and throw appropriate errors
#
# table: Features data frame
# counts: GO terms data frame
# groups: A list of groups
# features: A list of features
# labels: A list of feature names for axis lables
check_input <- function(table=data.frame(),counts=data.frame(),groups=list(),features=list(),labels=list()) {
  # Check groups
  if (length(counts) > 0 & length(groups) > 0) {
    for (g in groups) {
      for (t in strsplit(g,".",fixed=TRUE)[[1]]) {
        if (!(t %in% counts$number) && !(t %in% counts$name)) {
          stop(paste("Group \'", t, "\' does not exist", sep=""))
        }
      }
    }
  }
  
  # Check features
  if (length(table) > 0 & length(features) > 0) {
    for (var in features) {
      if (!(var %in% names(table))) {
        stop(paste("Column \'", var, "\' does not exist", sep=""))
      }
    }
  }
  
  # Check labels
  if (length(groups) > 0 & length(labels) > 0) {
    if (length(labels) != length(names)) {
      stop("Length of labels must match length of input")
    }
  }
}