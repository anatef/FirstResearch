# Converts GO term descriptions to their corresponding numbers
#
# groups: A list of groups to be converted
# freq_counts: The GO term table that contains the mapping
groups_to_numbers <- function(groups,freq_counts) {
  # Convert GO descriptions to numbers
  ret <- vector(length=length(groups))
  for (i in 1:length(groups)) {
    if (is.character(groups[i])) {
      temp <- ""
      for (t in strsplit(groups[i],".",fixed=TRUE)[[1]]) {
        # Just append the first number
        if (nchar(temp) == 0) {
          temp <- toString(freq_counts$number[which(freq_counts$name == t)])
        # Otherwise add a '.' as a delimiter
        } else {
          temp <- paste(temp,".",toString(freq_counts$number[which(freq_counts$name == t)]),sep="")
        }
      }
      ret[i] <- temp
    }
  }
  return(ret)
}