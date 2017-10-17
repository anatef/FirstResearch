# Swims up the GO tree
#
# Pre-req: freq_counts must be computed and saved in a csv
# 
# instance_cutoff: Min number of instances for domain to be included (10, 50, or 100)
# data_path: The path to freq_counts (not including the file itself)
domain_go <- function(instance_cutoff,data_path) {
  # Get packages
  #source("http://bioconductor.org/biocLite.R")
  #biocLite("GO.db")
  library(GO.db)
  
  # Find file
  filename <- paste("freq_counts",instance_cutoff,".csv",sep="")
  
  # Load file
  freq_counts <- read.csv(paste0(data_path, filename), header = TRUE, sep = ',', row.names = 1, stringsAsFactors = FALSE)
  
  # Get GO hierarchy
  master <- c(as.list(GOBPANCESTOR), as.list(GOMFANCESTOR), as.list(GOCCANCESTOR))
  
  # Combine terms where possible
  for (i in 1:20) {
    for (r in 1:nrow(freq_counts)) {
      num = freq_counts$number[r]
      count = freq_counts$count[r]
      domains = freq_counts$domains[r]
      
      # Convert the GO term to the format used
      key <- toString(num)
      while (nchar(key) < 7) {
        key <- paste("0", key, sep="")
      }
      key <- paste("GO:", key, sep="")
      
      # Get parents
      parents <- master[[key]]
      for (p in 1:length(parents)) {
        # Check if parent is an existing tag and handle merging
        parent_num = as.integer(substr(parents[[p]],4,10))
        if (parents[[p]] != "all") {
          if (parent_num %in% freq_counts$number) {
            index <- which(freq_counts$number == parent_num)
            # Combine counts and domains
            for (d in as.list(strsplit(domains, ".", fixed=TRUE)[[1]])) {
              # Make sure domain is not already included in parent GO term
              if (regexpr(d, freq_counts$domains[index], fixed=TRUE)[1] == -1) {
                freq_counts$domains[index] <- paste(freq_counts$domains[index],".",d,sep="")
                freq_counts$count[index] <- freq_counts$count[index] + 1
              }
            }
            # Otherwise, create a new row and append it
          } else {
            temp <- data.frame(parent_num, count, Term(parents[[p]]), domains)
            names(temp) <- c("number", "count", "name", "domains")
            freq_counts <- rbind(freq_counts, temp)
          }
        }
      }
    }
  }
  
  # Format data frame and save to csv
  row.names(freq_counts) <- 1:nrow(freq_counts)
  write.csv(file=paste("freq_counts_broad",instance_cutoff,sep=""), x=freq_counts)
}