# Get packages
library(org.Sc.sgd.db)
library(GO.db)
library(ggplot2)
library(grid)
library(gridExtra)

# Find file
instance_cutoff <- "10"
data_path <- "/Users/davidandrewtodd/summer_research/ExAC/9.Features_exploration/"
filename <- paste("freq_counts",instance_cutoff,".csv",sep="")

# Load file
freq_counts <- read.csv(paste0(data_path, filename), header = TRUE, sep = ',', row.names = 1, stringsAsFactors = FALSE)

# Get GO hierarchy
master <- c(as.list(GOBPPARENTS), as.list(GOMFPARENTS), as.list(GOCCPARENTS))

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

# Remove groups with less than 10 domains
#freq_counts <- freq_counts[-which(freq_counts$count < 10),]
row.names(freq_counts) <- 1:nrow(freq_counts)

# Clear values
rm("count","d","domains","filename","i","index","key","num","p","parent_num","parents","r")

# Export file
write.csv(file=paste("freq_counts_broad",instance_cutoff,sep=""), x=freq_counts)