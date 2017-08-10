# Restricts features table to the specified groups and adds a group column
#
# domains_table: the features table
# groups: A list of the groups of interest
# labels: Optionally replaces the default names of the groups
restrict_groups <- function(domains_table,groups,labels=list()) {
  counts <- data.frame()
  domains_table_agg <- data.frame()
  cap_names <- list()
  for (i in 1:length(groups)) {
    g <- groups[i]
    gene_count <- 0
    domains <- list()
    # Find domains corresponding to this groups
    for (t in strsplit(g,".",fixed=TRUE)[[1]]) {
      t <- as.numeric(t)
      index <- which(freq_counts$number == t)
      domains <- append(domains,as.list(strsplit(freq_counts$domains, ".", fixed=TRUE)[[index]]))
    }
    # Remove overlap and find the number of genes
    domains_unique <- list()
    for (d in domains) {
      if (d != "" && !(d %in% domains_unique)) {
        domains_unique <- append(domains_unique,d)
        gene_count <- gene_count + domains_table[d,"num_genes"]
      }
    }
    group_table <- domains_table[which(row.names(domains_table) %in% domains_unique),]
    # Keep track of the number of domains in each group
    counts <- rbind(counts,c(length(domains_unique),gene_count))
    # Find and format group names
    if (length(labels) == 0) {
      name <- freq_counts$name[which(freq_counts$number %in% as.numeric(strsplit(g,".",fixed=TRUE)[[1]][1]))]
    } else {
      name <- labels[i]
    }
    capitalized <- paste(toupper(substr(name, 1, 1)),substr(name, 2, nchar(name)),sep="")
    group_table$Group <- capitalized
    cap_names <- append(cap_names,capitalized)
    # Combine groups
    domains_table_agg <- rbind(domains_table_agg,group_table)
  }
  row.names(counts) <- cap_names
  names(counts) <- c("Domains","Genes")
  return(list(domains_table_agg,counts))
}