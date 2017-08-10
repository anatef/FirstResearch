# From https://cran.r-project.org/web/packages/gridExtra/vignettes/tableGrob.html
# Some slight formatting modifications were made
#
# Creates a table grob
# d: Data frame to be converted
# padding: Spacing around text
# size: Size of text
# xshift: x coord of upper left corner
# yshift: y coord of upper left corner
ftable <- function(d, padding = unit(4, "mm"), size = 1, xshift = 0, yshift = 0,...) {
  
  nc <- ncol(d)
  nr <- nrow(d)
  
  ## character table with added row and column names
  extended_matrix <- cbind(c("", rownames(d)),
                           rbind(colnames(d),
                                 as.matrix(d)))
  
  ## string width and height
  w <- apply(extended_matrix, 2, strwidth, "inch", cex=size)
  h <- apply(extended_matrix, 2, strheight, "inch", cex=size)
  
  widths <- apply(w, 2, max)
  heights <- apply(h, 1, max)
  
  padding <- convertUnit(padding, unitTo = "in", valueOnly = TRUE)
  
  x <- xshift + cumsum(widths + padding) - 0.5 * padding
  y <- yshift + cumsum(heights + padding) - padding
  
  rg <- rectGrob(x = unit(x - widths/2, "in"),
                 y = unit(1, "npc") - unit(rep(y, each = nc + 1), "in"),
                 width = unit(widths + padding, "in"),
                 height = unit(heights + padding, "in"))
  
  tg <- textGrob(c(t(extended_matrix)), x = unit(x - widths/2, "in"),
                 y = unit(1, "npc") - unit(rep(y, each = nc + 1), "in"),
                 just = "center",gp=gpar(cex=size))
  
  g <- gTree(children = gList(rg, tg), ...,
             x = x, y = y, widths = widths, heights = heights)
  
  return(g)
}