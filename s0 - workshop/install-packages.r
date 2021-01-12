
# Function to call all packages
install_packages <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg, dependencies = TRUE)
  sapply(pkg, require, character.only = TRUE)
}

# lists of all packages
list_packages = c("DBI", "odbc", "RPostgresSQL", "xlsx", "tidyr", "stringr",
                     "lubridate", "ggvis", "rgl", "htmlwidgets", "googleVis",
                     "car", "mgcv", "shiny", "xtable", "maptools", "ggmap",
                     "zoo", "xts", "Rcpp", "data.table", "devtools")


# install

install_packages(list_packages)
