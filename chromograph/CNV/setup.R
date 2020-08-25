## Install a whole bunch op packages

install.packages(c("beanplot", "mixtools", "pheatmap", "zoo", "squash", "devtools", "Rtsne"), "/home/camiel/anaconda3/envs/conics/lib/R/library/", repos = "http://cran.us.r-project.org")

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", "/home/camiel/anaconda3/envs/conics/lib/R/library/", repos = "http://cran.us.r-project.org")

BiocManager::install("scran", lib="/home/camiel/anaconda3/envs/conics/lib/R/library/")

#devtools::install_github("diazlab/CONICS/CONICSmat", dep = FALSE, "/home/camiel/anaconda3/envs/conics/lib/R/library/")
