library(parallel)

generatePopulation <- function(cell_type){
  library(SingleCellExperiment)
  library(Seurat)
  library(dplyr)
  library(SPARSim)
  library(Matrix)
  sce_path = "/albona/nobackup/biostat/datasets/singlecell/COVID19/"
  sce_file = "sce_COVID19_PBMC_10x_cohort1.rds"
  path = "/albona/nobackup/biostat/datasets/singlecell/COVID19/processed/"
  metadata_file = "logcounts_Schulte_cohort1_metadata.csv"
  genenames_file = "logcounts_Schulte_cohort1_geneNames.csv"
  

  save_directory = "/albona/nobackup/andrewl/honours_code/population_params/"
  
  save_path = paste0(save_directory)
  
  if (!dir.exists(save_path)){
    dir.create(save_path, recursive = TRUE)
  } else {
    print("Dir already exists!")
  }
  
  # Yingxin's file
  sce <- readRDS(paste0(sce_path, sce_file))
  # Raw counts matrix
  raw_counts <- counts(sce)
  # Log normalised via seurat
  norm_counts <- NormalizeData(object = raw_counts, normalization.method = "LogNormalize")
  
  # Get metadata 
  df <- read.csv(paste0(path, metadata_file))
  # Get the cells we want to keep after removing 2 classes
  allowed_cells = df %>% filter(!(celltypes %in% c("intermediate", "unassigned")) ) %>% pull(X)
  # Normalised counts matrix with the 2 classes removed
  allowed_cell_counts <- norm_counts[,colnames(norm_counts) %in% allowed_cells]
  
  # Gene names
  gene_names_df = read.csv(paste0(path, genenames_file))
  genes = gene_names_df$x
  # Keeping only the genes that Yingxin's logCounts processed file had
  allowed_genes_counts <- allowed_cell_counts[rownames(allowed_cell_counts) %in% genes, ]
  # Ordering matrix is same order
  ordered_counts <- allowed_genes_counts[genes, ]
  
  # Getting the cells that Yingxin's logCounts processed file had
  celltypes_list <- df %>% filter(X %in% colnames(ordered_counts))
  # Ordering matrix's cells in same order
  filter_cond <- celltypes_list %>% pull(X)
  celltypes_order <- celltypes_list %>% pull(celltypes)
  ordered_cells_counts <- ordered_counts[, filter_cond]
  
  # Applying similar process to raw counts for sparSim
  ordered_raw_counts <- raw_counts[rownames(ordered_cells_counts), colnames(ordered_cells_counts)]
  
  cond <- list(as.vector(which(celltypes_order == cell_type)))
  names(cond) = c(cell_type)
  SPARSim_sim_param <- SPARSim_estimate_parameter_from_data(
    raw_data = as.matrix(ordered_raw_counts),
    norm_data = as.matrix(ordered_cells_counts),
    conditions = cond
  )
  
  # Save a particular repeat's population profile
  saveRDS(SPARSim_sim_param, paste0(save_path, cell_type, ".Rds"))
  
}



ncores <- 8
cl <- makeCluster(ncores)

cell_types = c("CD14 Monocyte", "NK", "CD4n T", "B", "DC", "CD8m T", "CD4m T", "CD16 Monocyte", "pDC", "CD8eff T", "Platelet", "Neutrophil", "IgG PB", "IgA PB", "Activated Granulocyte", "SC & Eosinophil", "RBC")

clusterApply(cl, cell_types, fun=generatePopulation)