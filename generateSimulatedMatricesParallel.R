library(parallel)
generateDE <- function(cell_type){
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
  
  save_directory = "/albona/nobackup/andrewl/honours_code/simulated_matrix/"

  save_path = paste0(save_directory, cell_type, "/")
  
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
  
  
  parameter_directory = "/albona/nobackup/andrewl/honours_code/population_params/"
  
  # Removing highly variable genes
  m_SPARSim_sim_param = readRDS(paste0(parameter_directory, cell_type, '.Rds'))
  q_limit = quantile(m_SPARSim_sim_param[[1]]$variability, 0.95, na.rm=TRUE)
  remove = as.vector(which(m_SPARSim_sim_param[[1]]$variability >= q_limit))
  ordered_raw_counts = ordered_raw_counts[-remove, ]
  ordered_cells_counts = ordered_cells_counts[-remove, ]
  
  total_matrix <- as.vector(which(celltypes_order == cell_type))
  
  write.csv(data.frame(gene_names = rownames(ordered_cells_counts)), paste0(save_path, "geneNames.csv"))
  
  # Different fc multipler bounds
  de_params = list(c(2,5))
  
  N_de = 100
  gene_total = nrow(ordered_cells_counts)
  range = 1:gene_total
  group_1 = sample(range, N_de) # Original population
  group_2 = sample(which(!(range %in% group_1)), N_de) # Completely different
  write.csv(data.frame(index = group_1), paste0(save_path, "group_1_genes.csv"))
  write.csv(data.frame(index = group_2), paste0(save_path,"group_2_genes.csv"))
  group_3 = sample(group_1, N_de/2)
  extra = sample(which(!(range %in% group_1)), N_de/2)
  group_3 = c(group_3, extra)
  write.csv(data.frame(index = group_3), paste0(save_path,"group_3_genes.csv"))
  
  markers_list = list(group_1, group_2, group_3)
  
  

  for(t in 1:30){
    
    count <- min(5000, length(total_matrix))
    sample_matrix <- sample(total_matrix, count)
    cond <- list(sample_matrix)
    names(cond) = c(cell_type)
    
    SPARSim_sim_param <- SPARSim_estimate_parameter_from_data(
      raw_data = as.matrix(ordered_raw_counts),
      norm_data = as.matrix(ordered_cells_counts),
      conditions = cond
    )
    
    # Save a particular repeat's population profile
    saveRDS(SPARSim_sim_param, paste0(save_path, "sparsim_params_", t, ".Rds"))
    
    for(k in 1:length(markers_list)){
      fc_index = markers_list[[k]]
      # Loop through all the lower and upper bounds
      for(j in 1:length(de_params)){
        min_val = de_params[[j]][1]
        max_val = de_params[[j]][2]
        
        celltype_pop = SPARSim_sim_param[[1]]
        # Build the fc multipliers where first 10 genes in prescribed values and rest to [1,4]
        cell_fc_multiplier <- rep(1, length(rownames(ordered_cells_counts)))
        cell_fc_multiplier[fc_index] = runif(N_de, min_val, max_val)
        id = paste0('repeat_', t, '_group_', k)
        # Identifier
        min_val = as.integer(min_val)
        # Generate the parameters for new cells
        new_cell_type <- SPARSim_create_DE_genes_parameter(
          sim_param = celltype_pop,
          fc_multiplier = cell_fc_multiplier,
          N_cells = length(SPARSim_sim_param[[1]]$lib_size),
          condition_name = paste0(cell_type, "_", id)
        )
        
        # Two conditions: origin population vs newly generated
        conds = list(SPARSim_sim_param[[1]], new_cell_type)
        # colnames
        names(conds) = c(allowed_cells[1], paste0(cell_type, "_", id))
        # Generate the 10000 cells
        SPARSim_results <- SPARSim_simulation(conds)
        
        # Save gene expression matricies and other files
        m <- as.matrix(SPARSim_results$count_matrix)
        n_cols <- dim(m)[2]
        Ndata = NormalizeData(object = m, normalization.method = "LogNormalize")
        if(k == 1){
          writeMM(Ndata[, 1:  (n_cols/2)], paste0(save_path, "matrix_base_", id ,'.txt'))
        }
        writeMM(Ndata[, (n_cols/2+1): n_cols], paste0(save_path, "matrix_sim_", id ,'.txt'))
      }
    }
    
  }
}



ncores <- 8
cl <- makeCluster(ncores)
cell_types = c("CD14 Monocyte", "NK", "CD4n T", "B", "DC", "CD8m T", "CD4m T", "CD16 Monocyte", "pDC", "CD8eff T", "Platelet", "Neutrophil", "IgG PB", "IgA PB", "Activated Granulocyte", "SC & Eosinophil", "RBC")
clusterApply(cl, cell_types, fun=generateDE)