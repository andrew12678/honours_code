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
  
  save_directory = "/albona/nobackup/andrewl/honours_code/combination_matrix_gamma_multiclass_exclude/"

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
  q_limit = quantile(m_SPARSim_sim_param[[1]]$variability, 0.90, na.rm=TRUE)
  # intensity = m_SPARSim_sim_param[[1]]$intensity
  remove = as.vector(which(m_SPARSim_sim_param[[1]]$variability >= q_limit))
  ordered_raw_counts = ordered_raw_counts[-remove, ]
  ordered_cells_counts = ordered_cells_counts[-remove, ]
  
  new_gene_names_df = data.frame(gene_names=rownames(ordered_cells_counts))
  write.csv(new_gene_names_df, paste0(save_path, "geneNames.csv"))
  
  total_matrix <- as.vector(which(celltypes_order == cell_type))
  

  # Different fc multipler bounds
  # de_params = list(c(1.2, 3), c(1.5, 5))
  # N_de_cand = c(100, 1000)
  # N_cells = c(100, 1000)
  de_params = list(c(1.2, 3))
  N_de_cand = c(100)
  N_cells = c(1000)
  gene_total = nrow(ordered_cells_counts)
  range = 1:gene_total
  p.lfc <- function(x) sample(c(-1,1), size=x,replace=T)*rgamma(x, shape = 3, rate = 2)
  par(mfrow=c(3,3))
  
  for(t in 1:10){ # Repeats
    for(cells in N_cells){
      for(N_de in N_de_cand){
        for(j in 1:length(de_params)){
          
          count <- min(cells, length(total_matrix))
          sample_matrix <- sample(total_matrix, count)
          cond <- list(sample_matrix)
          names(cond) = c(cell_type)
          
          SPARSim_sim_param <- SPARSim_estimate_parameter_from_data(
            raw_data = as.matrix(ordered_raw_counts),
            norm_data = as.matrix(ordered_cells_counts),
            conditions = cond
          )
          
          min_val = de_params[[j]][1]
          max_val = de_params[[j]][2]
          celltype_pop = SPARSim_sim_param[[1]]
          # Build the fc multipliers where first 10 genes in prescribed values and rest to [1,4]
          cell_fc_multiplier <- rep(1, length(rownames(ordered_cells_counts)))
          cell_fc_multiplier2 <- rep(1, length(rownames(ordered_cells_counts)))
          
          fc_index = sample(range, N_de) # Original population
          fc_index2 = sample(setdiff(range, fc_index), N_de)
            
          values = p.lfc(N_de)
          final = 2^values
          final[final > max_val] = max_val
          final[(final < min_val) & (final > 1)] = min_val
          min_small = 1/max_val
          max_small = 1/min_val
          final[final < min_small] = min_small
          final[(final > max_small) & (final < 1)] = max_small
            
          values2 = p.lfc(N_de)
          final2 = 2^values2
          final2[final2 > max_val] = max_val
          final2[(final2 < min_val) & (final2 > 1)] = min_val
          final2[final2 < min_small] = min_small
          final2[(final2 > max_small) & (final2 < 1)] = max_small
          
          cell_fc_multiplier[fc_index] = final
          id = paste0('NumCells_', cells, '_NumDE_', N_de, '_min_lfc_', min_val, '_max_lfc_', max_val, '_repeat_', t)
          write.csv(data.frame(gene_index = fc_index), paste0(save_path, id, '_de_genes1.csv'))
            
          cell_fc_multiplier2[fc_index2] = final2
          write.csv(data.frame(gene_index = fc_index2), paste0(save_path, id, '_de_genes2.csv'))

          new_cell_type <- SPARSim_create_DE_genes_parameter(
            sim_param = celltype_pop,
            fc_multiplier = cell_fc_multiplier,
            N_cells = length(SPARSim_sim_param[[1]]$lib_size),
            condition_name = paste0(cell_type, "_", id)
          )
          new_cell_type2 <- SPARSim_create_DE_genes_parameter(
            sim_param = celltype_pop,
            fc_multiplier = cell_fc_multiplier2,
            N_cells = length(SPARSim_sim_param[[1]]$lib_size),
            condition_name = paste0(cell_type, "_", id)
          )
          
          # Two conditions: origin population vs newly generated
          conds = list(SPARSim_sim_param[[1]], new_cell_type)
          conds2 = list(SPARSim_sim_param[[1]], new_cell_type2)
          # colnames
          names(conds) = c(allowed_cells[1], paste0(cell_type, "_", id))
          names(conds2) = c(allowed_cells[1], paste0(cell_type, "_", id))  
          # Generate the 10000 cells
          SPARSim_results <- SPARSim_simulation(conds)
          SPARSim_results2 <- SPARSim_simulation(conds2)  
          
          # Save gene expression matricies and other files
          m <- as.matrix(SPARSim_results$count_matrix)
          m2 <-   as.matrix(SPARSim_results2$count_matrix)
          n_cols <- dim(m)[2]
          Ndata = NormalizeData(object = m, normalization.method = "LogNormalize")
          Ndata2 = NormalizeData(object = m2, normalization.method = "LogNormalize")  
          writeMM(Ndata[, 1:  (n_cols/2)], paste0(save_path, "matrix_base_", id ,'.txt'))
          writeMM(Ndata[, (n_cols/2+1): n_cols], paste0(save_path, "matrix_sim1_", id ,'.txt'))
          writeMM(Ndata2[, (n_cols/2+1): n_cols], paste0(save_path, "matrix_sim2_", id ,'.txt'))  
        }
      }
    }
  }
}


ncores <- 2
cl <- makeCluster(ncores)
cell_types = c("CD4n T", "B")
clusterApply(cl, cell_types, fun=generateDE)
#generateDE("B")
#generateDE("CD4n T")