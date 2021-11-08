library(parallel)

performDE = function(cell_type){
  library(stringr)
  library(limma)
  library(MAST)

  
  simulated_path = paste0('combination_matrix_gamma/', cell_type)
  nn_path = paste0('combination_binary_gamma/', cell_type) # change
  save_path = paste0('de_analysis_combination_binary_gamma/', cell_type) # change
  
  gene_names = read.csv(paste0(simulated_path, '/', 'geneNames.csv'))$gene_names
  
  de_methods = c("wilcox", "MAST", "bimod", "t", "LR")
  
  
  experiments = data.frame(str_match(Sys.glob(paste0(nn_path, '/NumCells_*/predictions.csv')), "(NumCells_.*)/predictions.csv"))
  for(i in 1:nrow(experiments)){
    text = experiments[i,]$X1
    if(!grepl("repeat", text, fixed=TRUE)){
        next
    }
    predictions = as.factor(as.character(read.csv(paste0(nn_path, '/', text))$predictions))
    suffix = experiments[i,]$X2
    matrix_orig = as.matrix(Matrix::readMM(paste0(simulated_path, '/matrix_base_', suffix)))
    matrix_group1 = as.matrix(Matrix::readMM(paste0(simulated_path, '/matrix_sim_', suffix)))
    combined_matrix <- as(cbind(matrix_orig, matrix_group1), "sparseMatrix")
    
    reps = dim(matrix_orig)[2]
    N = dim(combined_matrix)[2]
    rownames(combined_matrix) = gene_names
    colnames(combined_matrix) = as.character(1:N)
    exprsMat = as.matrix(combined_matrix)
    
    for(group in 1:2){
      
      
      tmp_celltype <- ifelse(predictions == group-1, 1, 0)
      design <- stats::model.matrix(~tmp_celltype)
      y <- methods::new("EList")
      y$E <- combined_matrix
      fit <- limma::lmFit(y, design = design)
      fit <- limma::eBayes(fit, trend = TRUE, robust = TRUE)
      topTable <- limma::topTable(fit, n = Inf, adjust.method = "BH", coef = 2, sort.by = 'logFC')
      topTable <- topTable[rownames(topTable)[order(topTable$adj.P.Val)],]
      limma_ranks = rownames(topTable)
      
      cdr <- scale(colMeans(exprsMat > 0))
      sca <- FromMatrix(exprsArray = exprsMat, cData = data.frame(wellKey = colnames(exprsMat), grp = tmp_celltype, cdr = cdr))
      zlmdata <- zlm(~cdr + grp, sca)
      mast <- lrTest(zlmdata, "grp")
      df <- data.frame(pval = mast[, "hurdle", "Pr(>Chisq)"],
                       lambda = mast[, "cont", "lambda"],
                       row.names = names(mast[, "hurdle", "Pr(>Chisq)"]))
      df$fdr <- stats::p.adjust(df$pval, method="BH")
      df = df[order(df$fdr), ]
      mast_ranks = rownames(df)
      
      tt = t(apply(exprsMat, 1, function(x) {
        
        res <- stats::wilcox.test(x ~ tmp_celltype)
        c(stats=res$statistic,
          pvalue=res$p.value)
      }))
      tt = data.frame(tt)
      tt$adj.pvalue <- stats::p.adjust(tt$pvalue, method = "BH")
      tt = tt[order(tt$adj.pvalue), ]
      wilcoxon_ranks = rownames(tt)
      
      data = data.frame(limma=limma_ranks, MAST = mast_ranks, wilcoxon = wilcoxon_ranks)
      save = paste0(save_path, '/', suffix, '/')
      if (!dir.exists(save)){
        dir.create(save, recursive = TRUE)
      } else {
        print("Dir already exists!")
      }
      write.csv(data, paste0(save, "rankings_group_", group , ".csv") )
      
    }
  }
  
}

#ncores <- 17
#cl <- makeCluster(ncores)
#cell_types = c("CD14 Monocyte", "NK", "CD4n T", "B", "DC", "CD8m T", "CD4m T", "CD16 Monocyte", "pDC", "CD8eff T", "Platelet", "Neutrophil", "IgG PB", "IgA PB", "Activated Granulocyte", "SC & Eosinophil", "RBC")
#clusterApply(cl, cell_types, fun=performDE)

#ncores <- 2
#cl <- makeCluster(ncores)
#cell_types = c("CD4n T", "B")
#clusterApply(cl, cell_types, fun=performDE)
#performDE("B")
performDE("CD4n T")