# Honours code

This contains parts of the codebase used for the honours thesis: `Interpretable deep learning on single-cell COVID-19 data`. A minimal amount of code is presented to create the simulations, training the neural network and intrepret it.

## Population parameters of cell-types

Running this R script allows for 'frontrunning' estimation of cell-type simulaiton parameters so we can exclude highly-variable genes.

```bash
Rscript generatePopulationParametersParallel.R
```

## Simulations

Running these R scripts will generate the simulations for all 3 experiments.

```bash
Rscript combinationSimulatedMatricesParallelTwoSideGamma.R # Experiment 1
Rscript combinationSimulatedMatricesParallelTwoSideGammaMulticlassExclusive.R # Experiment 2
Rscript combinationSimulatedMatricesParallelTwoSideGammaMulticlassShared.R # Experiment 3
```

## Neural network training

Running these python scripts will train the cell-type prediction neural networks

```bash
python3 new_python_nn_train_only_one_sided_de_binary.py # Experiment 1
python3 new_python_nn_train_only_one_sided_de_binary_gamma_multiclass_exclude.py # Experiment 2
python3 new_python_nn_train_only_one_sided_de_binary_gamma_multiclass_shared.py # Experiment 3
```

## Saliency evaluation framework

Running these python scripts will use saliency methods to interpret the important genes

```bash
python3 new_python_nn_run_saliency_one_sided_binary_gamma_all_methods.py # Experiment 1
python3 new_python_nn_run_saliency_one_sided_binary_gamma_multiclass_exclude_all_methods.py # Experiment 2
python3 new_python_nn_run_saliency_one_sided_binary_gamma_multiclass_shared_all_methods.py # Experiment 3
```

## DE method analysis

Running these R scripts will run DE methods (limma, MAST, wilcoxon) to interpret the genes

```bash
Rscript new_de_analysis_one_sided_de_binary_gamma.R # Experiment 1
Rscript new_de_analysis_one_sided_de_binary_gamma_multiclass_exclude.R # Experiment 2
Rscript new_de_analysis_one_sided_de_binary_gamma_multiclass_shared.R # Experiment 3 
```
