This directory contains code to reproduce the experiments of our paper or to start new experiments.
For details, please view each files comments. Configuration of the language pairs, explainers and metrics, is usually done directly in the function definition.

Once the test data in `xaiMetrics/data` and output folders are prepared, you should be able to simply
start these scripts to generate new explanations, graphs and correlations in `xaiMetrics/outputs`

### Evaluation on calibration and evaluation datasets
The scripts in `dataset_configurations` will produce scores for a grid search across the respective filenames. The 
grid search goes over all configured configurations. We compute all scores for calibration and evaluation sets, however, 
for the evaluation sets we strictly evaluate fixed hyperparameters following the hyper-parameter selection outlined in the 
paper (in the scripts `compute_results_<>.py`). Depending on the configuration in each file, correlations will either be 
computed here or also in the `compute_results_<>.py` files. 

### Run analysis and create plots
The `compute_results_<>.py` scripts will create the plots displayed in our paper as well as plots and scores for all 
parameters our grid search is iterating over. 

The `test_custom_<>.py` scripts evaluate the scores with BMX compared to other fine-tuning options.

For further details, you can view the comments in the files. 
