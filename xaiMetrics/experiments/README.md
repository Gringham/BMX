This directory contains code to reproduce the experiments of our paper or to start new experiments.
For details, please view each files comments. Configuration of the language pairs, explainers and metrics, is usually #
done directly in the function definition.

One the test data in xaiMetrics/data is prepared, you should be able to simply
start these scripts to generate new explanations, graphs and correlations in
xaiMetrics/outputs

### Evaluation on the three main datasets
eval4nlp21_test_pmeans.py
wmt_17_test_pmeans.py
wmt_22_expl_test_pmeans.py (MLQE-PE)

This evaluation usually follows these steps (see resp. files):

1. define the language pairs to run on
2. define the explainers to explain with
3. define the metrics to run with
4. load a dataset as pandas dataframe (usually for each lamguage pair)
5. run `apply_explainers_on_metrics` with the defined components to apply explanations for all specified components and save them as dill files
6. Load the explanations from dill files (extra step, as we might want to reuse old explanations), and initialise a `FeatureImportanceExplanationCollection` object per file. This object has functionalities to copmute powermeans on the explanations it contains.
7. Run `explanations_to_scores_and_eval` to apply the powermean on specified `FeatureImportanceExplanationCollection` objects, and evaluate correlation to a specified  dataset

### Second iteration
eval4nlp21_test_second_iteration.py

### Held datasets
mqm21_test_pmeans.py
wmt22_test_setl_test_pmeans.py

### Run analysis and create plots
summarize_progress.py  

Before this can be run, the explanations and correlations of the datasets 
have to be computed. We include the correlations from our experiments.
