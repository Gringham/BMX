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

### Second iteration
eval4nlp21_test_second_iteration.py

### Held datasets
mqm21_test_pmeans.py
wmt22_test_setl_test_pmeans.py

### Run analysis and create plots
summarize_progress.py  

Before this can be run, the explanations and correlations of the datasets 
have to be computed. We include the correlations from our experiments.
