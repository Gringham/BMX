# BMX - Boosting Natural Language Generation Metrics with Explainability

This repository contains code for our paper - BMX: Boosting Natural Language Generation with Explainability.
If you use it, please cite: 

```
@inproceedings{leiter-etal-2024-bmx,
    title = "{BMX}: Boosting Natural Language Generation Metrics with Explainability",
    author = "Leiter, Christoph  and
      Nguyen, Hoa  and
      Eger, Steffen",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.150",
    pages = "2274--2288",
}

```

To run the experiments, please follow the following steps:

- Prepare the datasets, by downloading the respective data and running the preparation scripts in xaiMetrics/data
- Create xaiMetrics/outputs/experiment_graphs, xaiMetrics/outputs/experiment_graphs_pdf, 
  xaiMetrics/outputs/experiment_graphs_pdf_stratification, xaiMetrics/outputs/experiment_results, 
  xaiMetrics/outputs/experiment_results_stratification, xaiMetrics/outputs/Images_Paper_Auto_Gen, 
  xaiMetrics/outputs/raw_explanations, xaiMetrics/outputs/sys_level_tables
- Follow the description of the README.md file in xaiMetrics/experiments

This code is structured as follows:
xaimetrics /  
data - should contain tsv files with the data we want to compute on. Helper Scripts and their comments help in building these corpora  
evalTools - loops to apply explanations on the data and powermeans on explanations  
experiments - scripts to run our experiments. You can follow the settings to run with own data  
explainer - explainer code  
explanations - explanation objects to allow some functions being run on them directly  
metrics - metrics code  
outputs - folder for outputs