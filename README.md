# BMX - Boosting Machine Translation Metrics with Explainability

This repository contains code for our paper - BMX: Boosting Machine Translation Metrics with Explainability.
If you use it, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2212.10469,
  doi = {10.48550/ARXIV.2212.10469},
  
  url = {https://arxiv.org/abs/2212.10469},
  
  author = {Leiter, Christoph and Nguyen, Hoa and Eger, Steffen},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {BMX: Boosting Machine Translation Metrics with Explainability},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}

```

To run the experiments, please follow the following steps:

* Prepare the datasets, by downloading the respective data and running the preparation scripts in xaiMetrics/data
* Follow the description of the README.md file in xaiMetrics/experiments


This code is structured as follows:

xaimetrics /   
data - should contain tsv files with the data we want to compute on. Helper Scripts and their comments help in building these corpora  
evalTools - loops to apply explanations on the data and powermeans on explanations  
experiments - scripts to run our experiments. You can follow the settings to run with own data  
explainer - explainer code  
explanations - explanation objects to allow some functions being run on them directly  
metrics - metrics code  
outputs - folder for outputs  

The releases contain files with all our generated explanations. To run further evaluations, these shoud be put in the outputs folder.   


