import pandas as pd
import os
import csv
from project_root import ROOT_DIR

# Script that can read the mlqe-pe corpus used as train set in the wmt22 QE shared task
# We downloaded the 2020 files here: https://github.com/WMT-QE-Task/wmt-qe-2022-data/tree/main/train-dev_data/task1_da/train
#Fomicheva, M., Sun, S., Fonseca, E., Blain, F., Chaudhary, V., Guzmán, F., … F. T. Martins, A. (2020).
# MLQE-PE: A Multilingual Quality Estimation and Post-Editing Dataset. ArXiv Preprint ArXiv:2010. 04480.



def load_data(path, type='test20', lp="ende"):
    src = pd.read_csv(os.path.join(path, type+'.src'),
                     delimiter='\t', encoding='utf-8', names=['SRC'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    hyp = pd.read_csv(os.path.join(path, type+'.mt'),
                       delimiter='\t', encoding='utf-8', names=['HYP'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    sentence_data = pd.read_csv(os.path.join(path, type+'.'+lp+'.df.short.tsv'),
                          delimiter='\t', encoding='utf-8', error_bad_lines=False, quoting=csv.QUOTE_NONE)
    sentence_data = sentence_data.rename({'z_mean': 'DA'}, axis=1)

    return pd.concat([src, hyp, sentence_data], axis=1)

if __name__ == '__main__':
    # To run change all occurences of the lp to the lp you want to have
    lps = ["en-de", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en"]

    for lp in lps:
        path = os.path.join(ROOT_DIR,'xaiMetrics','data','wmt22_expl','{lp}-test20'.format(lp=lp))

        df = load_data(path, "test20", lp.replace("-",""))
        df['LP'] = [lp]*len(df)
        df['REF'] = 'dummy'
        df['SYSTEM'] = 'dummy'


        df.to_csv(os.path.join(ROOT_DIR,'xaiMetrics','data','wmt22_expl','eval4nlp_test_{lp}.tsv'.format(lp=lp)), sep='\t')