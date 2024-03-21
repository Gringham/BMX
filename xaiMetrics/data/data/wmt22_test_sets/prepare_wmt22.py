import pandas as pd
import os
import csv
from project_root import ROOT_DIR

# Script that can read test data of the wmt qe shared task 2022
# Data was downloaded from here: https://github.com/WMT-QE-Task/wmt-qe-2022-data/tree/main/test_data-gold_labels/task1_da

# Zerva, C., Blain, F., Rei, R., Lertvittayakumjorn, P., de Souza, J. G. C., Eger, S., … Specia, L. (2022, December).
# Findings of the WMT 2022 Shared Task on Quality Estimation.
# Proceedings of the Seventh Conference on Machine Translation. Abu Dhabi: Association for Computational Linguistics.


def load_data(path, type='test.2022', lp="en-de"):
    src = pd.read_csv(os.path.join(path, type+'.src'),
                     delimiter='\t', encoding='utf-8', names=['SRC'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    hyp = pd.read_csv(os.path.join(path, type+'.mt'),
                       delimiter='\t', encoding='utf-8', names=['HYP'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    sentence_data = pd.read_csv(os.path.join(path, type+'.'+lp+'.da_score'),
                          delimiter='\t', encoding='utf-8', names=['DA'], error_bad_lines=False, quoting=csv.QUOTE_NONE)

    return pd.concat([src, hyp, sentence_data], axis=1)

if __name__ == '__main__':
    # To run change all occurrences of the lp to the lp you want to have
    lps = ["en-cs", "en-ja", "en-mr", "en-yo", "km-en", "ps-en"]

    for lp in lps:
        path = os.path.join(ROOT_DIR,'xaiMetrics','data','wmt22_test_sets',lp)

        df = load_data(path, "test.2022", lp)
        df['LP'] = [lp]*len(df)
        df['REF'] = 'dummy'
        df['SYSTEM'] = 'dummy'


        df.to_csv(os.path.join(ROOT_DIR,'xaiMetrics','data','wmt22_test_sets','wmt22_test_{lp}.tsv'.format(lp=lp)), sep='\t')