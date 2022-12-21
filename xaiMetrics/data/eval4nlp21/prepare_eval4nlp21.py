import pandas as pd
import os
import csv
from project_root import ROOT_DIR

# Data can be downloaded from here: https://github.com/eval4nlp/SharedTask2021
# The reader is specifically for the testdata

# Dataset from:
# Fomicheva, M., Lertvittayakumjorn, P., Zhao, W., Eger, S., & Gao, Y. (2021).
# The Eval4NLP Shared Task on Explainable Quality Estimation: Overview and Results.
# In Proceedings of the 2nd Workshop on Evaluation and Comparison of NLP Systems
# (pp. 165–178). Association for Computational Linguistics.

def load_data(path, type='test21'):
    # Read the different files provided into single pandas df's then concatenate them
    tgt_tags = pd.read_csv(os.path.join(path, type+'.tgt-tags'),
                      delimiter='\t', encoding='utf-8', names=['TAGS_HYP'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    src_tags = pd.read_csv(os.path.join(path, type+'.src-tags'),
                           delimiter='\t', encoding='utf-8', names=['TAGS_SRC'], error_bad_lines=False,
                           quoting=csv.QUOTE_NONE)
    src = pd.read_csv(os.path.join(path, type+'.src'),
                     delimiter='\t', encoding='utf-8', names=['SRC'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    hyp = pd.read_csv(os.path.join(path, type+'.mt'),
                       delimiter='\t', encoding='utf-8', names=['HYP'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    zscores = pd.read_csv(os.path.join(path, type+'.da'),
                          delimiter='\t', encoding='utf-8', names=['DA'], error_bad_lines=False, quoting=csv.QUOTE_NONE)
    return pd.concat([src, hyp, zscores, tgt_tags, src_tags], axis=1)

if __name__ == '__main__':
    # To run change all occurences of the lp to the lp you want to have
    lp = 'ro-en'
    path = os.path.join(ROOT_DIR,'xaiMetrics','data','eval4nlp21','{lp}-test21'.format(lp=lp))

    df = load_data(path)
    df['LP'] = [lp]*len(df)
    df['REF'] = 'dummy'
    df['SYSTEM'] = 'dummy'


    df.to_csv(os.path.join(ROOT_DIR,'xaiMetrics','data','eval4nlp21','eval4nlp_test_{lp}.tsv'.format(lp=lp)), sep='\t')
