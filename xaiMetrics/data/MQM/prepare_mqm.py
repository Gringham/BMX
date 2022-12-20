import pandas as pd
import numpy as np
import os
import csv
from project_root import ROOT_DIR

# A reader for the 2021 newstest mqm annotations by:
# Markus Freitag, George Foster, David Grangier, Viresh Ratnakar, Qĳun Tan, & Wolfgang Macherey. (2021).
# Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation.

# Data can be downloaded from here: https://github.com/google/wmt-mqm-human-evaluation



def load_data(path, lp):
    short_rep = lp.replace("-", "")
    seg_avg_df = pd.read_csv(os.path.join(path, "mqm_newstest2021_{}.avg_seg_scores.tsv").format(short_rep),
                             sep = ' |\t')
    seg_avg_df = seg_avg_df[seg_avg_df["mqm_avg_score"]!="None"]

    ratings_df = pd.read_csv(os.path.join(path, "mqm_newstest2021_{}.tsv").format(short_rep),
                             delimiter='\t', encoding="utf-8", quoting=csv.QUOTE_NONE)


    ratings_df = ratings_df.sort_values(by=["seg_id"])
    ratings_df['target'] = ratings_df['target'].map(lambda x: x.replace("<v>","").replace("</v>",""))
    ratings_df['system'] = ratings_df['system'].map(lambda x: x.replace("hyp.", "").replace("ref.", "ref-"))
    ratings_df = ratings_df[["system","seg_id", "source", "target"]].drop_duplicates()
    combined_df = pd.merge(seg_avg_df, ratings_df,  how='left', left_on=["system","seg_id"], right_on = ["system","seg_id"])

    # Its not DA, but for simplicity
    combined_df = combined_df.rename(columns={"target": "HYP", "source": "SRC", "mqm_avg_score": "DA"})
    return combined_df


if __name__ == '__main__':
    # To run change all occurrences of the lp to the lp you want to have
    lps = ["en-de", "zh-en"]

    for lp in lps:
        path = os.path.join(ROOT_DIR, 'xaiMetrics', 'data', 'MQM')

        df = load_data(path, lp)
        df['LP'] = [lp] * len(df)
        df['REF'] = 'dummy'
        df['SYSTEM'] = 'dummy'

        df.to_csv(os.path.join(ROOT_DIR,'xaiMetrics','data','MQM','mqm21_{lp}.tsv'.format(lp=lp)), sep='\t')
