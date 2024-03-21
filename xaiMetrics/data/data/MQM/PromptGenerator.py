import pandas as pd

en_de = pd.read_csv("mqm21_en-de.tsv", delimiter="\t")
en_de = en_de.sample(frac=1, random_state=1).reset_index(drop=True)

samples = en_de[0:4]
rest = en_de[50:100]

print("Consider the following examples:")
for r in samples.iterrows():
    row = r[1]
    print(row.to_string())

#print(
#    "Each sample consists of a source sentence SRC that was translated into a target sentence TGT. Each binary label in TAGS_HYP indicates whether a word at that position in HYP is erroneous (1) or not (0). TAGS_TGT describes the same for TGT. SCORE shows the general quality of the translation.")
#print("Please generate correct values for SRC_TAGS, TGT_TAGS and SCORE for the following sentence pairs:\n")



rest.to_csv("chatGPTEval.tsv", sep="\t")
cnt = 0

for r in rest.iterrows():
    if cnt % 5 == 0:
        print("--------------------------")
        print("Consider the following sentence pairs of a source sentence (SRC) and its translation (TGT):")
    row = r[1]
    print("SRC: {SRC}\nTGT: {TGT}\n".format(SRC=row["SRC"], TGT=row["HYP"]))
    if (cnt +1) % 5 == 0:
        print("Please grade each translation with a score between 0 and 1, describe in a structured way why it is a bad translation and give a confidence for your answer.")
    cnt +=1


# Add original target annoations

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

    ratings_df = pd.read_csv(os.path.join(path, "mqm_newstest2021_{}.tsv").format(short_rep),
                             delimiter='\t', encoding="utf-8", quoting=csv.QUOTE_NONE)


    ratings_df = ratings_df.sort_values(by=["seg_id"])
    ratings_df['system'] = ratings_df['system'].map(lambda x: x.replace("hyp.", "").replace("ref.", "ref-"))
    ratings_df = ratings_df[["system","seg_id", "source", "target"]].drop_duplicates()

    # Its not DA, but for simplicity
    combined_df = ratings_df.rename(columns={"target": "HYP", "source": "SRC", "mqm_avg_score": "DA"})
    return combined_df



lps = ["en-de"]

for lp in lps:
    path = os.path.join(ROOT_DIR, 'xaiMetrics', 'data', 'MQM')

    df = load_data(path, lp)

    ids = rest["seg_id"].tolist()

    combined_df = pd.merge(rest, df,  how='left', left_on=["system","seg_id"], right_on = ["system","seg_id"])
combined_df.to_csv("chatGPTEvalComb.tsv", sep="\t", encoding="utf-8")

