import os
from statistics import mean

import scipy

from project_root import ROOT_DIR
from xaiMetrics.constants import REFERENCE_BASED
from xaiMetrics.metrics.wrappers.BARTScore import BARTScore

import pandas as pd

# Compare BARTScore fine-tuned on split 1 to human scores

b = BARTScore(mode=REFERENCE_BASED, batch_size=16, use_pth=False,#
                             checkpoint=os.path.join(ROOT_DIR, "xaiMetrics", "metrics", "wrappers", "source",
                                                     "BARTScore",
                                                     "train", "summ_eval_bartscore3"))
print("Calculating regression")


summ_df = pd.read_json(
        os.path.join(ROOT_DIR, "xaiMetrics", "data", "cnndm", "SummEval.json")) [208:]

print("Calculating scores")
scores = b.evaluate_df(summ_df)

import json
#with open('XLMR.json', 'w') as f:
#    json.dump(scores1, f)
with open('XLMRLG.json', 'w') as f:
    json.dump(scores, f)

#import json
#with open('CustomBARTScoreScores.json', 'r') as f:
#    scores = json.load(f)

#print(scores1, scores2[0], scores2[1], scores2[2], scores2[3])

hyp = summ_df["HYP"].tolist()
ref = summ_df["REF"].tolist()
sys = summ_df["model_id"].tolist()
consistency = [d["consistency"] for d in summ_df["expert_avg"].to_list()]
coherence = [d["coherence"] for d in summ_df["expert_avg"].to_list()]
fluency = [d["fluency"] for d in summ_df["expert_avg"].to_list()]
relevance = [d["relevance"] for d in summ_df["expert_avg"].to_list()]



def print_correlation(sys, scores, human):
    sys_dict = {m: ([], []) for m in set(sys)}
    for m, s, h in zip(sys, scores, human):
        sys_dict[m][0].append(s)
        sys_dict[m][1].append(h)

    s_avg = []
    h_avg = []
    for k, v in sys_dict.items():
        s_avg.append(sum(v[0]) / len(v[0]))
        h_avg.append(sum(v[1]) / len(v[1]))

    t, pt = scipy.stats.kendalltau(s_avg, h_avg)
    s, ps = scipy.stats.spearmanr(s_avg, h_avg)
    return (t, s)

t1, s1 = print_correlation(sys, scores, coherence)
t2, s2 = print_correlation(sys, scores, consistency)
t3, s3 = print_correlation(sys, scores, fluency)
t4, s4 = print_correlation(sys, scores, relevance)

#t5, s5 = print_correlation(sys, scores2[0], coherence)
#t6, s6 = print_correlation(sys, scores2[1], consistency)
#t7, s7 = print_correlation(sys, scores2[2], fluency)
#t8, s8 = print_correlation(sys, scores2[3], relevance)

print(t1, t2, t3, t4, mean([t1, t2, t3, t4]))
print(s1, s2, s3, s4, mean([s1, s2, s3, s4]))

#print(t5, t6, t7, t8, mean([t5, t6, t7, t8]))
#print(s5, s6, s7, s8, mean([s5, s6, s7, s8]))