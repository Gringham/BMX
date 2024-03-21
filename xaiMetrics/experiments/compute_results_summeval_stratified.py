import json
import math
from statistics import mean

from project_root import ROOT_DIR
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import linear_model
import nlpstats.correlations.correlations

# Evaluation script for the stratified result files of SummEval and RealSumm

def find_best_p_and_w_alternative(strat_number=-1, realsumm = False, first=False):
    # Load stratification splits by split number. The number will already be added to the filetags correctly when using
    # the stratification method of explanations_to_scores.py
    sns.set_theme()
    explainer_names = ["LimeExplainer"]
    alt_explainer_names = ["LIME"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "BARTScore"]

    if not realsumm:
        config_dict = {
            "SummEval-coherence": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
            "SummEval-consistency": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
            "SummEval-fluency": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
            "SummEval-relevance": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
            "SummEval-coherence-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
            "SummEval-consistency-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
            "SummEval-fluency-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
            "SummEval-relevance-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based
            },
        }

    elif first:
        config_dict = {
            "SummEvalkendall-coherence-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            "SummEvalkendall-consistency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-KD"
            },
            "SummEvalkendall-fluency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-KD"
            },
            "SummEvalkendall-relevance-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-KD"
            },
            "SummEvalspearman-coherence-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-SP"
            },
            "SummEvalspearman-consistency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-SP"
            },
            "SummEvalspearman-fluency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-SP"
            },
            "SummEvalspearman-relevance-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-SP"
            },
        }

    else:
        config_dict = {
            "realsumm": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            "realsumm-pearson": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            }}


    lp_l = []
    expl_l = []
    metric_l = []
    diff_l = []
    p_l = []
    w_l = []
    orig_l = []
    max_l = []
    tag_l = []

    for tag in config_dict.keys():
        lps = config_dict[tag]["lps"]
        explainers = config_dict[tag]["explainers"]
        metrics = config_dict[tag]["metrics"]

        for lp in lps:
            for explainer in explainers:
                for metric in metrics:
                    # Load df with resulting correlations for w and p values
                    filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results_stratification",
                                            "__".join([tag, explainer, metric, lp]) + "_train_" + str(
                                                strat_number) + ".tsv")

                    try:
                        df = pd.read_csv(filepath, delimiter='\t')
                    except Exception as e:
                        print(str(e) + "This is expected for Transquest and COMET with other explainers")
                        continue

                    orig = df[df["w"] == 1]["corr"].iat[0]  # Determine the original correlation
# Maximum correlation for dataset, lp, explainer, metric combination
                    p_best = df[df["corr"] > orig]["p"].tolist()
                    w_best = df[df["corr"] > orig]["w"].tolist()

                    for w, p in zip(w_best,p_best):
                        lp_l.append(lp)
                        tag_l.append(tag)
                        expl_l.append(explainer)
                        metric_l.append(metric)
                        diff_l.append(df[(df["p"]==p)&(df["w"]==w)]-orig)
                        p_l.append(p)
                        w_l.append(w)
                        orig_l.append(orig)

    print("Settings with improvement:", len(lp_l))

    # Build a pandas df of the result values for each metric, explainer, lp combination
    res_df = pd.DataFrame([tag_l, lp_l, expl_l, metric_l, diff_l, p_l, w_l, orig_l, max_l]).transpose()
    res_df.columns = ["tag", "lp", "explainer", "metric", "diff", "p", "w", "orig", "max"]
    res_df = res_df[res_df["orig"] > 0]
    print("Settings with improvement:", len(res_df))

    p_w_per_metric = {}
    for metric_name in ref_based:
        fig, axs = plt.subplots(2, len(explainer_names))
        p_w_dict = {}
        for x, explainer in enumerate(explainer_names):
            explainer_df = res_df[(res_df["explainer"] == explainer) & (res_df["metric"] == metric_name)]
            if len(explainer_df) == 0:
                continue
            _, w_dict = explainer_df["w"].plot.box(ax=axs[0], return_type="both")
            _, p_dict = explainer_df["p"].plot.box(ax=axs[1], return_type="both")
            p_w_dict[explainer] = {"p": round(p_dict['medians'][0].get_ydata()[0], 3),
                                   "w": round(w_dict['medians'][0].get_ydata()[0], 3)}
            axs[0].set_ylim(-0.1, 1.1)
            axs[1].set_ylim(-34, 34)
            axs[0].set_xticklabels([])
            axs[1].set_xticklabels([])
            if x != 0:
                axs[0].set_yticklabels([])
                axs[1].set_yticklabels([])
            else:
                axs[0].set_ylabel("w")
                axs[1].set_ylabel("p")
            axs[0].set_title('Md: ' + str(p_w_dict[explainer]["w"]), fontsize=8)
            axs[1].set_title('Md: ' + str(p_w_dict[explainer]["p"]), fontsize=8)
            axs[0].annotate(alt_explainer_names[x], xy=(0.5, 1), xytext=(0, 20),
                            xycoords='axes fraction', textcoords='offset points',
                            size='medium', ha='center', va='baseline')

        fig.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        #plt.savefig(
        #    "C:\\Users\\USERNAME\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\4_best_p_and_w.pdf")
        plt.show()

        print("Best p and w dict (medians):", p_w_dict)
        p_w_per_metric[metric_name] = p_w_dict

    return p_w_per_metric


def evaluate_p_w_best_fix_2(p_w_dict, w_threshold=0.01, p_threshold=0.01, realsumm=False,
                            strat_number=-1, early_return=False, first=False):
    # Query scores of the test sets using the selected p and w values
    sns.set_theme()
    explainer_names = ["LimeExplainer"]
    alt_explainer_names = ["LIME"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "BARTScore"]

    if not realsumm and not first:
        config_dict = {
            "SummEval-coherence": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            "SummEval-consistency": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-KD"
            },
            "SummEval-fluency": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-KD"
            },
            "SummEval-relevance": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-KD"
            },
            "SummEval-coherence-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-SP"
            },
            "SummEval-consistency-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-SP"
            },
            "SummEval-fluency-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-SP"
            },
            "SummEval-relevance-spearman": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-SP"
            },
        }
    elif first:
        config_dict = {
            "SummEvalkendall-coherence-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            "SummEvalkendall-consistency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-KD"
            },
            "SummEvalkendall-fluency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-KD"
            },
            "SummEvalkendall-relevance-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-KD"
            },
            "SummEvalspearman-coherence-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-SP"
            },
            "SummEvalspearman-consistency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-SP"
            },
            "SummEvalspearman-fluency-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-SP"
            },
            "SummEvalspearman-relevance-first-ref": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-SP"
            },
        }
    else:
        config_dict = {
            "realsumm": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            "realsumm-pearson": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            }
            ,}
    # ------------------- Get dataframe with fixed p and w selection ------------------------
    # Fix w and p values based on the previous analysis can be set as function arguments
    lp_l = []
    expl_l = []
    metric_l = []
    diff_l = []
    orig_l = []
    new_corr = []
    tag_l = []
    perc_l = []

    max_non_fix = []
    max_perc_non_fix = []

    for tag in config_dict.keys():
        lps = config_dict[tag]["lps"]
        explainers = config_dict[tag]["explainers"]
        metrics = config_dict[tag]["metrics"]

        for lp in lps:
            for explainer in explainers:
                for metric in metrics:
                    # Load df with resulting correlations for w and p values
                    #if not realsumm:
                    filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results_stratification",
                                    "__".join([tag, explainer, metric, lp]) + "_test_" + str(
                                        strat_number) + ".tsv")
                    #else:
                    #    filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results",
                    #                            "__".join([tag, explainer, metric, lp]) + ".tsv")

                    try:
                        df = pd.read_csv(filepath, delimiter='\t')
                    except Exception as e:
                        print(str(e) + "This is expected for Transquest and COMET with other explainers")
                        continue

                    try:
                        fix_p = mean([p_w_d[metric][explainer]["p"] for p_w_d in p_w_dict])
                        fix_w = mean([p_w_d[metric][explainer]["w"] for p_w_d in p_w_dict])
                    except:
                        fix_p = p_w_dict[metric][explainer]["p"]
                        fix_w = p_w_dict[metric][explainer]["w"]

                    print("Selected: w:", fix_w, " p: ", fix_p )

                    # find closest to given medians
                    p_condition = abs(
                        df["p"] - min(np.arange(-30, 30, 0.1), key=lambda x: abs(x - fix_p))) < p_threshold
                    w_condition = abs(
                        df["w"] - min([0, 0.2, 0.4, 0.6, 0.8, 1], key=lambda x: abs(x - fix_w))) < w_threshold

                    selected_correlation = df[p_condition & w_condition]["corr"].tolist()[0]
                    orig = df[df["w"] == 1]["corr"].iat[0]  # Determine the original correlation
                    diff = selected_correlation - orig  # Determine the difference between all correlations and the original

                    try:
                        max_non_fix.append(max([(a, b, c) for a, b, c in list(
                            zip(np.array(df["corr"].tolist()) - orig, df["corr"].tolist(),
                                [orig] * len(df["corr"].tolist())))
                                                if c > 0], key=lambda item: item[0]))
                    except:
                        pass
                    max_perc_non_fix.append(
                        max([((s / orig) * 100, orig, s) if s > 0 and orig > 0 else (0, orig, s) for s in
                             df["corr"].tolist()],
                            key=lambda item: item[0]))

                    lp_l.append(lp)
                    tag_l.append(tag)
                    expl_l.append(explainer)
                    metric_l.append(metric)
                    diff_l.append(diff)
                    orig_l.append(orig)
                    new_corr.append(selected_correlation)

                    percentages = []
                    if selected_correlation >= 0 and orig > 0:
                        percentages.append((diff / orig) * 100)
                    else:
                        percentages.append(None)
                    if len(percentages) == 1:
                        percentages = percentages[0]
                    perc_l.append(percentages)

    print("Max list: ", max(max_non_fix, key=lambda x: x[0]), max_non_fix)
    print("Max perc list: ", max(max_perc_non_fix), max_perc_non_fix)
    # Build a pandas df of the result values for each metric, explainer, lp combination
    # Uses fix p and w values from function definition
    fix_df = pd.DataFrame([tag_l, lp_l, expl_l, metric_l, diff_l, orig_l, new_corr, perc_l]).transpose()
    fix_df.columns = ["tag", "lp", "explainer", "metric", "diff", "orig", "new_correlation", "improvement_percent"]
    print(fix_df)
    fix_df["orig"] = fix_df["orig"].astype("float")
    fix_df["new_correlation"] = fix_df["new_correlation"].astype("float")

    # Generate Latex Tables
    alt_names = {"BERTSCORE_REF_FREE_XNLI": "XBERTScore",
                 "XLMRSBERT": "XLMR-SBERT",
                 "Transquest": "TRANSQUEST",
                 "COMET": "COMET",
                 "BERTSCORE_REF_BASED_ROBERTA_DEFAULT": "BERTScore",
                 "SentenceBLEU_REF_BASED": "SentenceBLEU",
                 "BARTScore": "BARTScore",
                 "RandomScore": "RandomScore",
                 "Rouge-1": "Rouge-1",
                 "Rouge-2": "Rouge-2",
                 "Rouge-l": "Rouge-l"}

    print("\\begin{table*}\n\\centering\\small\n\\begin{tabular}{l|" + "".join(
        "c" * len(config_dict[tag]["metrics"]) + "}\\toprule"))
    print("Dataset & " + " & ".join(
        ["\\textbf{" + alt_names[m] + "}" for m in config_dict[tag]["metrics"]]))
    print("\\\\\\midrule")
    scores_per_tag = {tag: [] for tag in config_dict.keys()}
    for tag in config_dict.keys():
        for lp in config_dict[tag]["lps"]:
            line_str = config_dict[tag]["alt"] + " & "
            for metric in config_dict[tag]["metrics"]:
                start = True
                scores = []
                for explainer in config_dict[tag]["explainers"]:
                    selection = fix_df[(fix_df["tag"] == tag) & (fix_df["metric"] == metric) & (fix_df["lp"] == lp) & (
                            fix_df["explainer"] == explainer)]
                    if len(selection) == 0:
                        continue
                    if start:
                        scores.append(round(selection["orig"].tolist()[0], 3))
                        start = False
                    scores.append(round(selection["new_correlation"].tolist()[0], 3))
                max_score_indices = np.argwhere(scores == np.amax(scores)).flatten().tolist()
                if not 0 in max_score_indices:
                    line_str += "\\tablegreen{"
                else:
                    line_str += "\\tablered{"
                for i, score in enumerate(scores):
                    val = str(score)
                    if i in max_score_indices:
                        if (0 in max_score_indices and i == 0) or not 0 in max_score_indices:
                            val = "\\textbf{" + val + "}"
                    val = "$" + val + "$"
                    if i != 0:
                        val = "/" + val
                    line_str += val
                line_str += "} & "
                scores_per_tag[tag].append(scores)
            line_str = line_str[:-2] + "\\\\"
            print(line_str)

    for condition in [("KD", "KD"),("SP", "SP")]:
        avg = []
        for key, value in scores_per_tag.items():
            if condition[1] in config_dict[key]["alt"]:
                avg.append(value)

        if len(avg) > 0:
            avg = [[round(mean([avg[z][i][j] for z in range(len(avg))]),3) for j in range(len(avg[0][i]))] for i in range(len(avg[0]))]

            avg2 = []
            for a in avg:
                max_score_indices = np.argwhere(a == np.amax(a)).flatten().tolist()
                if not 0 in max_score_indices:
                    avg2.append("\\tablegreen{")
                else:
                    avg2.append("\\tablered{")
                for i, score in enumerate(a):
                    val = str(score)
                    if i in max_score_indices:
                        if (0 in max_score_indices and i == 0) or not 0 in max_score_indices:
                            val = "\\textbf{" + val + "}"
                    if i != 0:
                        avg2[-1] += "/"
                    avg2[-1] += "$" + val + "$"
                avg2[-1] += "}"

            line_str = "AVG-" + condition[0] + " & " + " & ".join(avg2) + "\\\\"
            print(line_str)
    print("\\bottomrule\n\\label{Split"+str(strat_number+1)+"}\n\\caption{Split"+str(strat_number+1)+"}\n\\end{tabular}\\end{table*}")
    if early_return:
        return fix_df

def evaluate_p_w_best_fix_3(p_w_dict, w_threshold=0.01, p_threshold=0.01, realsumm=False,
                            strat_number=-1, early_return=False, first=False, type="kendall", level="system"):
    # Query scores of the test sets using the selected p and w values
    sns.set_theme()
    explainer_names = ["LimeExplainer"]
    alt_explainer_names = ["LIME"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "BARTScore"]

    if not realsumm and not first:
        config_dict = {
            "SummEvalkendall-coherence-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            "SummEvalkendall-consistency-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-KD"
            },
            "SummEvalkendall-fluency-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-KD"
            },
            "SummEvalkendall-relevance-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-KD"
            }
        }
    else:
        config_dict = {
            "realsumm": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            }
    # ------------------- Get dataframe with fixed p and w selection ------------------------
    # Fix w and p values based on the previous analysis can be set as function arguments
    lp_l = []
    expl_l = []
    metric_l = []
    diff_l = []
    orig_l = []
    new_corr = []
    tag_l = []
    perc_l = []
    p_values_l = []

    max_non_fix = []
    max_perc_non_fix = []

    for tag in config_dict.keys():
        lps = config_dict[tag]["lps"]
        explainers = config_dict[tag]["explainers"]
        metrics = config_dict[tag]["metrics"]

        for lp in lps:
            for explainer in explainers:
                for metric in metrics:
                    # Load df with resulting correlations for w and p values
                    if not realsumm:
                        filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "sys_level_tables",
                                    "__".join([tag, explainer, metric, lp]) + "_test_" + str(
                                        strat_number) + ".json")
                    else:
                        filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "sys_level_tables",
                                     "__".join([tag, explainer, metric, lp]) + "_sysLevel" + ".json")

                    try:
                        with open(filepath, encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception as e:
                        print(str(e) + "This is expected for Transquest and COMET with other explainers")
                        continue



                    try:
                        fix_p = mean([p_w_d[metric][explainer]["p"] for p_w_d in p_w_dict])
                        fix_w = mean([p_w_d[metric][explainer]["w"] for p_w_d in p_w_dict])
                        fix_p = min(np.arange(-30, 30, 0.1), key=lambda x: abs(x - fix_p))
                        fix_w = min([0, 0.2, 0.4, 0.6, 0.8, 1], key=lambda x: abs(x - fix_w))
                    except:
                        fix_p = p_w_dict[metric][explainer]["p"]
                        fix_w = p_w_dict[metric][explainer]["w"]

                    print("Selected: w:", fix_w, " p: ", fix_p )

                    for i, d in enumerate(data):
                        if math.isclose(list(d.values())[0]["p"], fix_p, abs_tol=0.00001) and math.isclose(list(d.values())[0]["w"],fix_w, abs_tol=0.00001):
                            selected_bmx = dict(sorted(d.items()))
                        if math.isclose(list(d.values())[0]["w"],1.0, abs_tol=0.00001):
                            original = dict(sorted(d.items()))

                    bmx_matrix = np.array([np.array(selected_bmx[sys]["bmx"]) for sys in selected_bmx.keys() ])
                    human1_matrix = np.array([np.array(selected_bmx[sys]["human"]) for sys in selected_bmx.keys()])
                    original_matrix = np.array([np.array(original[sys]["bmx"]) for sys in original.keys()])
                    human2_matrix = np.array([np.array(original[sys]["human"]) for sys in original.keys()])


                    bmx_corr = nlpstats.correlations.correlations.correlate(bmx_matrix, human1_matrix, level, type)
                    orig_corr = nlpstats.correlations.correlations.correlate(original_matrix, human1_matrix, level, type)
                    significance = nlpstats.correlations.permutation.permutation_test(bmx_matrix, original_matrix, human1_matrix,
                                                                       level, type, "both", alternative="greater", n_resamples=9999)
                    p_values_l += [significance.pvalue]

                    print(significance.pvalue)
                    diff = bmx_corr - orig_corr  # Determine the difference between all correlations and the original


                    lp_l.append(lp)
                    tag_l.append(tag)
                    expl_l.append(explainer)
                    metric_l.append(metric)
                    diff_l.append(diff)
                    orig_l.append(orig_corr)
                    new_corr.append(bmx_corr)

    # Build a pandas df of the result values for each metric, explainer, lp combination
    # Uses fix p and w values from function definition
    sign = [True if pv <= 0.05 else False for pv in p_values_l]
    fix_df = pd.DataFrame([tag_l, lp_l, expl_l, metric_l, diff_l, orig_l, new_corr, p_values_l, sign]).transpose()
    fix_df.columns = ["tag", "lp", "explainer", "metric", "diff", "orig", "new_correlation", "p_value", "significant"]
    print(fix_df)
    fix_df["orig"] = fix_df["orig"].astype("float")
    fix_df["new_correlation"] = fix_df["new_correlation"].astype("float")

    # Generate Latex Tables
    alt_names = {"BERTSCORE_REF_FREE_XNLI": "XBERTScore",
                 "XLMRSBERT": "XLMR-SBERT",
                 "Transquest": "TRANSQUEST",
                 "COMET": "COMET",
                 "BERTSCORE_REF_BASED_ROBERTA_DEFAULT": "BERTScore",
                 "SentenceBLEU_REF_BASED": "SentenceBLEU",
                 "BARTScore": "BARTScore",
                 "RandomScore": "RandomScore",
                 "Rouge-1": "Rouge-1",
                 "Rouge-2": "Rouge-2",
                 "Rouge-l": "Rouge-l"}

    print("\\begin{table*}\n\\centering\\small\n\\begin{tabular}{l|" + "".join(
        "c" * len(config_dict[tag]["metrics"]) + "}\\toprule"))
    print("Dataset & " + " & ".join(
        ["\\textbf{" + alt_names[m] + "}" for m in config_dict[tag]["metrics"]]))
    print("\\\\\\midrule")
    scores_per_tag = {tag: [] for tag in config_dict.keys()}
    for tag in config_dict.keys():
        for lp in config_dict[tag]["lps"]:
            line_str = config_dict[tag]["alt"] + " & "
            for metric in config_dict[tag]["metrics"]:
                start = True
                scores = []
                for explainer in config_dict[tag]["explainers"]:
                    selection = fix_df[(fix_df["tag"] == tag) & (fix_df["metric"] == metric) & (fix_df["lp"] == lp) & (
                            fix_df["explainer"] == explainer)]
                    if len(selection) == 0:
                        continue
                    if start:
                        scores.append(round(selection["orig"].tolist()[0], 3))
                        start = False
                    scores.append(round(selection["new_correlation"].tolist()[0], 3))
                max_score_indices = np.argwhere(scores == np.amax(scores)).flatten().tolist()
                if not 0 in max_score_indices:
                    line_str += "\\tablegreen{"
                else:
                    line_str += "\\tablered{"
                for i, score in enumerate(scores):
                    val = str(score)
                    if i in max_score_indices:
                        if (0 in max_score_indices and i == 0) or not 0 in max_score_indices:
                            val = "\\textbf{" + val + "}"
                    val = "$" + val + "$"
                    if i != 0:
                        val = "/" + val
                    line_str += val
                line_str += "} & "
                scores_per_tag[tag].append(scores)
            line_str = line_str[:-2] + "\\\\"
            print(line_str)

    for condition in [("KD", "KD"),("SP", "SP")]:
        avg = []
        for key, value in scores_per_tag.items():
            if condition[1] in config_dict[key]["alt"]:
                avg.append(value)

        if len(avg) > 0:
            avg = [[round(mean([avg[z][i][j] for z in range(len(avg))]),3) for j in range(len(avg[0][i]))] for i in range(len(avg[0]))]

            avg2 = []
            for a in avg:
                max_score_indices = np.argwhere(a == np.amax(a)).flatten().tolist()
                if not 0 in max_score_indices:
                    avg2.append("\\tablegreen{")
                else:
                    avg2.append("\\tablered{")
                for i, score in enumerate(a):
                    val = str(score)
                    if i in max_score_indices:
                        if (0 in max_score_indices and i == 0) or not 0 in max_score_indices:
                            val = "\\textbf{" + val + "}"
                    if i != 0:
                        avg2[-1] += "/"
                    avg2[-1] += "$" + val + "$"
                avg2[-1] += "}"

            line_str = "AVG-" + condition[0] + " & " + " & ".join(avg2) + "\\\\"
            print(line_str)
    print("\\bottomrule\n\\label{Split"+str(strat_number+1)+"}\n\\caption{Split"+str(strat_number+1)+"}\n\\end{tabular}\\end{table*}")
    if early_return:
        return fix_df

def evaluate_p_w_best_fix_4(p_w_dict, w_threshold=0.01, p_threshold=0.01, realsumm=False,
                            strat_number=-1, early_return=False, first=False, type="kendall", level="system"):
    # Query scores of the test sets using the selected p and w values
    sns.set_theme()
    explainer_names = ["LimeExplainer"]
    alt_explainer_names = ["LIME"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "BARTScore"]

    if not realsumm and not first:
        config_dict = {
            "SummEvalkendall-coherence-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            "SummEvalkendall-consistency-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Consistency-KD"
            },
            "SummEvalkendall-fluency-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Fluency-KD"
            },
            "SummEvalkendall-relevance-only-scores": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Relevance-KD"
            }
        }
    else:
        config_dict = {
            "realsumm": {
                "lps": ["en"],
                "explainers": explainer_names,
                "metrics": ref_based,
                "alt": "Coherence-KD"
            },
            }
    # ------------------- Get dataframe with fixed p and w selection ------------------------
    # Fix w and p values based on the previous analysis can be set as function arguments
    lp_l = []
    expl_l = []
    metric_l = []
    diff_l = []
    orig_l = []
    new_corr = []
    tag_l = []
    perc_l = []
    p_values_l = []

    max_non_fix = []
    max_perc_non_fix = []

    for tag in config_dict.keys():
        lps = config_dict[tag]["lps"]
        explainers = config_dict[tag]["explainers"]
        metrics = config_dict[tag]["metrics"]

        for lp in lps:
            for explainer in explainers:
                for metric in metrics:
                    # Load df with resulting correlations for w and p values
                    if not realsumm:
                        filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "sys_level_tables",
                                    "__".join([tag, explainer, metric, lp]) + "_test_" + str(
                                        strat_number) + ".json")
                    else:
                        filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "sys_level_tables",
                                     "__".join([tag, explainer, metric, lp]) + "_sysLevel" + ".json")

                    try:
                        with open(filepath, encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception as e:
                        print(str(e) + "This is expected for Transquest and COMET with other explainers")
                        continue



                    try:
                        fix_p = mean([p_w_d[metric][explainer]["p"] for p_w_d in p_w_dict])
                        fix_w = mean([p_w_d[metric][explainer]["w"] for p_w_d in p_w_dict])
                        fix_p = min(np.arange(-30, 30, 0.1), key=lambda x: abs(x - fix_p))
                        fix_w = min([0, 0.2, 0.4, 0.6, 0.8, 1], key=lambda x: abs(x - fix_w))
                    except:
                        fix_p = p_w_dict[metric][explainer]["p"]
                        fix_w = p_w_dict[metric][explainer]["w"]

                    print("Selected: w:", fix_w, " p: ", fix_p )

                    for i, d in enumerate(data):
                        if math.isclose(list(d.values())[0]["p"], fix_p, abs_tol=0.00001) and math.isclose(list(d.values())[0]["w"],fix_w, abs_tol=0.00001):
                            selected_bmx = dict(sorted(d.items()))
                        if math.isclose(list(d.values())[0]["w"],1.0, abs_tol=0.00001):
                            original = dict(sorted(d.items()))

                    bmx_matrix = np.array([np.array(selected_bmx[sys]["bmx"]) for sys in selected_bmx.keys() ])
                    human1_matrix = np.array([np.array(selected_bmx[sys]["human"]) for sys in selected_bmx.keys()])
                    original_matrix = np.array([np.array(original[sys]["bmx"]) for sys in original.keys()])
                    human2_matrix = np.array([np.array(original[sys]["human"]) for sys in original.keys()])

                    h = human1_matrix.mean(axis=1)
                    o = original_matrix.mean(axis=1)
                    b = bmx_matrix.mean(axis=1)

                    h = (h - h.mean()) / h.std()
                    o = (o - o.mean()) / o.std()
                    b = (b - b.mean()) / b.std()

                    def NormalizeData(data):
                        return (data - np.min(data)) / (np.max(data) - np.min(data))

                    #h = NormalizeData(h)
                    #o = NormalizeData(o)
                    #b = NormalizeData(b)

                    # Sort by human
                    h,o,b = zip(*sorted(zip(h, o, b), key=lambda x: x[0]))

                    #bmx_corr = nlpstats.correlations.correlations.correlate(bmx_matrix, human1_matrix, level, type)
                    #orig_corr = nlpstats.correlations.correlations.correlate(original_matrix, human1_matrix, level, type)
                    #significance = nlpstats.correlations.permutation.permutation_test(bmx_matrix, original_matrix, human1_matrix,
                    #                                                   level, type, "both", alternative="greater", n_resamples=9999)

                    counter = np.array([x for x in range(len(h))])
                    m1, b1 = np.polyfit(counter, o, deg=1)
                    m2, b2 = np.polyfit(counter, b, deg=1)



                    # fix_df.plot(x="counter", y=['human', 'orig', 'bmx'], legend=False)
                    plt.title(tag+"_"+metric+"_"+explainer)
                    plt.plot(counter, b, label="expl", color="red")
                    plt.plot(counter, o, label="orig", color="blue")
                    #plt.plot(counter, m2 * counter + b2, color='red')
                    #plt.plot(counter, m1 * counter + b1, color='blue')
                    plt.plot(counter, h, label="human", color='green')
                    plt.legend()
                    plt.show()


                    #p_values_l += [significance.pvalue]

                    #print(significance.pvalue)
                    #diff = bmx_corr - orig_corr  # Determine the difference between all correlations and the original


                    lp_l.append(lp)
                    tag_l.append(tag)
                    expl_l.append(explainer)
                    metric_l.append(metric)
                    #diff_l.append(diff)
                    #orig_l.append(orig_corr)
                    #new_corr.append(bmx_corr)

    # Build a pandas df of the result values for each metric, explainer, lp combination
    # Uses fix p and w values from function definition
    sign = [True if pv <= 0.05 else False for pv in p_values_l]
    fix_df = pd.DataFrame([tag_l, lp_l, expl_l, metric_l, diff_l,   p_values_l, sign]).transpose()
    fix_df.columns = ["tag", "lp", "explainer", "metric", "diff", "p_value", "significant"]
    print(fix_df)
    fix_df["orig"] = fix_df["orig"].astype("float")
    fix_df["new_correlation"] = fix_df["new_correlation"].astype("float")

    # Generate Latex Tables
    alt_names = {"BERTSCORE_REF_FREE_XNLI": "XBERTScore",
                 "XLMRSBERT": "XLMR-SBERT",
                 "Transquest": "TRANSQUEST",
                 "COMET": "COMET",
                 "BERTSCORE_REF_BASED_ROBERTA_DEFAULT": "BERTScore",
                 "SentenceBLEU_REF_BASED": "SentenceBLEU",
                 "BARTScore": "BARTScore",
                 "RandomScore": "RandomScore",
                 "Rouge-1": "Rouge-1",
                 "Rouge-2": "Rouge-2",
                 "Rouge-l": "Rouge-l"}

def average_testsets(res_list):
    combined = pd.concat(res_list)
    for r in res_list:
        print(r.to_string())
    sign = [d["significant"].tolist() for d in res_list]
    sign_counts = [sum([sign[x][y] for x in range(len(sign))]) for y in range(len(sign[0]))]
    print(sign_counts)
    c1 = combined[combined["tag"].str.contains("spearman")]
    c2 = combined[~combined["tag"].str.contains("spearman")]
    m1 = c1.groupby(["tag", "explainer", "metric"]).mean()
    m2 = c2.groupby(["tag", "explainer", "metric"]).mean()
    print(m1.to_string())
    print(m2.to_string())
    print(m1.groupby(["metric"]).mean().to_string())
    print(m2.groupby(["metric"]).mean().to_string())

def bonferroni_partial_conjunction_pvalue_test(pvalues, alpha=0.05):
    N = len(pvalues)

    pvalues_with_indices = [(p, i) for i, p in enumerate(pvalues)]
    pvalues_with_indices = sorted(pvalues_with_indices)
    pvalues = [p for p, _ in pvalues_with_indices]
    indices = [i for _, i in pvalues_with_indices]

    p_u = [(N - (u + 1) + 1) * pvalues[u] for u in range(N)]  # (u + 1) because this is 0-indexed
    p_star = []
    k_hat = 0
    for u in range(N):
        if u == 0:
            p_star.append(p_u[u])
        else:
            p_star.append(max(p_star[-1], p_u[u]))
        if p_star[-1] <= alpha:
            k_hat += 1

    # The Holm procedure will always reject the lowest p-values
    significant_datasets = indices[:k_hat]
    return k_hat, significant_datasets


if __name__ == '__main__':
    # sns.set(rc={'figure.figsize': (3.15, 2.5)})
    # explore_all()
    res_list = []
    pw_dicts = []

    # For Realsumm we have 12 splits
    for x in range(0,8):
        sns.set(rc={'figure.figsize': (3.15, 3)})
        pw_dict = find_best_p_and_w_alternative(strat_number=x, realsumm=False)
        pw_dicts.append(pw_dict)
        sns.set(rc={'figure.figsize': (3.15, 2)})
        res_list.append(evaluate_p_w_best_fix_3(pw_dict, early_return=True,strat_number=x, realsumm=False, level="system", type="kendall"))
        res_list[-1]["split"] = [x]*res_list[-1].shape[0]


    p_values = [r["p_value"] for r in res_list]
    p_tag = [r["tag"] for r in res_list]
    p_per_property = [[p_values[x][y] for x in range(len(p_values))] for y in range(len(p_values[0]))]
    tag_per_property = [[p_values[x][y] for x in range(len(p_tag))] for y in range(len(p_tag[0]))]
    property_wise_bonferroni = [bonferroni_partial_conjunction_pvalue_test(p) for p in p_per_property]

    for t, p in zip(tag_per_property, property_wise_bonferroni):
        print(t)
        print(p)

    print(bonferroni_partial_conjunction_pvalue_test(sum(p_per_property, [])))



    print("Averaging results:")
    average_testsets(res_list)
    #res_list.append(evaluate_p_w_best_fix_3(pw_dicts, realsumm=True, early_return=True))
    #res_list.append(evaluate_p_w_best_fix_3(pw_dicts, realsumm=True, early_return=True,level="input",type="pearson"))

    #with open("C:\\Users\\USERNAME\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\other\\pw_calibration_method1.json", 'w') as f:
    #    json.dump({"method1":pw_dicts}, f)
