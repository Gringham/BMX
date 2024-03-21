import json
import math
from statistics import mean

import nlpstats.correlations.correlations
from project_root import ROOT_DIR
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import linear_model


def explore_all():
    sns.set_theme()
    explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer", "InputMarginalizationExplainer"]
    alt_explainer_names = ["Erasure", "LIME", "SHAP", "IM"]
    ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping"]

    config_dict = {
        "wmt17": {
            "lps": ["cs-en", "de-en", "fi-en", "lv-en", "ru-en", "tr-en", "zh-en"],
            "explainers": explainer_names,
            "metrics": ref_free + ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "SentenceBLEU_REF_BASED"]
        },
        "eval4_nlp_21_test": {
            "lps": ["ro-en", "et-en", "ru-de", "de-zh"],
            "explainers": explainer_names,
            "metrics": ref_free
        },
        "wmt_22_expl_train": {
            "lps": ["ro-en", "et-en", "ru-en", "si-en", "ne-en", "en-zh", "en-de"],
            "explainers": explainer_names,
            "metrics": ref_free
        },
    }

    all_df = None
    unassigned = True

    for tag in config_dict.keys():
        lps = config_dict[tag]["lps"]
        explainers = config_dict[tag]["explainers"]
        metrics = config_dict[tag]["metrics"]

        for lp in lps:
            for explainer in explainers:
                for metric in metrics:
                    # Load df with resulting correlations for w and p values
                    filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results",
                                            "__".join([tag, explainer, metric, lp]) + ".tsv")
                    df = pd.read_csv(filepath, delimiter='\t')
                    df["explainer"] = [explainer] * len(df)
                    df["metric"] = [metric] * len(df)
                    df["lp"] = [lp] * len(df)
                    df["tag"] = [tag] * len(df)
                    orig = df[df["w"] == 1]["corr"].iat[0]

                    # Get difference to original for all of the values

                    df = df[df["w"] != 1]
                    df["diff"] = df[df["w"] != 1]["corr"] - orig
                    df["orig"] = [orig] * len(df["diff"])

                    if unassigned:
                        all_df = df
                        unassigned = False
                    else:
                        all_df = pd.concat([all_df, df])

    print("Number of overall datapoints: ", len(all_df))
    all_df = all_df[all_df["orig"] > 0]
    print("Number of overall datapoints: ", len(all_df))
    print(all_df.groupby("explainer").max())
    print(all_df.groupby("explainer").median())
    # Get Regressors
    all_df_dummy = pd.get_dummies(data=all_df, columns=["explainer", "metric", "lp", "tag"])

    cols = [c for c in all_df_dummy.columns if c not in ["Unnamed: 0", "corr", "diff"]]
    x = all_df_dummy[cols]
    y = all_df_dummy['diff']

    regr = linear_model.Ridge()
    regr.fit(x, y)

    for n, c in zip(cols, regr.coef_.tolist()):
        print(n, round(c, 3))
    print(regr.intercept_)

    # Plot distribution
    expl_density_list = []
    for e in explainer_names:
        expl_density_list.append(all_df[all_df["explainer"] == e]["diff"].tolist())

    expl_density_list = pd.DataFrame(expl_density_list).transpose()
    expl_density_list.columns = alt_explainer_names
    plot = sns.kdeplot(data=expl_density_list, bw_adjust=0.9, fill=True)
    sns.move_legend(plot, "upper left", title='Explainer')
    plt.axvline(0, color="black")
    plt.xlim(-0.4, 0.25)
    plot.set_yticklabels([])
    plot.set_ylabel("")
    plt.tight_layout()
    plt.savefig(
        "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\3_All_Correlations.pdf")
    plt.show()


def find_best_p_and_w(only_hyp=False):
    sns.set_theme()
    explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer"]  # , "InputMarginalizationExplainer"]
    alt_explainer_names = ["Erasure", "LIME", "SHAP"]  # , "IM"]
    # explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer", "InputMarginalizationExplainer", "RandomExplainer"]
    ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping", "Transquest", "COMET"]
    # ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping", "RandomScore"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "SentenceBLEU_REF_BASED"]

    config_dict = {
        "wmt17": {
            "lps": ["cs-en", "de-en", "fi-en", "lv-en", "ru-en", "tr-en", "zh-en"],
            "explainers": explainer_names,
            "metrics": ref_free + ref_based
        },
        "eval4_nlp_21_test": {
            "lps": ["ro-en", "et-en", "ru-de", "de-zh"],
            "explainers": explainer_names,
            "metrics": ref_free
        },
        "wmt_22_expl_train": {
            "lps": ["ro-en", "et-en", "ru-en", "si-en", "ne-en", "en-zh", "en-de"],
            "explainers": explainer_names,
            "metrics": ref_free
        }
    }

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
                    filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results",
                                            "__".join([tag, explainer, metric, lp]) + ".tsv")
                    if only_hyp:
                        filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results",
                                                "__".join([tag, explainer, metric, lp, "only_hyp"]) + ".tsv")

                    try:
                        df = pd.read_csv(filepath, delimiter='\t')
                    except Exception as e:
                        print(str(e) + "This is expected for Transquest and COMET with other explainers")
                        continue

                    max_correlation = df[
                        "corr"].max()  # Maximum correlation for dataset, lp, explainer, metric combination
                    p_best = df[df["corr"] == max_correlation]["p"].iat[
                        0]  # Get the best p for the current dataset, lp, explainer, metric combination
                    w_best = df[df["corr"] == max_correlation]["w"].iat[
                        0]  # Get the best w for the current dataset, lp, explainer, metric combination
                    orig = df[df["w"] == 1]["corr"].iat[0]  # Determine the original correlation
                    diff = max_correlation - orig  # Determine the difference between all correlations and the original

                    lp_l.append(lp)
                    tag_l.append(tag)
                    expl_l.append(explainer)
                    metric_l.append(metric)
                    diff_l.append(diff)
                    p_l.append(p_best)
                    w_l.append(w_best)
                    orig_l.append(orig)
                    max_l.append(max_correlation)

    print("Settings with improvement:", len(lp_l))

    # Build a pandas df of the result values for each metric, explainer, lp combination
    res_df = pd.DataFrame([tag_l, lp_l, expl_l, metric_l, diff_l, p_l, w_l, orig_l, max_l]).transpose()
    res_df.columns = ["tag", "lp", "explainer", "metric", "diff", "p", "w", "orig", "max"]
    res_df = res_df[res_df["orig"] > 0]
    print("Settings with improvement:", len(res_df))

    p_w_per_metric = {}
    for metric_name in ref_free + ref_based:
        fig, axs = plt.subplots(2, len(explainer_names))
        p_w_dict = {}
        for x, explainer in enumerate(explainer_names):
            explainer_df = res_df[(res_df["explainer"] == explainer) & (res_df["metric"] == metric_name)]
            if len(explainer_df) == 0:
                continue
            _, w_dict = explainer_df["w"].plot.box(ax=axs[0, x], return_type="both")
            _, p_dict = explainer_df[explainer_df["diff"] >= 0]["p"].plot.box(ax=axs[1, x], return_type="both")
            p_w_dict[explainer] = {"p": round(p_dict['medians'][0].get_ydata()[0], 3),
                                   "w": round(w_dict['medians'][0].get_ydata()[0], 3)}
            axs[0, x].set_ylim(-0.1, 1.1)
            axs[1, x].set_ylim(-34, 34)
            axs[0, x].set_xticklabels([])
            axs[1, x].set_xticklabels([])
            if x != 0:
                axs[0, x].set_yticklabels([])
                axs[1, x].set_yticklabels([])
            else:
                axs[0, x].set_ylabel("w")
                axs[1, x].set_ylabel("p")
            axs[0, x].set_title('Md: ' + str(p_w_dict[explainer]["w"]), fontsize=8)
            axs[1, x].set_title('Md: ' + str(p_w_dict[explainer]["p"]), fontsize=8)
            axs[0, x].annotate(alt_explainer_names[x], xy=(0.5, 1), xytext=(0, 20),
                               xycoords='axes fraction', textcoords='offset points',
                               size='medium', ha='center', va='baseline')

        fig.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.savefig(
            "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\4_best_p_and_w.pdf")
        plt.show()

        print("Best p and w dict (medians):", p_w_dict)
        p_w_per_metric[metric_name] = p_w_dict

    return p_w_per_metric


def find_best_p_and_w_alternative(only_hyp=False):
    sns.set_theme()
    explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer"]  # , "InputMarginalizationExplainer"]
    alt_explainer_names = ["Erasure", "LIME", "SHAP"]  # , "IM"]
    # explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer", "InputMarginalizationExplainer", "RandomExplainer"]
    ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping", "Transquest", "COMET"]
    # ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping", "RandomScore"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "SentenceBLEU_REF_BASED"]

    config_dict = {
        # "wmt17": {
        #    "lps": ["cs-en", "de-en", "fi-en", "lv-en", "ru-en", "tr-en", "zh-en"],
        #    "explainers": explainer_names,
        #    "metrics": ref_free + ref_based
        # },
        "eval4_nlp_21_test": {
            "lps": ["ro-en", "et-en", "ru-de", "de-zh"],
            "explainers": explainer_names,
            "metrics": ref_free
        },
        "wmt_22_expl_train": {
            "lps": ["ro-en", "et-en", "ru-en", "si-en", "ne-en", "en-zh", "en-de"],
            "explainers": explainer_names,
            "metrics": ref_free
        }
    }

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
                    filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results",
                                            "__".join([tag, explainer, metric, lp]) + ".tsv")
                    if only_hyp:
                        filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results",
                                                "__".join([tag, explainer, metric, lp, "only_hyp"]) + ".tsv")

                    try:
                        df = pd.read_csv(filepath, delimiter='\t')
                    except Exception as e:
                        print(str(e) + "This is expected for Transquest and COMET with other explainers")
                        continue

                    orig = df[df["w"] == 1]["corr"].iat[0]  # Determine the original correlation
                    # Maximum correlation for dataset, lp, explainer, metric combination
                    p_best = df[df["corr"] > orig]["p"].tolist()
                    w_best = df[df["corr"] > orig]["w"].tolist()

                    for w, p in zip(w_best, p_best):
                        lp_l.append(lp)
                        tag_l.append(tag)
                        expl_l.append(explainer)
                        metric_l.append(metric)
                        diff_l.append(df[(df["p"] == p) & (df["w"] == w)] - orig)
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
    for metric_name in ref_free + ref_based:
        fig, axs = plt.subplots(2, len(explainer_names))
        p_w_dict = {}
        for x, explainer in enumerate(explainer_names):
            explainer_df = res_df[(res_df["explainer"] == explainer) & (res_df["metric"] == metric_name)]
            if len(explainer_df) == 0:
                continue
            _, w_dict = explainer_df["w"].plot.box(ax=axs[0, x], return_type="both")
            _, p_dict = explainer_df["p"].plot.box(ax=axs[1, x], return_type="both")
            p_w_dict[explainer] = {"p": round(p_dict['medians'][0].get_ydata()[0], 3),
                                   "w": round(w_dict['medians'][0].get_ydata()[0], 3)}
            axs[0, x].set_ylim(-0.1, 1.1)
            axs[1, x].set_ylim(-34, 34)
            axs[0, x].set_xticklabels([])
            axs[1, x].set_xticklabels([])
            if x != 0:
                axs[0, x].set_yticklabels([])
                axs[1, x].set_yticklabels([])
            else:
                axs[0, x].set_ylabel("w")
                axs[1, x].set_ylabel("p")
            axs[0, x].set_title('Md: ' + str(p_w_dict[explainer]["w"]), fontsize=8)
            axs[1, x].set_title('Md: ' + str(p_w_dict[explainer]["p"]), fontsize=8)
            axs[0, x].annotate(alt_explainer_names[x], xy=(0.5, 1), xytext=(0, 20),
                               xycoords='axes fraction', textcoords='offset points',
                               size='medium', ha='center', va='baseline')

        fig.subplots_adjust(hspace=0.3)
        plt.tight_layout()
        plt.savefig(
            "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\4_best_p_and_w.pdf")
        plt.show()

        print("Best p and w dict (medians):", p_w_dict)
        p_w_per_metric[metric_name] = p_w_dict

    return p_w_per_metric



def evaluate_p_w_best_fix(p_w_dict, w_threshold=0.01, p_threshold=0.01, wmt22=False, mqm=False, early_return=False,
                            first=False, type="kendall", level="system"):
    print(p_w_dict)
    sns.set_theme()
    explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer"]  # , "InputMarginalizationExplainer"]
    ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "Transquest", "COMET"]

    if wmt22:
        config_dict = {"wmt_22_test_sets": {
            "lps": ["en-cs", "en-ja", "en-mr", "en-yo", "km-en", "ps-en"],
            "explainers": explainer_names,
            "metrics": ref_free
        }}
        alt_tags = ["WMT22"]

        # Uncomment this for the MQM analysis

    if mqm:
        config_dict = {"mqm21-kendall-sys-level2": {
            "lps": ["en-de", "zh-en"],
            "explainers": ["LimeExplainer"],
            "metrics": ref_free
        }}
        alt_tags = ["MQM"]
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

    # mqm = False
    for tag in config_dict.keys():
        lps = config_dict[tag]["lps"]
        explainers = config_dict[tag]["explainers"]
        metrics = config_dict[tag]["metrics"]

        for lp in lps:
            for explainer in explainers:
                for metric in metrics:
                    # Load df with resulting correlations for w and p values
                    if not mqm:
                        filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "sys_level_tables",
                                                "__".join([tag, explainer, metric, lp]) + "_segLevel" + ".json")
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

                    print("Selected: w:", fix_w, " p: ", fix_p)

                    if mqm:
                        # fix for the format of system level scores
                        for i, d in enumerate(data):
                            if math.isclose(list(d.values())[0]["p"], fix_p, abs_tol=0.00001) and math.isclose(
                                    list(d.values())[0]["w"], fix_w, abs_tol=0.00001):
                                selected_bmx = dict(sorted(d.items()))
                            if math.isclose(list(d.values())[0]["w"], 1.0, abs_tol=0.00001):
                                original = dict(sorted(d.items()))

                    else:
                        for i, d in enumerate(data):
                            if math.isclose(d["p"], fix_p, abs_tol=0.00001) and math.isclose(d["w"], fix_w,
                                                                                             abs_tol=0.00001):
                                selected_bmx = d
                            if math.isclose(d["w"], 1.0, abs_tol=0.00001):
                                original = d

                    if mqm:
                        # Workaround to make samples equal length
                        for sys in selected_bmx.keys():
                            for x in range(0, 5000):
                                if not x in selected_bmx[sys]["seg"]:
                                    selected_bmx[sys]["seg"].append(x)
                                    selected_bmx[sys]["bmx"].append(-9999)
                                    selected_bmx[sys]["human"].append(-9999)
                            srt = sorted(list(
                                zip(selected_bmx[sys]["seg"], selected_bmx[sys]["bmx"], selected_bmx[sys]["human"])),
                                key=lambda x: x[0])
                            selected_bmx[sys]["seg"] = [s[0] for s in srt]
                            selected_bmx[sys]["bmx"] = [s[1] for s in srt]
                            selected_bmx[sys]["human"] = [s[2] for s in srt]

                        for sys in original.keys():
                            for x in range(0, 5000):
                                if not x in original[sys]["seg"]:
                                    original[sys]["seg"].append(x)
                                    original[sys]["bmx"].append(-9999)
                                    original[sys]["human"].append(-9999)
                            srt = sorted(
                                list(zip(original[sys]["seg"], original[sys]["bmx"], original[sys]["human"])),
                                key=lambda x: x[0])
                            original[sys]["seg"] = [s[0] for s in srt]
                            original[sys]["bmx"] = [s[1] for s in srt]
                            original[sys]["human"] = [s[2] for s in srt]

                        bmx_matrix = np.array(
                            [np.array(selected_bmx[sys]["bmx"], dtype=np.float64) for sys in selected_bmx.keys()],
                            dtype=np.float64)
                        human1_matrix = np.array(
                            [np.array(selected_bmx[sys]["human"], dtype=np.float64) for sys in selected_bmx.keys()],
                            dtype=np.float64)
                        original_matrix = np.array(
                            [np.array(original[sys]["bmx"], dtype=np.float64) for sys in original.keys()],
                            dtype=np.float64)
                        human2_matrix = np.array(
                            [np.array(original[sys]["human"], dtype=np.float64) for sys in original.keys()],
                            dtype=np.float64)
                        bmx_matrix = bmx_matrix[:, bmx_matrix.min(axis=0) >= -9000]
                        human1_matrix = human1_matrix[:, human1_matrix.min(axis=0) >= -9000]
                        original_matrix = original_matrix[:, original_matrix.min(axis=0) >= -9000]


                    else:
                        bmx_matrix = np.array([np.array(selected_bmx["bmx"])]).T
                        human1_matrix = np.array([np.array(selected_bmx["human"])]).T
                        original_matrix = np.array([np.array(original["bmx"])]).T
                        human2_matrix = np.array([np.array(original["human"])])

                    bmx_corr = nlpstats.correlations.correlations.correlate(bmx_matrix, human1_matrix, level, type)
                    orig_corr = nlpstats.correlations.correlations.correlate(original_matrix, human1_matrix, level,
                                                                             type)
                    significance = nlpstats.correlations.permutation.permutation_test(bmx_matrix, original_matrix,
                                                                                      human1_matrix,
                                                                                      level, type, "both",
                                                                                      alternative="greater",
                                                                                      n_resamples=9999)
                    p_values_l += [significance.pvalue]

                    print(bmx_corr, orig_corr, significance.pvalue)
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
    print(fix_df.to_string())
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
                 "Rouge-l": "Rouge-l",
                 }

    for tag in config_dict.keys():
        print("\\begin{table*}\n\\centering\\small\n\\begin{tabular}{l|" + "".join(
            "c" * len(config_dict[tag]["metrics"]) + "}\\toprule"))
        print("LP & " + " & ".join(
            ["\\textbf{" + alt_names[m] + "}" for m in config_dict[tag]["metrics"]]))
        print("\\\\\\midrule")
        scores_per_metric = {metric: [] for metric in config_dict[tag]["metrics"]}

        for lp in config_dict[tag]["lps"]:
            line_str = lp + " & "
            for metric in config_dict[tag]["metrics"]:
                start = True
                scores = []
                for explainer in config_dict[tag]["explainers"]:
                    selection = fix_df[(fix_df["tag"] == tag) & (fix_df["metric"] == metric) & (fix_df["lp"] == lp) & (
                            fix_df["explainer"] == explainer)]
                    if len(selection) == 0:
                        continue
                    if start:
                        scores.append(selection["orig"].tolist()[0])
                        start = False
                    scores.append(selection["new_correlation"].tolist()[0])
                max_score_indices = np.argwhere(scores == np.amax(scores)).flatten().tolist()
                # if not 0 in max_score_indices:
                #    line_str += "\\tablegreen{"
                # else:
                #    line_str += "\\tablered{"
                for i, score in enumerate(scores):
                    val = str(round(score, 3))
                    if i in max_score_indices:
                        if (0 in max_score_indices and i == 0) or not 0 in max_score_indices:
                            val = "\\textbf{" + val + "}"
                    if i != 0 and score > scores[0]:
                        val = "\\tablegreen{" + val + "}"
                    val = "$" + val + "$"
                    if i != 0:
                        val = "/" + val
                    line_str += val
                line_str += " & "
                scores_per_metric[metric].append(scores)
            line_str = line_str[:-2] + "\\\\"
            print(line_str)

        avg = []
        for value in scores_per_metric.values():
            avg.append([])
            for x in range(len(value[0])):
                avg[-1].append(round(mean([value[y][x] for y in range(len(value))]), 3))
        # averages = [[str(mean(v)) for v in list(zip(value))] for value in scores_per_metric.values()]

        avg2 = []
        for a in avg:

            max_score_indices = np.argwhere(a == np.amax(a)).flatten().tolist()
            # if not 0 in max_score_indices:
            #    avg2.append("\\tablegreen{")
            # else:
            #    avg2.append("\\tablered{")
            avg2.append("")
            for i, score in enumerate(a):
                val = str(score)
                if i in max_score_indices:
                    if (0 in max_score_indices and i == 0) or not 0 in max_score_indices:
                        val = "\\textbf{" + val + "}"
                if i != 0:
                    avg2[-1] += "/"
                    if score > a[0]:
                        val = "\\tablegreen{" + val + "}"
                avg2[-1] += "$" + val + "$"

        line_str = "AVG & " + " & ".join(avg2) + "\\\\"
        print(line_str)

        print("\\bottomrule\n\\end{tabular}\\end{table*}")
    return fix_df

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
    sns.set(rc={'figure.figsize': (3.15, 3)})
    pw_dict = find_best_p_and_w_alternative()
    sns.set(rc={'figure.figsize': (3.15, 2)})
    f1 = evaluate_p_w_best_fix(pw_dict, wmt22=True, mqm=False, type="spearman", level="global")
    sns.set(rc={'figure.figsize': (3.15, 2)})
    f2 = evaluate_p_w_best_fix(pw_dict, wmt22=False, mqm=True, type="kendall", level="global")
    f3 = evaluate_p_w_best_fix(pw_dict, wmt22=False, mqm=True, type="kendall", level="system")

    df = f2  # pd.concat([f1, f2])
    groups = df.groupby(["metric", "explainer"])
    for name, group in groups:
        print(name)
        print(group["p_value"].tolist())
        print(group["tag"], group["lp"])
        print(bonferroni_partial_conjunction_pvalue_test(group["p_value"].tolist()))

    df = f3  # pd.concat([f1, f2])
    groups = df.groupby(["metric", "explainer"])
    for name, group in groups:
        print(name)
        print(group["p_value"].tolist())
        print(group["tag"], group["lp"])
        print(bonferroni_partial_conjunction_pvalue_test(group["p_value"].tolist()))
