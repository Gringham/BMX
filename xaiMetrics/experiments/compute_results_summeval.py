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
    explainer_names = ["LimeExplainer"]  # , "InputMarginalizationExplainer"]
    alt_explainer_names = ["LIME"]  # , "IM"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "BARTScore"]

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

    config_dict = {
        "realsumm": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Coherence-KD"
        },}

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
        # plt.savefig(
        #    "C:\\Users\\USERNAME\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\4_best_p_and_w.pdf")
        plt.show()

        print("Best p and w dict (medians):", p_w_dict)
        p_w_per_metric[metric_name] = p_w_dict

    return p_w_per_metric


def evaluate_p_w_best_fix_2(p_w_dict, w_threshold=0.01, p_threshold=0.01, only_hyp=False, wmt22=False, mqm=False):
    sns.set_theme()
    explainer_names = ["LimeExplainer"]  # , "InputMarginalizationExplainer"]
    alt_explainer_names = ["LIME"]  # , "IM"]
    ref_based = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "BARTScore"]

    config_dict = {
        "SummEvalSummEval-completecoherence-kendall": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Coherence-KD"
        },
        "SummEvalSummEval-completeconsistency-kendall": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Consistency-KD"
        },
        "SummEvalSummEval-completefluency-kendall": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Fluency-KD"
        },
        "SummEvalSummEval-completerelevance-kendall": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Relevance-KD"
        },
        "SummEvalSummEval-completecoherence-spearman": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Coherence-SP"
        },
        "SummEvalSummEval-completeconsistency-spearman": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Consistency-SP"
        },
        "SummEvalSummEval-completefluency-spearman": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Fluency-SP"
        },
        "SummEvalSummEval-completerelevance-spearman": {
            "lps": ["en"],
            "explainers": explainer_names,
            "metrics": ref_based,
            "alt": "Relevance-SP"
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
                    filepath = os.path.join(ROOT_DIR, "xaiMetrics", "outputs", "experiment_results",
                                            "__".join([tag, explainer, metric, lp]) + ".tsv")
                    try:
                        df = pd.read_csv(filepath, delimiter='\t')
                    except Exception as e:
                        print(str(e) + "This is expected for Transquest and COMET with other explainers")
                        continue


                    mean = lambda x: sum(x)/len(x)
                    try:
                        fix_p = mean([p_w_d[metric][explainer]["p"] for p_w_d in p_w_dict])
                        fix_w = mean([p_w_d[metric][explainer]["w"] for p_w_d in p_w_dict])
                    except:
                        fix_p = p_w_dict[metric][explainer]["p"]
                        fix_w = p_w_dict[metric][explainer]["w"]

                    print("Selected: w:", fix_w, " p: ", fix_p)

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
                fix_df.columns = ["tag", "lp", "explainer", "metric", "diff", "orig", "new_correlation",
                                  "improvement_percent"]
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
                                selection = fix_df[
                                    (fix_df["tag"] == tag) & (fix_df["metric"] == metric) & (fix_df["lp"] == lp) & (
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

                for condition in [("KD", "KD"), ("SP", "SP")]:
                    avg = []
                    for key, value in scores_per_tag.items():
                        if condition[1] in config_dict[key]["alt"]:
                            avg.append(value)

                    if len(avg) > 0:
                        avg = [[round(mean([avg[z][i][j] for z in range(len(avg))]), 3) for j in range(len(avg[0][i]))]
                               for i in range(len(avg[0]))]

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



if __name__ == '__main__':
    # sns.set(rc={'figure.figsize': (3.15, 2.5)})
    # explore_all()
    sns.set(rc={'figure.figsize': (3.15, 3)})
    pw_dict = find_best_p_and_w()
    sns.set(rc={'figure.figsize': (3.15, 2)})
    evaluate_p_w_best_fix_2(pw_dict, wmt22=False, mqm=False)
