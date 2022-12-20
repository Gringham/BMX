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
    explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer", "InputMarginalizationExplainer",
                       "RandomExplainer"]
    alt_explainer_names = ["Erasure", "LIME", "SHAP", "IM", "Random"]
    ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping", "RandomScore"]

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
                    df["diff"] = df[df["w"] != 1]["corr"] - orig

                    df = df[df["w"] != 1]

                    if unassigned:
                        all_df = df
                        unassigned = False
                    else:
                        all_df = pd.concat([all_df, df])

    print("Number of overall datapoints: ", len(all_df))

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
    explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer", "InputMarginalizationExplainer"]
    alt_explainer_names = ["Erasure", "LIME", "SHAP", "IM"]
    # explainer_names = ["ErasureExplainer", "LimeExplainer", "ShapExplainer", "InputMarginalizationExplainer", "RandomExplainer"]
    ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping"]
    # ref_free = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping", "RandomScore"]

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
                    df = pd.read_csv(filepath, delimiter='\t')

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

    fig, axs = plt.subplots(2, len(explainer_names))
    p_w_dict = {}
    for x, explainer in enumerate(explainer_names):
        explainer_df = res_df[res_df["explainer"] == explainer]
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

    return p_w_dict


def evaluate_p_w_best_fix_2(p_w_dict, w_threshold=0.01, p_threshold=0.01, only_hyp=False, wmt22=False):
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
    alt_tags = ["WMT17", "Eval4NLP21", "MLQE-PE"]

    if wmt22:
        config_dict = {"wmt_22_test_sets": {
            "lps": ["en-cs", "en-ja", "en-mr", "en-yo", "km-en", "ps-en"],
            "explainers": explainer_names,
            "metrics": ref_free
        }}
        alt_tags = ["WMT22"]

        # Uncomment this for the MQM analysis
        #config_dict = {"mqm21": {
        #    "lps": ["en-de", "zh-en"],
        #    "explainers": ["LimeExplainer"],
        #    "metrics": ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT"]
        #}}
        #alt_tags = ["MQM"]

    # ------------------- Get dataframe with fixed p and w selection ------------------------
    # Fix w and p values based on the previous analysis can be set as function arguments
    lp_l = []
    expl_l = []
    metric_l = []
    diff_l = []
    orig_l = []
    new_corr = []
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
                    df = pd.read_csv(filepath, delimiter='\t')

                    fix_p = p_w_dict[explainer]["p"]
                    fix_w = p_w_dict[explainer]["w"]

                    # find closest to given medians
                    p_condition = abs(
                        df["p"] - min(np.arange(-30, 30, 0.1), key=lambda x: abs(x - fix_p))) < p_threshold
                    w_condition = abs(
                        df["w"] - min([0, 0.2, 0.4, 0.6, 0.8, 1], key=lambda x: abs(x - fix_w))) < w_threshold

                    selected_correlation = df[p_condition & w_condition]["corr"].tolist()
                    orig = df[df["w"] == 1]["corr"].iat[0]  # Determine the original correlation
                    diff = selected_correlation - orig  # Determine the difference between all correlations and the original

                    lp_l.append(lp)
                    tag_l.append(tag)
                    expl_l.append(explainer)
                    metric_l.append(metric)
                    diff_l.append(diff)
                    orig_l.append(orig)
                    new_corr.append(selected_correlation)

    # Build a pandas df of the result values for each metric, explainer, lp combination
    # Uses fix p and w values from function definition
    fix_df = pd.DataFrame([tag_l, lp_l, expl_l, metric_l, diff_l, orig_l, new_corr]).transpose()
    fix_df.columns = ["tag", "lp", "explainer", "metric", "diff", "orig", "new_correlation"]
    print(fix_df)

    # One hot encode
    fix_df_dummy = pd.get_dummies(data=fix_df, columns=["explainer", "metric", "lp", "tag"])

    cols = [c for c in fix_df_dummy.columns if c not in ["Unnamed: 0", "corr", "diff", "new_correlation", "orig"]]
    x = fix_df_dummy[cols]
    y = fix_df_dummy['diff']

    regr = linear_model.Ridge()
    regr.fit(x, y)

    tpls = []
    for n, c in zip(cols, regr.coef_.tolist()):
        tpls.append((n, c))

    # Only display 3 best and worst
    tpls = sorted(tpls, key=lambda x: x[1], reverse=True)
    tpls = tpls[:3] + tpls[-3:]
    print("Ordered Regression Weights:", tpls)
    print(regr.intercept_)
    cols = [e[0] for e in tpls]
    rename_dict = {"lp_en-cs":"en-cs",
                   "explainer_LimeExplainer":"LIME",
                   "metric_BERTSCORE_REF_FREE_XNLI":"XBERTScore",
                   "metric_XMoverScore_No_Mapping":"XMoverScore",
                   "explainer_ErasureExplainer":"Erasure",
                   "lp_en-yo":"en-yo",
                   "lp_de-zh":"de-zh",
                   "lp_tr-en":"tr-en",
                   "tag_wmt_22_expl_train":"MLQE-PE",
                   "lp_si-en":"si-en",
                   "lp_en-zh":"en-zh",
                   "lp_en-de":"en-de",
                   "lp_zh-en":"zh-en",
                   "metric_BERTSCORE_REF_BASED_ROBERTA_DEFAULT":"BERTScore",
                   "tag_mqm21":"mqm21",
                   "metric_XLMRSBERT":"XLMR-SBERT"}
    cols = [rename_dict[c] for c in cols]
    weights = [e[1] for e in tpls]
    y_pos = np.arange(len(weights))
    positive = np.array(["b" if w >= 0 else "r" for w in weights])

    plt.barh(y_pos, weights, color=positive)
    plt.yticks(y_pos, cols)
    plt.tight_layout()
    plt.savefig(
        "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\12_regressors.pdf")
    plt.show()

    fig, axs = plt.subplots(1)


    for y, tag in enumerate([""]):
        tuples = []
        for x, explainer in enumerate(explainer_names):
            explainer_df = fix_df[fix_df["explainer"] == explainer]
            better = explainer_df[explainer_df["diff"] > 0.00001]["diff"].count()
            worse = explainer_df[explainer_df["diff"] <= 0]["diff"].count()
            tuples.append(["better", better, alt_explainer_names[x]])
            tuples.append(["worse or equal", worse, alt_explainer_names[x]])

        better_worse_df = pd.DataFrame(tuples)
        better_worse_df.columns = ["Result", tag, "Explainer"]

        b = sns.barplot(data=better_worse_df, x="Explainer", y=tag, ax=axs, hue="Result")

    b.set_xlabel("")
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    if not "wmt_22_test_sets" in config_dict.keys():
        plt.savefig(
            "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\5_preselected_p_w_general.pdf")
    else:
        plt.savefig(
            "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\9_performance on_wmt22.pdf")


    plt.show()

    sns.set(rc={'figure.figsize': (3.5, 3)})
    expl_density_list = []
    names = config_dict.keys()
    for x, ob in enumerate(names):
        expl_density_list.append(
            [f[0] for f in
             fix_df[(fix_df["explainer"] == "LimeExplainer") & (fix_df["tag"] == ob)]["diff"].tolist()])

    expl_density_list = pd.DataFrame(expl_density_list).transpose()
    expl_density_list.columns = alt_tags
    plot = sns.kdeplot(data=expl_density_list, bw_adjust=0.9, fill=True)
    sns.move_legend(plot, "upper left", title='Dataset')
    plt.axvline(0, color="black")
    plot.set_yticklabels([])
    plot.set_ylabel("")
    plt.setp(plot.get_legend().get_texts(), fontsize='8')
    if not "wmt_22_test_sets" in config_dict.keys():
        plt.savefig(
            "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\6_density_plot_ds.pdf")

    plt.show()

    sns.set(rc={'figure.figsize': (3.15, 3)})
    expl_density_list = []
    names = ["BERTSCORE_REF_BASED_ROBERTA_DEFAULT", "SentenceBLEU_REF_BASED", "BERTSCORE_REF_FREE_XNLI",
             "XLMRSBERT", "XMoverScore_No_Mapping"]
    alt_names = ["BERTScore", "BLEU", "XBERTScore", "XLMR-SBERT", "XMoverScore-NoMap"]
    if "wmt_22_test_sets" in config_dict.keys():
        names = ["BERTSCORE_REF_FREE_XNLI", "XLMRSBERT", "XMoverScore_No_Mapping"]
        alt_names = ["XBERTScore", "XLMR-SBERT", "XMoverScore"]
    for x, ob in enumerate(names):
        expl_density_list.append(
            [f[0] for f in
             fix_df[(fix_df["explainer"] == "LimeExplainer") & (fix_df["metric"] == ob)]["diff"].tolist()])

    expl_density_list = pd.DataFrame(expl_density_list).transpose()

    expl_density_list.columns = [alt_names[x] + ": " + str(round(fix_df[fix_df["metric"] == n]["orig"].mean(), 3)) for
                                 x, n in
                                 enumerate(names)]

    plot = sns.kdeplot(data=expl_density_list, bw_adjust=0.9, fill=True)
    sns.move_legend(plot, "upper right", title="Metric")
    plt.setp(plot.get_legend().get_texts(), fontsize='8')
    plot.set_yticklabels([])
    plot.set_ylabel("")
    plt.tight_layout()
    plt.axvline(0, color="black")
    if not "wmt_22_test_sets" in config_dict.keys():
        plt.savefig(
            "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\7_density_plot_metrics.pdf")
    else:
        plt.savefig(
            "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\10_density_plot_metrics_wmt22.pdf")
    plt.show()

    expl_density_list = []
    names = ["ru-en", "ro-en", "cs-en", "si-en", "ru-de"]
    for x, ob in enumerate(names):
        expl_density_list.append(
            [f[0] for f in
             fix_df[(fix_df["explainer"] == "LimeExplainer") & (fix_df["lp"] == ob)]["diff"].tolist()])

    expl_density_list = pd.DataFrame(expl_density_list).transpose()
    expl_density_list.columns = names
    plot = sns.kdeplot(data=expl_density_list, bw_adjust=0.9, fill=True)
    try:
        sns.move_legend(plot, "upper left", title='Language Pair')
        plt.axvline(0, color="black")
        if not "wmt_22_test_sets" in config_dict.keys():
            plt.savefig(
                "C:\\Users\\Jirac\\PycharmProjects\\ExplainableMetrics\\xaiMetrics\\outputs\\Images_Paper_Auto_Gen\\8_density_plot_lps.pdf")
    except:
        pass
    plt.show()

    general_low_high_correlation = scipy.stats.pearsonr([n[0] for n in fix_df["new_correlation"]], fix_df["orig"]).statistic
    print("Corr:", general_low_high_correlation)

    lime_df = fix_df[(fix_df["explainer"] == "LimeExplainer")]
    print("Dataset lime median:", lime_df["diff"].max(), lime_df[lime_df["diff"]>0].count(), lime_df[lime_df["diff"]<=0].count())


if __name__ == '__main__':
    sns.set(rc={'figure.figsize': (3.15, 2.5)})
    explore_all()
    sns.set(rc={'figure.figsize': (3.15, 3)})
    pw_dict = find_best_p_and_w()
    sns.set(rc={'figure.figsize': (3.15, 2)})
    evaluate_p_w_best_fix_2(pw_dict)


