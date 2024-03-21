import json

import dill, os
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from project_root import ROOT_DIR
from xaiMetrics.evalTools.utils.power_mean import power_mean
from xaiMetrics.explanations.FeatureImportanceExplanation import FeatureImportanceExplanationCollection


def explanations_to_scores_and_eval(df, attributions, p=None, w=None, dataset="UNDEFINED", limit=True, lp="UNDEFINED",
                                    save_fig=False, plot=False, save_data=True, result_data_dir="experiment_results",
                                    result_images="experiment_graphs_pdf", only_hyp=False, kendall=False,
                                    sys_level=False, spearman=False, normalize=False, appendix="", rm_ref=False):
    """
    :param df: dataframe to compute new scores on
    :param attributions: Explanations made with one of the projects explainers
    :param p: A list of (ideally equally spaced) p values to compute with the powermeans
    :param w: A list of (ideally equally spaced weights for combination with the original metric
    :param dataset: dataset name (important for savefile)
    :param lp: current language pair (important for savefile)
    :param save_fig: Whether to save a figure
    :param plot: whether to plot a figure
    :param save_data: Whether to save all correlations as a tsv
    :param result_data_dir: Directory name in outputs to save tsv outputs
    :param result_images: directory name in outputs to sace result images
    :param only_hyp: only consider hypothesis values if true, else gt + hyp
    :return:
    """
    dataset = dataset + appendix
    if rm_ref:
        df = df[~df["system"].str.contains('ref')]
    da = df["DA"].tolist()
    hyp = df["HYP"].tolist()
    src = df["SRC"].tolist()
    if sys_level:
        if "model_id" in df.columns:
            sys = df["model_id"].tolist()
        elif "SYS" in df.columns:
            sys = df["SYS"].tolist()
        elif "system" in df.columns:
            sys = df["system"].tolist()
    res_correlations = None
    cnt = 0
    if p == None:
        p = np.arange(-30, 30, 0.1)
    if w == None:
        w = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for explainer, metricdict in tqdm.tqdm(attributions.items()):
        for metric, fiCollection in tqdm.tqdm(metricdict.items()):
            # Verify that a correct dataframe was given for comparison
            fiCollection.test_hyp(hyp)

            # Get linearcombination with powermean. The method for this is integrated in the explanation objects
            scores, w_indices, p_indices = fiCollection.combine_pmean_with_scores(w, p, only_hyp, normalize)
            res_list = []

            for x in range(scores.shape[0]):
                for z in range(scores.shape[2]):
                    selection = scores[x, :, z]
                    try:
                        # get correlation of human scores and our new scores
                        if kendall:
                            if sys_level:
                                sys_dict = {m: ([], []) for m in set(sys)}

                                for m, s, h, hy, sr in zip(sys, selection, da, hyp, src):
                                    if not "<v>" in hy and not "<v>" in sr:
                                        sys_dict[m][0].append(s)
                                        sys_dict[m][1].append(h)

                                s_avg = []
                                h_avg = []
                                for k, v in sys_dict.items():
                                    s_avg.append(sum(v[0]) / len(v[0]))
                                    h_avg.append(sum(v[1]) / len(v[1]))

                                res_list.append(scipy.stats.kendalltau(s_avg, h_avg)[0])


                            else:
                                res_list.append(scipy.stats.kendalltau(selection, da)[0])



                        elif spearman:
                            if sys_level:
                                sys_dict = {m: ([], []) for m in set(sys)}

                                for m, s, h in zip(sys, selection, da):
                                    sys_dict[m][0].append(s)
                                    sys_dict[m][1].append(h)

                                s_avg = []
                                h_avg = []
                                for k, v in sys_dict.items():
                                    s_avg.append(sum(v[0]) / len(v[0]))
                                    h_avg.append(sum(v[1]) / len(v[1]))

                                res_list.append(scipy.stats.spearmanr(s_avg, h_avg)[0])
                            else:
                                res_list.append(scipy.stats.spearmanr(selection, da)[0])
                        else:
                            if sys_level:
                                sys_dict = {m: ([], []) for m in set(sys)}

                                for m, s, h in zip(sys, selection, da):
                                    sys_dict[m][0].append(s)
                                    sys_dict[m][1].append(h)

                                s_avg = []
                                h_avg = []
                                for k, v in sys_dict.items():
                                    s_avg.append(sum(v[0]) / len(v[0]))
                                    h_avg.append(sum(v[1]) / len(v[1]))

                                res_list.append(scipy.stats.pearsonr(s_avg, h_avg).statistic)
                            else:
                                res_list.append(scipy.stats.pearsonr(selection, da).statistic)
                    except Exception as e:
                        print(e)

            fr = pd.DataFrame(res_list, columns=["corr"])
            fr["w"] = w_indices
            fr["p"] = p_indices
            mx = fr["corr"].max()
            max_correlation = fr[mx == fr["corr"]].head(1)
            print(max_correlation["p"].item())
            print(max_correlation["w"].item())
            print(max_correlation["corr"].item())
            new_line = pd.DataFrame({"dataset": [dataset],
                                     "lp": [lp],
                                     "metric": [metric],
                                     "explainer": [explainer],
                                     "p_max": [max_correlation["p"].item()],
                                     "w_max": [max_correlation["w"].item()],
                                     "corr": [max_correlation["corr"].item()]})
            if cnt == 0:
                res_correlations = new_line
                cnt += 1
            else:
                res_correlations = pd.concat([res_correlations, new_line])
                cnt += 1

            filetag = dataset + "__" + explainer + "__" + metric + "__" + lp
            if only_hyp:
                filetag += "__only_hyp"
            if save_fig:
                sns.set_palette("bright")
                sns.set(rc={'figure.figsize': (3.15, 1.5)})
                from matplotlib import rcParams

                #fr = fr[fr["w"] != 0]
                # figure size in inches
                rcParams['figure.figsize'] = 3.15, 1.5

                pa = sns.relplot(x="p", y="corr", hue="w", data=fr, kind="line", palette="bright")
                # rel.fig.suptitle(("Config: " + filetag))
                plt.savefig(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_images, filetag + ".pdf"))
                plt.show()
            if plot:
                rel = sns.relplot(x="p", y="corr", hue="w", height=10, data=fr)
                rel.fig.suptitle(("Config: " + filetag))
                plt.show()

            if save_data:
                fr.to_csv(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_data_dir, filetag + ".tsv"),
                          sep="\t")

    return res_correlations


def explanations_to_scores_and_eval_stratified(df, attributions, p=None, w=None, dataset="UNDEFINED", limit=True,
                                               lp="UNDEFINED",
                                               save_fig=False, plot=False, save_data=True,
                                               result_data_dir="experiment_results_stratification",
                                               result_images="experiment_graphs_pdf_stratification", only_hyp=False,
                                               kendall=False,
                                               sys_level=False, spearman=False, minimum_strat_size=200, first_ref=False,
                                               appendix=""):
    """
    :param df: dataframe to compute new scores on
    :param attributions: Explanations made with one of the projects explainers
    :param p: A list of (ideally equally spaced) p values to compute with the powermeans
    :param w: A list of (ideally equally spaced weights for combination with the original metric
    :param dataset: dataset name (important for savefile)
    :param lp: current language pair (important for savefile)
    :param save_fig: Whether to save a figure
    :param plot: whether to plot a figure
    :param save_data: Whether to save all correlations as a tsv
    :param result_data_dir: Directory name in outputs to save tsv outputs
    :param result_images: directory name in outputs to sace result images
    :param only_hyp: only consider hypothesis values if true, else gt + hyp
    :return:
    """
    # Calculate stratification parts
    hyp = df["HYP"].tolist()
    ref = df["references"].tolist()
    da = df["DA"].tolist()
    if sys_level:
        try:
            sys = df["model_id"].tolist()
        except:
            sys = df["SYS"].tolist()
    dataset = dataset + appendix
    stratification_subsets_hyp = [hyp[:minimum_strat_size]]
    stratification_subsets_ref = [ref[:minimum_strat_size]]
    stratification_subsets_da = [da[:minimum_strat_size]]
    stratification_subsets_sys = [sys[:minimum_strat_size]]
    test_subsets_hyp = []
    test_subsets_ref = []
    test_subsets_da = []
    test_subsets_sys = []
    coursor = minimum_strat_size

    strat_indices = []
    test_indices = []

    while coursor < len(hyp):
        while stratification_subsets_ref[-1][-1][0] == ref[coursor][0]:
            stratification_subsets_ref[-1].append(ref[coursor])
            stratification_subsets_hyp[-1].append(hyp[coursor])
            stratification_subsets_da[-1].append(da[coursor])
            stratification_subsets_sys[-1].append(sys[coursor])
            coursor += 1

        strat_indices.append((coursor - len(stratification_subsets_hyp[-1]), coursor))
        test_indices.append((0, coursor - len(stratification_subsets_hyp[-1]), coursor, len(hyp)))
        test_subsets_hyp.append(hyp[:coursor - len(stratification_subsets_hyp[-1])] + hyp[coursor:])
        test_subsets_ref.append(ref[:coursor - len(stratification_subsets_ref[-1])] + ref[coursor:])
        test_subsets_da.append(da[:coursor - len(stratification_subsets_da[-1])] + da[coursor:])
        test_subsets_sys.append(sys[:coursor - len(stratification_subsets_sys[-1])] + sys[coursor:])

        stratification_subsets_ref.append(ref[coursor:coursor + minimum_strat_size])
        stratification_subsets_hyp.append(hyp[coursor:coursor + minimum_strat_size])
        stratification_subsets_da.append(da[coursor:coursor + minimum_strat_size])
        stratification_subsets_sys.append(sys[coursor:coursor + minimum_strat_size])
        coursor = coursor + minimum_strat_size

    strat_indices.append((len(hyp) - len(stratification_subsets_hyp[-1]), len(hyp)))
    test_indices.append((0, len(da) - len(stratification_subsets_da[-1]), len(hyp), len(hyp)))
    test_subsets_hyp.append(hyp[:len(hyp) - len(stratification_subsets_hyp[-1])])
    test_subsets_ref.append(ref[:len(ref) - len(stratification_subsets_ref[-1])])
    test_subsets_da.append(da[:len(da) - len(stratification_subsets_da[-1])])
    test_subsets_sys.append(sys[:len(sys) - len(stratification_subsets_sys[-1])])

    res_correlations = None
    cnt = 0
    if p == None:
        p = np.arange(-30, 30, 0.1)
    if w == None:
        w = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for explainer, metricdict in tqdm.tqdm(attributions.items()):
        for metric, fiCollectionall in tqdm.tqdm(metricdict.items()):
            count = 0
            for idx, hy, sy, d, r in tqdm.tqdm(
                    zip(strat_indices, stratification_subsets_hyp, stratification_subsets_sys,
                        stratification_subsets_da, stratification_subsets_ref)):
                assert (idx[1] - idx[0] == len(hy) == len(sy) == len(d))
                fiCollection = fiCollectionall.select_range(idx)
                # Verify that a correct dataframe was given for comparison
                fiCollection.test_hyp(hy)
                if first_ref:
                    fiCollection.refs = r

                # Get linearcombination with powermean. The method for this is integrated in the explanation objects
                scores, w_indices, p_indices = fiCollection.combine_pmean_with_scores(w, p, only_hyp,
                                                                                      first_ref=first_ref)
                res_list = []
                p_list = []

                for x in range(scores.shape[0]):
                    for z in range(scores.shape[2]):
                        selection = scores[x, :, z]
                        try:
                            # get correlation of human scores and our new scores
                            if kendall:
                                if sys_level:
                                    sys_dict = {m: ([], []) for m in set(sy)}

                                    for m, s, h in zip(sy, selection, d):
                                        sys_dict[m][0].append(s)
                                        sys_dict[m][1].append(h)

                                    s_avg = []
                                    h_avg = []
                                    for k, v in sys_dict.items():
                                        s_avg.append(sum(v[0]) / len(v[0]))
                                        h_avg.append(sum(v[1]) / len(v[1]))

                                    t, pk = scipy.stats.kendalltau(s_avg, h_avg)
                                    res_list.append(t)
                                    p_list.append(pk)


                                else:
                                    res_list.append(scipy.stats.kendalltau(selection, d)[0])

                            elif spearman:
                                if sys_level:
                                    sys_dict = {m: ([], []) for m in set(sy)}

                                    for m, s, h in zip(sy, selection, d):
                                        sys_dict[m][0].append(s)
                                        sys_dict[m][1].append(h)

                                    s_avg = []
                                    h_avg = []
                                    for k, v in sys_dict.items():
                                        s_avg.append(sum(v[0]) / len(v[0]))
                                        h_avg.append(sum(v[1]) / len(v[1]))

                                    res_list.append(scipy.stats.spearmanr(s_avg, h_avg)[0])
                            else:
                                res_list.append(scipy.stats.pearsonr(selection, d).statistic)
                        except Exception as e:
                            print(e)

                fr = pd.DataFrame(res_list, columns=["corr"])
                fr["w"] = w_indices
                fr["p"] = p_indices
                mx = fr["corr"].max()
                max_correlation = fr[mx == fr["corr"]].head(1)
                print(max_correlation["p"].item())
                print(max_correlation["w"].item())
                print(max_correlation["corr"].item())
                # fr["corr_p_value"] = p_list
                new_line = pd.DataFrame({"dataset": [dataset],
                                         "lp": [lp],
                                         "metric": [metric],
                                         "explainer": [explainer],
                                         "p_max": [max_correlation["p"].item()],
                                         "w_max": [max_correlation["w"].item()],
                                         "corr": [max_correlation["corr"].item()]})
                if cnt == 0:
                    res_correlations = new_line
                    cnt += 1
                else:
                    res_correlations = pd.concat([res_correlations, new_line])
                    cnt += 1

                filetag = dataset + "__" + explainer + "__" + metric + "__" + lp
                if only_hyp:
                    filetag += "__only_hyp"
                if save_fig:
                    sns.set_theme()
                    sns.set(rc={'figure.figsize': (3.15, 2)})
                    from matplotlib import rcParams

                    # figure size in inches
                    rcParams['figure.figsize'] = 3.15, 2

                    pa = sns.relplot(x="p", y="corr", hue="w", data=fr, kind="line")
                    # rel.fig.suptitle(("Config: " + filetag))
                    plt.savefig(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_images,
                                             filetag + "_train_" + str(count) + ".pdf"))
                if plot:
                    rel = sns.relplot(x="p", y="corr", hue="w", height=10, data=fr)
                    rel.fig.suptitle(("Config: " + filetag))
                    plt.show()

                if save_data:
                    fr.to_csv(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_data_dir,
                                           filetag + "_train_" + str(count) + ".tsv"),
                              sep="\t")

                count += 1

            count = 0
            for idx, hy, sy, d, r in tqdm.tqdm(
                    zip(test_indices, test_subsets_hyp, test_subsets_sys, test_subsets_da, stratification_subsets_ref)):
                assert (len(hy) == len(sy) == len(d))
                fiCollection = fiCollectionall.select_range(idx)
                # Verify that a correct dataframe was given for comparison
                fiCollection.test_hyp(hy)
                if first_ref:
                    fiCollection.refs = r

                # Get linearcombination with powermean. The method for this is integrated in the explanation objects
                scores, w_indices, p_indices = fiCollection.combine_pmean_with_scores(w, p, only_hyp, first_ref)
                res_list = []
                p_list = []

                for x in range(scores.shape[0]):
                    for z in range(scores.shape[2]):
                        selection = scores[x, :, z]
                        try:
                            # get correlation of human scores and our new scores
                            if kendall:
                                if sys_level:
                                    sys_dict = {m: ([], []) for m in set(sy)}

                                    for m, s, h in zip(sy, selection, d):
                                        sys_dict[m][0].append(s)
                                        sys_dict[m][1].append(h)

                                    s_avg = []
                                    h_avg = []
                                    for k, v in sys_dict.items():
                                        s_avg.append(sum(v[0]) / len(v[0]))
                                        h_avg.append(sum(v[1]) / len(v[1]))

                                    t, pk = scipy.stats.kendalltau(s_avg, h_avg)
                                    res_list.append(t)
                                    p_list.append(pk)


                                else:
                                    res_list.append(scipy.stats.kendalltau(selection, d)[0])

                            elif spearman:
                                if sys_level:
                                    sys_dict = {m: ([], []) for m in set(sy)}

                                    for m, s, h in zip(sy, selection, d):
                                        sys_dict[m][0].append(s)
                                        sys_dict[m][1].append(h)

                                    s_avg = []
                                    h_avg = []
                                    for k, v in sys_dict.items():
                                        s_avg.append(sum(v[0]) / len(v[0]))
                                        h_avg.append(sum(v[1]) / len(v[1]))

                                    res_list.append(scipy.stats.spearmanr(s_avg, h_avg)[0])
                            else:
                                res_list.append(scipy.stats.pearsonr(selection, d).statistic)
                        except Exception as e:
                            print(e)

                fr = pd.DataFrame(res_list, columns=["corr"])
                fr["w"] = w_indices
                fr["p"] = p_indices
                mx = fr["corr"].max()
                max_correlation = fr[mx == fr["corr"]].head(1)
                print(max_correlation["p"].item())
                print(max_correlation["w"].item())
                print(max_correlation["corr"].item())
                # fr["corr_p_value"] = p_list
                new_line = pd.DataFrame({"dataset": [dataset],
                                         "lp": [lp],
                                         "metric": [metric],
                                         "explainer": [explainer],
                                         "p_max": [max_correlation["p"].item()],
                                         "w_max": [max_correlation["w"].item()],
                                         "corr": [max_correlation["corr"].item()]})
                if cnt == 0:
                    res_correlations = new_line
                    cnt += 1
                else:
                    res_correlations = pd.concat([res_correlations, new_line])
                    cnt += 1

                filetag = dataset + "__" + explainer + "__" + metric + "__" + lp
                if only_hyp:
                    filetag += "__only_hyp"
                if save_fig:
                    sns.set_theme()
                    sns.set(rc={'figure.figsize': (3.15, 2)})
                    from matplotlib import rcParams

                    # figure size in inches
                    rcParams['figure.figsize'] = 3.15, 2

                    pa = sns.relplot(x="p", y="corr", hue="w", data=fr, kind="line")
                    # rel.fig.suptitle(("Config: " + filetag))
                    plt.savefig(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_images,
                                             filetag + "_test_" + str(count) + ".pdf"))
                if plot:
                    rel = sns.relplot(x="p", y="corr", hue="w", height=10, data=fr)
                    rel.fig.suptitle(("Config: " + filetag))
                    plt.show()

                if save_data:
                    fr.to_csv(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_data_dir,
                                           filetag + "_test_" + str(count) + ".tsv"),
                              sep="\t")
                count += 1

    return res_correlations


def explanations_to_scores_only(df, attributions, p=None, w=None, dataset="UNDEFINED", limit=True, lp="UNDEFINED",
                                save_fig=False, plot=False, save_data=True, result_data_dir="sys_level_tables",
                                result_images="experiment_graphs_pdf", only_hyp=False, kendall=False,
                                sys_level=False, spearman=False, normalize=False, appendix="", rm_ref=False):
    """
    :param df: dataframe to compute new scores on
    :param attributions: Explanations made with one of the projects explainers
    :param p: A list of (ideally equally spaced) p values to compute with the powermeans
    :param w: A list of (ideally equally spaced weights for combination with the original metric
    :param dataset: dataset name (important for savefile)
    :param lp: current language pair (important for savefile)
    :param save_fig: Whether to save a figure
    :param plot: whether to plot a figure
    :param save_data: Whether to save all correlations as a tsv
    :param result_data_dir: Directory name in outputs to save tsv outputs
    :param result_images: directory name in outputs to sace result images
    :param only_hyp: only consider hypothesis values if true, else gt + hyp
    :return:
    """
    if rm_ref:
        df = df[~df["system"].str.contains('ref')]
    dataset = dataset + appendix
    da = df["DA"].tolist()
    hyp = df["HYP"].tolist()
    src = df["SRC"].tolist()
    seg = df["seg_id"].tolist()
    if sys_level:
        if "model_id" in df.columns:
            sys = df["model_id"].tolist()
        elif "SYS" in df.columns:
            sys = df["SYS"].tolist()
        elif "system" in df.columns:
            sys = df["system"].tolist()
    res_correlations = None
    cnt = 0
    if p == None:
        p = np.arange(-30, 30, 0.1)
    if w == None:
        w = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for explainer, metricdict in tqdm.tqdm(attributions.items()):
        for metric, fiCollection in tqdm.tqdm(metricdict.items()):
            # Verify that a correct dataframe was given for comparison
            fiCollection.test_hyp(hyp)

            # Get linearcombination with powermean. The method for this is integrated in the explanation objects
            scores, w_indices, p_indices = fiCollection.combine_pmean_with_scores(w, p, only_hyp, normalize)
            res_list = []

            for x in range(scores.shape[0]):
                for z in range(scores.shape[2]):
                    selection = scores[x, :, z]

                    if sys_level:
                        sys_dict = {m: {"bmx": [], "human": [], "p": p[x], "w": w[z], "seg": []} for m in set(
                            sys)}

                        for m, s, h, sg, hy, sr in zip(sys, selection, da, seg, hyp, src):
                            if not "<v>" in hy and not "<v>" in sr:
                                sys_dict[m]["bmx"].append(s)
                                sys_dict[m]["human"].append(h)
                                sys_dict[m]["seg"].append(sg)
                                #sys_dict[m]["src"].append(sr)

                        res_list.append(sys_dict)
                    else:
                        res_list.append({"bmx": selection.tolist(), "human": da, "p": p[x], "w": w[z]})

            filetag = dataset + "__" + explainer + "__" + metric + "__" + lp

            if save_data:
                json_object = json.dumps(res_list)

                if sys_level:
                    sys_str = "_sysLevel"
                else:
                    sys_str = "_segLevel"

                # Writing to sample.json
                with open(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_data_dir,
                                       filetag + sys_str + ".json"), "w") as outfile:
                    outfile.write(json_object)

    return None


def explanations_to_scores_only_stratified(df, attributions, p=None, w=None, dataset="UNDEFINED", limit=True,
                                           lp="UNDEFINED",
                                           save_fig=False, plot=False, save_data=True,
                                           result_data_dir="sys_level_tables",
                                           result_images="experiment_graphs_pdf_stratification", only_hyp=False,
                                           kendall=False,
                                           sys_level=False, spearman=False, minimum_strat_size=200, first_ref=False,
                                           appendix=""):
    """
    :param df: dataframe to compute new scores on
    :param attributions: Explanations made with one of the projects explainers
    :param p: A list of (ideally equally spaced) p values to compute with the powermeans
    :param w: A list of (ideally equally spaced weights for combination with the original metric
    :param dataset: dataset name (important for savefile)
    :param lp: current language pair (important for savefile)
    :param save_fig: Whether to save a figure
    :param plot: whether to plot a figure
    :param save_data: Whether to save all correlations as a tsv
    :param result_data_dir: Directory name in outputs to save tsv outputs
    :param result_images: directory name in outputs to sace result images
    :param only_hyp: only consider hypothesis values if true, else gt + hyp
    :return:
    """
    # Calculate stratification parts
    hyp = df["HYP"].tolist()
    ref = df["references"].tolist()
    da = df["DA"].tolist()
    if sys_level:
        try:
            sys = df["model_id"].tolist()
        except:
            sys = df["SYS"].tolist()
    dataset = dataset + appendix
    stratification_subsets_hyp = [hyp[:minimum_strat_size]]
    stratification_subsets_ref = [ref[:minimum_strat_size]]
    stratification_subsets_da = [da[:minimum_strat_size]]
    stratification_subsets_sys = [sys[:minimum_strat_size]]
    test_subsets_hyp = []
    test_subsets_ref = []
    test_subsets_da = []
    test_subsets_sys = []
    coursor = minimum_strat_size

    strat_indices = []
    test_indices = []

    while coursor < len(hyp):
        while stratification_subsets_ref[-1][-1][0] == ref[coursor][0]:
            stratification_subsets_ref[-1].append(ref[coursor])
            stratification_subsets_hyp[-1].append(hyp[coursor])
            stratification_subsets_da[-1].append(da[coursor])
            stratification_subsets_sys[-1].append(sys[coursor])
            coursor += 1

        strat_indices.append((coursor - len(stratification_subsets_hyp[-1]), coursor))
        test_indices.append((0, coursor - len(stratification_subsets_hyp[-1]), coursor, len(hyp)))
        test_subsets_hyp.append(hyp[:coursor - len(stratification_subsets_hyp[-1])] + hyp[coursor:])
        test_subsets_ref.append(ref[:coursor - len(stratification_subsets_ref[-1])] + ref[coursor:])
        test_subsets_da.append(da[:coursor - len(stratification_subsets_da[-1])] + da[coursor:])
        test_subsets_sys.append(sys[:coursor - len(stratification_subsets_sys[-1])] + sys[coursor:])

        stratification_subsets_ref.append(ref[coursor:coursor + minimum_strat_size])
        stratification_subsets_hyp.append(hyp[coursor:coursor + minimum_strat_size])
        stratification_subsets_da.append(da[coursor:coursor + minimum_strat_size])
        stratification_subsets_sys.append(sys[coursor:coursor + minimum_strat_size])
        coursor = coursor + minimum_strat_size

    strat_indices.append((len(hyp) - len(stratification_subsets_hyp[-1]), len(hyp)))
    test_indices.append((0, len(da) - len(stratification_subsets_da[-1]), len(hyp), len(hyp)))
    test_subsets_hyp.append(hyp[:len(hyp) - len(stratification_subsets_hyp[-1])])
    test_subsets_ref.append(ref[:len(ref) - len(stratification_subsets_ref[-1])])
    test_subsets_da.append(da[:len(da) - len(stratification_subsets_da[-1])])
    test_subsets_sys.append(sys[:len(sys) - len(stratification_subsets_sys[-1])])

    res_correlations = None
    cnt = 0
    if p == None:
        p = np.arange(-30, 30, 0.1)
    if w == None:
        w = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for explainer, metricdict in tqdm.tqdm(attributions.items()):
        for metric, fiCollectionall in tqdm.tqdm(metricdict.items()):
            count = 0
            for idx, hy, sy, d, r in tqdm.tqdm(
                    zip(strat_indices, stratification_subsets_hyp, stratification_subsets_sys,
                        stratification_subsets_da, stratification_subsets_ref)):
                assert (idx[1] - idx[0] == len(hy) == len(sy) == len(d))
                fiCollection = fiCollectionall.select_range(idx)
                # Verify that a correct dataframe was given for comparison
                fiCollection.test_hyp(hy)
                if first_ref:
                    fiCollection.refs = r

                # Get linearcombination with powermean. The method for this is integrated in the explanation objects
                scores, w_indices, p_indices = fiCollection.combine_pmean_with_scores(w, p, only_hyp)
                res_list = []

                for x in range(scores.shape[0]):
                    for z in range(scores.shape[2]):
                        selection = scores[x, :, z]
                        sys_dict = {m: {"bmx": [], "human": [], "p": p[x], "w": w[z]} for m in set(sy)}

                        for m, s, h in zip(sy, selection, d):
                            sys_dict[m]["bmx"].append(s)
                            sys_dict[m]["human"].append(h)

                        res_list.append(sys_dict)

                filetag = dataset + "__" + explainer + "__" + metric + "__" + lp

                if save_data:
                    json_object = json.dumps(res_list)

                    # Writing to sample.json
                    with open(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_data_dir,
                                           filetag + "_train_" + str(count) + ".json"), "w") as outfile:
                        outfile.write(json_object)

                count += 1
            count = 0
            for idx, hy, sy, d, r in tqdm.tqdm(
                    zip(test_indices, test_subsets_hyp, test_subsets_sys, test_subsets_da,
                        stratification_subsets_ref)):
                assert (len(hy) == len(sy) == len(d))
                fiCollection = fiCollectionall.select_range(idx)
                # Verify that a correct dataframe was given for comparison
                fiCollection.test_hyp(hy)
                if first_ref:
                    fiCollection.refs = r

                # Get linearcombination with powermean. The method for this is integrated in the explanation objects
                scores, w_indices, p_indices = fiCollection.combine_pmean_with_scores(w, p, only_hyp, first_ref)
                res_list = []
                p_list = []

                for x in range(scores.shape[0]):
                    for z in range(scores.shape[2]):
                        selection = scores[x, :, z]
                        sys_dict = {m: {"bmx": [], "human": [], "p": p[x], "w": w[z]} for m in set(sy)}

                        for m, s, h in zip(sy, selection, d):
                            sys_dict[m]["bmx"].append(s)
                            sys_dict[m]["human"].append(h)

                        res_list.append(sys_dict)

                filetag = dataset + "__" + explainer + "__" + metric + "__" + lp

                if save_data:
                    json_object = json.dumps(res_list)

                    # Writing to sample.json
                    with open(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_data_dir,
                                           filetag + "_test_" + str(count) + ".json"), "w") as outfile:
                        outfile.write(json_object)

                count += 1

    return res_correlations


if __name__ == '__main__':
    p = np.arange(-30, 30, 0.1)
    print(len(p))
