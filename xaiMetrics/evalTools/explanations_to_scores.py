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
                                    save_fig=False, plot=False, save_data=True, result_data_dir = "experiment_results",
                                    result_images = "experiment_graphs_pdf", only_hyp=False):
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
    da = df["DA"].tolist()
    hyp = df["HYP"].tolist()
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
            scores, w_indices, p_indices = fiCollection.combine_pmean_with_scores(w, p, only_hyp)
            res_list = []

            for x in range(scores.shape[0]):
                for z in range(scores.shape[2]):
                    selection = scores[x, :, z]
                    try:
                        # get correlation of human scores and our new scores
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
                sns.set_theme()
                sns.set(rc={'figure.figsize': (3.15, 3)})
                sns.relplot(x="p", y="corr", hue="w", data=fr, kind="line")
                #rel.fig.suptitle(("Config: " + filetag))
                plt.savefig(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_images, filetag + ".pdf"))
                plt.show(block=True)
            if plot:
                rel = sns.relplot(x="p", y="corr", hue="w", height=10, data=fr)
                rel.fig.suptitle(("Config: " + filetag))
                plt.show()

            if save_data:
                fr.to_csv(os.path.join(ROOT_DIR, "xaiMetrics", "outputs", result_data_dir, filetag + ".tsv"),
                          sep="\t")

    return res_correlations


if __name__ == '__main__':
    p = np.arange(-30, 30, 0.1)
    print(len(p))
