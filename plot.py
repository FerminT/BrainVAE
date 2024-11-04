from pandas import read_csv
from pathlib import Path
from scripts.constants import EVALUATION_PATH
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve


CFGS_RENAMING = {'default': 'Age-agnostic',
                 'invariant_float': 'Age-invariant',
                 'age_predictor': 'Age-aware',
                 'bmi_invariant': 'BMI-invariant',
                 'gender_invariant': 'Gender-invariant',
                 'age': 'Age',
                 'baseline': 'Shuffled'}


def plot(results_path, cfgs, target_labels):
    evaluated_cfgs = []
    labels_results = {label: {} for label in target_labels}
    for cfg in cfgs:
        cfg_path = Path(results_path, cfg)
        if len(Path(cfg).parts) == 1:
            models = [dir_ for dir_ in cfg_path.iterdir() if dir_.is_dir()]
            if len(models) == 0:
                raise ValueError(f'No models found in {cfg_path}')
            model = models[-1]
            name = CFGS_RENAMING.get(cfg, cfg)
            evaluated_cfgs.append(name)
        else:
            model = cfg_path
            name = CFGS_RENAMING.get(model.parent.name, model.parent.name)
            evaluated_cfgs.append(name)
        for target in target_labels:
            results = read_csv(Path(model, f'{target}_predictions.csv'))
            labels_results[target][name] = results

    roc_curves = build_roc_curves(labels_results, target_labels, evaluated_cfgs)
    pr_curves = build_precision_recall_curves(labels_results, target_labels, evaluated_cfgs)
    plot_curves(roc_curves, 'False Positive Rate', 'True Positive Rate', True, 25, results_path / 'roc_curves.png')
    plot_curves(pr_curves, 'Recall', 'Precision', False, 25, results_path / 'pr_curves.png')


def plot_curves(curves, xlabel, ylabel, identity_line, fontsize, filename):
    sns.set_theme()
    fig, axs = plt.subplots(1, len(curves.keys()), figsize=(18, 7), sharey=True)

    for i, label in enumerate(curves):
        for model in curves[label]:
            mean_curve = curves[label][model]['mean']
            stderr_curve = curves[label][model]['stderr']
            mean_fpr = [x[0] for x in mean_curve]
            mean_tpr = [x[1] for x in mean_curve]
            stderr_tpr = [x[1] for x in stderr_curve]

            axs[i].plot(mean_fpr, mean_tpr, label=model)
            axs[i].fill_between(mean_fpr,
                                np.array(mean_tpr) - np.array(stderr_tpr),
                                np.array(mean_tpr) + np.array(stderr_tpr),
                                alpha=0.2)
            print(f'{label} {model} AUC: {np.trapz(mean_tpr, mean_fpr)}')
        axs[i].set_xlabel(xlabel, fontsize=fontsize)
        if identity_line:
            axs[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        if i == 0:
            axs[i].set_ylabel(ylabel, fontsize=fontsize)
        axs[i].set_title(label.upper(), fontsize=fontsize)
    fig.patch.set_alpha(0)
    plt.subplots_adjust(wspace=0.05)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.00), ncol=len(labels), fontsize=fontsize)
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.show()


def build_roc_curves(labels_results, labels, models):
    thresholds = np.linspace(0, 1, 100)
    roc_curves = {label: {} for label in labels}
    for label in labels:
        for model in models:
            roc_curves[label][model] = {'mean': [], 'stderr': []}
            all_fpr = []
            all_tpr = []

            for run in labels_results[label][model].columns:
                if run.startswith('pred_'):
                    fpr_tpr = []
                    for threshold in thresholds:
                        tp, fp, tn, fn = count_tp_fp_tn_fn(labels_results[label][model][run].values,
                                                           labels_results[label][model]['label'].values, threshold)
                        tpr = tp / (tp + fn)
                        fpr = fp / (fp + tn)
                        fpr_tpr.append((fpr, tpr))
                    all_fpr.append([x[0] for x in fpr_tpr])
                    all_tpr.append([x[1] for x in fpr_tpr])

            mean_fpr = np.mean(all_fpr, axis=0)
            mean_tpr = np.mean(all_tpr, axis=0)
            stderr_tpr = np.std(all_tpr, axis=0) / np.sqrt(len(all_tpr))

            roc_curves[label][model]['mean'] = list(zip(mean_fpr, mean_tpr))
            roc_curves[label][model]['stderr'] = list(zip(mean_fpr, stderr_tpr))

    return roc_curves


def build_precision_recall_curves(labels_results, labels, models):
    pr_curves = {label: {} for label in labels}
    for label in labels:
        for model in models:
            pr_curves[label][model] = {'mean': [], 'stderr': []}
            all_precision = []
            all_recall = []

            for run in labels_results[label][model].columns:
                if run.startswith('pred_'):
                    precision, recall, _ = precision_recall_curve(
                        labels_results[label][model]['label'].values,
                        labels_results[label][model][run].values
                    )
                    all_precision.append(precision)
                    all_recall.append(recall)

            common_recall = np.linspace(0, 1, 100)
            interpolated_precisions = []
            for precision, recall in zip(all_precision, all_recall):
                interp_func = interp1d(recall, precision, bounds_error=False, fill_value=(0, 0))
                interpolated_precisions.append(interp_func(common_recall))
            interpolated_precisions = np.array(interpolated_precisions)
            mean_precision = np.mean(interpolated_precisions, axis=0)
            std_error_precision = np.std(interpolated_precisions, axis=0) / np.sqrt(len(interpolated_precisions))
            pr_curves[label][model]['mean'] = list(zip(common_recall, mean_precision))
            pr_curves[label][model]['stderr'] = list(zip(common_recall, std_error_precision))

    return pr_curves


def count_tp_fp_tn_fn(predictions, labels, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(predictions)):
        if predictions[i] >= threshold:
            if labels[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if labels[i] == 1:
                fn += 1
            else:
                tn += 1
    return tp, fp, tn, fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='diseased', help='dataset where to look for results')
    parser.add_argument('-t', '--targets', type=str, nargs='+', default=['dvh', 'dvp', 'hvp'],
                        help='target labels to plot')
    parser.add_argument('-c', '--cfgs', nargs='+', type=str, default=['default', 'invariant_float', 'age_predictor'],
                        help='configurations to plot')
    parser.add_argument('--set', type=str, default='val', help='set to plot evaluations from (val or test)')
    args = parser.parse_args()
    results_path = Path(EVALUATION_PATH, args.dataset, args.set)

    plot(results_path, args.cfgs, args.targets)
