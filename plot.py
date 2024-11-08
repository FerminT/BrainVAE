from pathlib import Path
from scripts.constants import EVALUATION_PATH
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve
from scripts.utils import load_predictions, get_age_windows, compute_metrics, metrics_to_df

CFGS_RENAMING = {'default': 'Age-agnostic',
                 'invariant_float': 'Age-invariant',
                 'age_predictor': 'Age-aware',
                 'bmi_invariant': 'BMI-invariant',
                 'gender_invariant': 'Sex-invariant',
                 'age': 'Age',
                 'baseline': 'Shuffled'}


def plot(results_path, cfgs, target_labels, bars, age_windows):
    labels_predictions, evaluated_cfgs = load_predictions(target_labels, cfgs, results_path)
    age_windows_ranges = get_age_windows(labels_predictions, target_labels, evaluated_cfgs, age_windows)

    if bars:
        metrics = compute_metrics(labels_predictions, target_labels, evaluated_cfgs)
        plot_bar_plots(metrics, target_labels, evaluated_cfgs, results_path)
    else:
        roc_curves = build_roc_curves(labels_predictions, target_labels, evaluated_cfgs, age_windows)
        pr_curves = build_precision_recall_curves(labels_predictions, target_labels, evaluated_cfgs, age_windows)
        plot_curves(roc_curves, 'False Positive Rate', 'True Positive Rate', True, 25,
                    results_path / 'roc_curves.png', age_windows_ranges)
        plot_curves(pr_curves, 'Recall', 'Precision', False, 25,
                    results_path / 'pr_curves.png', age_windows_ranges)


def plot_bar_plots(metrics, target_labels, evaluated_cfgs, results_path):
    sns.set_theme(font_scale=1.5)
    fig, axs = plt.subplots(1, len(target_labels), figsize=(12, 6))

    for ax, label in zip(axs.flat, target_labels):
        data = metrics_to_df(metrics, label)
        metric = 'MAE' if 'MAE' in data['Metric'].values else 'Accuracy'
        sns.barplot(x='Model', y='Value', hue='Model', data=data[data['Metric'] == metric], ax=ax, errorbar=None,
                    width=1.0)
        for i, bar in enumerate(ax.patches):
            error = data.iloc[i // len(metrics[label])]['Error']
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', c='black')

        if metric == 'MAE':
            ax2 = ax.twinx()
            sns.lineplot(x='Model', y='Value', data=data[data['Metric'] == 'Correlation'], ax=ax2,
                         marker='o', linestyle='--', color='#d4aa00ff', markeredgecolor='black')
            ax2.set_ylabel('Correlation')
            ax2.set_ylim(0, 1)
            ax2.grid(False)

        ax.set_title(label)
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.grid(False)
        ax.set_xticklabels([])

    fig.tight_layout()
    fig.patch.set_alpha(0)
    colors = sns.color_palette(n_colors=len(evaluated_cfgs))
    fig.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               labels=evaluated_cfgs, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(evaluated_cfgs) // 2,
               fontsize='large')
    fig.savefig(results_path / 'bar_plots.png', format='png', bbox_inches='tight', transparent=True)
    plt.subplots_adjust(wspace=0.8)
    plt.show()


def plot_curves(curves, xlabel, ylabel, identity_line, fontsize, filename, age_windows_ranges):
    sns.set_theme()
    has_windows = any(age_windows_ranges.values())
    for i, label in enumerate(curves):
        n_columns = len(age_windows_ranges[label].keys()) if has_windows else len(curves.keys())
        fig, axs = plt.subplots(1, n_columns, figsize=(18, 7), sharey=True)
        fig.patch.set_alpha(0)
        plt.subplots_adjust(wspace=0.03)
        if has_windows:
            fig.suptitle(f'{label.upper()}', fontsize=fontsize)
            label_age_ranges = age_windows_ranges[label]
            filename = filename.parent / f'age_{filename.stem}_{label}{filename.suffix}'
            for window in range(n_columns):
                for model in curves[label]:
                    if f'window_{window}' in model:
                        plot_mean(curves[label][model]['mean'], curves[label][model]['stderr'], model.split('_')[0],
                                  axs[window])

                window_age_range = label_age_ranges[f'window_{window}']
                window_title = f'Age {window_age_range[0]:.1f}-{window_age_range[1]:.1f}'
                configure_axes(axs[window], xlabel, ylabel, identity_line, fontsize, window_title, window == 0)
        else:
            for model in curves[label]:
                plot_mean(curves[label][model]['mean'], curves[label][model]['stderr'], model, axs[i])
            configure_axes(axs[i], xlabel, ylabel, identity_line, fontsize, label, i == 0)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.00),
                   ncol=len(labels), fontsize=fontsize)
        plt.savefig(filename, format='png', bbox_inches='tight')
        plt.show()


def plot_mean(mean_data, stderr_data, model_label, ax):
    mean_fpr = [x[0] for x in mean_data]
    mean_tpr = [x[1] for x in mean_data]
    stderr_tpr = [x[1] for x in stderr_data]
    ax.plot(mean_fpr, mean_tpr, label=model_label)
    ax.fill_between(mean_fpr, np.array(mean_tpr) - np.array(stderr_tpr), np.array(mean_tpr) + np.array(stderr_tpr),
                    alpha=0.2)


def configure_axes(ax, xlabel, ylabel, identity_line, fontsize, label, is_first_column):
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if identity_line:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    if is_first_column:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(label.upper(), fontsize=fontsize)


def build_roc_curves(labels_results, labels, models, age_windows):
    thresholds = np.linspace(0, 1, 100)
    roc_curves = {label: {} for label in labels}
    for label in labels:
        for model in models:
            if age_windows > 0:
                for window in range(age_windows):
                    window_data = labels_results[label][model][labels_results[label][model]['age_window'] == window]
                    mean_roc(window_data, thresholds, label, f'{model}_window_{window}', roc_curves)
            else:
                mean_roc(labels_results[label][model], thresholds, label, model, roc_curves)
    return roc_curves


def mean_roc(data, thresholds, label, model_name, roc_curves):
    all_fpr, all_tpr = [], []
    for run in data.columns:
        if run.startswith('pred_'):
            fpr_tpr = []
            for threshold in thresholds:
                tp, fp, tn, fn = count_tp_fp_tn_fn(data[run].values, data['label'].values, threshold)
                tpr = tp / (tp + fn)
                fpr = fp / (fp + tn)
                fpr_tpr.append((fpr, tpr))
            all_fpr.append([x[0] for x in fpr_tpr])
            all_tpr.append([x[1] for x in fpr_tpr])
    mean_fpr, mean_tpr = np.mean(all_fpr, axis=0), np.mean(all_tpr, axis=0)
    stderr_tpr = np.std(all_tpr, axis=0) / np.sqrt(len(all_tpr))
    roc_curves[label][model_name] = {'mean': list(zip(mean_fpr, mean_tpr)), 'stderr': list(zip(mean_fpr, stderr_tpr))}


def build_precision_recall_curves(labels_results, labels, models, age_windows):
    pr_curves = {label: {} for label in labels}
    common_recall = np.linspace(0, 1, 100)
    for label in labels:
        for model in models:
            model_preds = labels_results[label][model]
            if age_windows > 0:
                for window in range(age_windows):
                    window_data = model_preds[model_preds['age_window'] == window]
                    mean_pr(window_data, common_recall, label, f'{model}_window_{window}', pr_curves)
            else:
                mean_pr(model_preds, common_recall, label, model, pr_curves)
    return pr_curves


def mean_pr(data, common_recall, label, model_name, pr_curves):
    all_precision = []
    all_recall = []
    for run in data.columns:
        if run.startswith('pred_'):
            precision, recall, _ = precision_recall_curve(data['label'].values, data[run].values)
            all_precision.append(precision)
            all_recall.append(recall)
    interpolated_precisions = []
    for precision, recall in zip(all_precision, all_recall):
        interp_func = interp1d(recall, precision, bounds_error=False, fill_value=(0, 0))
        interpolated_precisions.append(interp_func(common_recall))
    interpolated_precisions = np.array(interpolated_precisions)
    mean_precision = np.mean(interpolated_precisions, axis=0)
    std_error_precision = np.std(interpolated_precisions, axis=0) / np.sqrt(len(interpolated_precisions))
    pr_curves[label][model_name] = {'mean': list(zip(common_recall, mean_precision)),
                                    'stderr': list(zip(common_recall, std_error_precision))}


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
    parser.add_argument('-b', '--bars', action='store_true', help='plot bars instead of curves')
    parser.add_argument('-c', '--cfgs', nargs='+', type=str, default=['invariant_float', 'default', 'age_predictor'],
                        help='configurations to plot')
    parser.add_argument('-w', '--age_windows', type=int, default=0,
                        help='Divide classifications in n equidistant age windows')
    parser.add_argument('--set', type=str, default='val', help='set to plot evaluations from (val or test)')
    args = parser.parse_args()
    results_path = Path(EVALUATION_PATH, args.dataset, args.set)

    plot(results_path, args.cfgs, args.targets, args.bars, args.age_windows)
