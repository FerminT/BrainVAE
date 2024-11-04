from pandas import read_csv
from pathlib import Path
from scripts.constants import EVALUATION_PATH
from scipy.interpolate import interp1d
from scipy.stats import sem
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import precision_recall_curve, mean_absolute_error, accuracy_score


CFGS_RENAMING = {'default': 'Age-agnostic',
                 'invariant_float': 'Age-invariant',
                 'age_predictor': 'Age-aware',
                 'bmi_invariant': 'BMI-invariant',
                 'gender_invariant': 'Gender-invariant',
                 'age': 'Age',
                 'baseline': 'Shuffled'}


def plot(results_path, cfgs, target_labels, bars, age_windows):
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
        for label in target_labels:
            results = read_csv(Path(model, f'{label}_predictions.csv'))
            labels_results[label][name] = results

    age_windows_ranges = {label: {} for label in target_labels}
    if age_windows > 0:
        for label in target_labels:
            for model in evaluated_cfgs:
                model_at_label = labels_results[label][model]
                model_at_label['age_window'] = pd.qcut(
                    model_at_label['age_at_scan'], age_windows, labels=False
                )
                age_windows_ranges[label] = {f'window_{i}': (model_at_label[model_at_label['age_window'] == i]['age_at_scan'].min(),
                                                             model_at_label[model_at_label['age_window'] == i]['age_at_scan'].max())
                                              for i in range(age_windows)}

    if bars:
        metrics = compute_metrics(labels_results, target_labels, evaluated_cfgs)
        plot_bar_plots(metrics, target_labels, evaluated_cfgs, results_path)
    else:
        roc_curves = build_roc_curves(labels_results, target_labels, evaluated_cfgs, age_windows)
        pr_curves = build_precision_recall_curves(labels_results, target_labels, evaluated_cfgs, age_windows)
        plot_curves(roc_curves, 'False Positive Rate', 'True Positive Rate', True, 25,
                    results_path / 'roc_curves.png', age_windows_ranges)
        plot_curves(pr_curves, 'Recall', 'Precision', False, 25,
                    results_path / 'pr_curves.png', age_windows_ranges)


def plot_bar_plots(metrics, target_labels, evaluated_cfgs, results_path):
    sns.set_theme(font_scale=1.5)
    fig, axs = plt.subplots(1, len(target_labels), figsize=(10, 6))

    for ax, label in zip(axs.flat, target_labels):
        data = []
        for model in metrics[label]:
            if 'MAE_mean' in metrics[label][model]:
                data.append({
                    'Model': model,
                    'Metric': 'MAE',
                    'Value': metrics[label][model]['MAE_mean'],
                    'Error': metrics[label][model]['MAE_stderr']
                })
                data.append({
                    'Model': model,
                    'Metric': 'Correlation',
                    'Value': metrics[label][model]['Correlation_mean'],
                    'Error': metrics[label][model]['Correlation_stderr']
                })
            if 'Accuracy_mean' in metrics[label][model]:
                data.append({
                    'Model': model,
                    'Metric': 'Accuracy',
                    'Value': metrics[label][model]['Accuracy_mean'],
                    'Error': metrics[label][model]['Accuracy_stderr']
                })

        df = pd.DataFrame(data)
        metric = 'MAE' if 'MAE' in df['Metric'].values else 'Accuracy'
        sns.barplot(x='Model', y='Value', hue='Model', data=df[df['Metric'] == metric], ax=ax, errorbar=None, width=1.0)
        for i, bar in enumerate(ax.patches):
            error = df.iloc[i // len(metrics[label])]['Error']
            ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', c='black')

        if metric == 'MAE':
            ax2 = ax.twinx()
            sns.lineplot(x='Model', y='Value', data=df[df['Metric'] == 'Correlation'], ax=ax2,
                         marker='o', linestyle='--', color='red', markeredgecolor='black')
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
    colors = sns.color_palette('tab10', len(evaluated_cfgs))
    fig.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               labels=evaluated_cfgs, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=len(evaluated_cfgs),
               fontsize='large')
    fig.savefig(results_path / 'bar_plots.png', format='png', bbox_inches='tight', transparent=True)
    plt.subplots_adjust(wspace=1.0)
    plt.show()


def compute_metrics(labels_results, target_labels, evaluated_cfgs):
    metrics = {label: {} for label in target_labels}
    for label in target_labels:
        for model in evaluated_cfgs:
            model_results = labels_results[label][model]
            mae_list = []
            corr_list = []
            acc_list = []
            for run in model_results.columns:
                if run.startswith('pred_'):
                    predictions = model_results[run].values
                    true_values = model_results['label'].values

                    if not np.array_equal(true_values, true_values.astype(bool)):
                        mae = mean_absolute_error(true_values, predictions)
                        corr = np.corrcoef(true_values, predictions)[0, 1]
                        mae_list.append(mae)
                        corr_list.append(corr)
                    else:
                        acc = accuracy_score(true_values, predictions >= 0.5)
                        acc_list.append(acc)
            if mae_list:
                metrics[label][model] = {
                    'MAE_mean': np.mean(mae_list),
                    'MAE_stderr': sem(mae_list),
                    'Correlation_mean': np.mean(corr_list) if np.mean(corr_list) > 0 else 0,
                    'Correlation_stderr': sem(corr_list)
                }
            if acc_list:
                metrics[label][model] = {
                    'Accuracy_mean': np.mean(acc_list),
                    'Accuracy_stderr': sem(acc_list)
                }

    return metrics


def plot_curves(curves, xlabel, ylabel, identity_line, fontsize, filename, age_windows_ranges):
    sns.set_theme()
    if any(age_windows_ranges):
        for label in curves:
            n_windows = len(age_windows_ranges[label].keys())
            fig, axs = plt.subplots(1, n_windows, figsize=(18, 7), sharey=True)
            label_age_ranges = age_windows_ranges[label]
            for window in range(n_windows):
                window_age_range = label_age_ranges[f'window_{window}']
                for model in curves[label]:
                    if f'window_{window}' in model:
                        mean_curve = curves[label][model]['mean']
                        stderr_curve = curves[label][model]['stderr']
                        mean_fpr = [x[0] for x in mean_curve]
                        mean_tpr = [x[1] for x in mean_curve]
                        stderr_tpr = [x[1] for x in stderr_curve]

                        axs[window].plot(mean_fpr, mean_tpr, label=model)
                        axs[window].fill_between(mean_fpr,
                                                 np.array(mean_tpr) - np.array(stderr_tpr),
                                                 np.array(mean_tpr) + np.array(stderr_tpr),
                                                 alpha=0.2)
                        print(f'{label} Ages {window_age_range[0]}-{window_age_range[1]} {model} '
                              f'AUC: {np.trapz(mean_tpr, mean_fpr):.2f}')

                axs[window].set_xlabel(xlabel, fontsize=fontsize)
                if identity_line:
                    axs[window].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                if window == 0:
                    axs[window].set_ylabel(ylabel, fontsize=fontsize)
                window_title = f'Age {window_age_range[0]:1f}-{window_age_range[1]:1f}'
                axs[window].set_title(window_title, fontsize=fontsize)
            fig.suptitle(f'{label.upper()}', fontsize=fontsize)
            fig.patch.set_alpha(0)
            plt.subplots_adjust(wspace=0.3)

            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=fontsize)
            plt.savefig(f'{filename.stem}_{label}{filename.suffix}', format='png', bbox_inches='tight')
            plt.show()
    else:
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


def build_roc_curves(labels_results, labels, models, age_windows):
    thresholds = np.linspace(0, 1, 100)
    roc_curves = {label: {} for label in labels}
    for label in labels:
        for model in models:
            if age_windows > 0:
                for window in range(age_windows):
                    roc_curves[label][f'{model}_window_{window}'] = {'mean': [], 'stderr': []}
                    all_fpr = []
                    all_tpr = []
                    window_data = labels_results[label][model][labels_results[label][model]['age_window'] == window]
                    for run in window_data.columns:
                        if run.startswith('pred_'):
                            fpr_tpr = []
                            for threshold in thresholds:
                                tp, fp, tn, fn = count_tp_fp_tn_fn(window_data[run].values,
                                                                   window_data['label'].values, threshold)
                                tpr = tp / (tp + fn)
                                fpr = fp / (fp + tn)
                                fpr_tpr.append((fpr, tpr))
                            all_fpr.append([x[0] for x in fpr_tpr])
                            all_tpr.append([x[1] for x in fpr_tpr])
                    mean_fpr = np.mean(all_fpr, axis=0)
                    mean_tpr = np.mean(all_tpr, axis=0)
                    stderr_tpr = np.std(all_tpr, axis=0) / np.sqrt(len(all_tpr))
                    roc_curves[label][f'{model}_window_{window}']['mean'] = list(zip(mean_fpr, mean_tpr))
                    roc_curves[label][f'{model}_window_{window}']['stderr'] = list(zip(mean_fpr, stderr_tpr))
            else:
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


def build_precision_recall_curves(labels_results, labels, models, age_windows):
    pr_curves = {label: {} for label in labels}
    for label in labels:
        for model in models:
            if age_windows > 0:
                for window in range(age_windows):
                    pr_curves[label][f'{model}_window_{window}'] = {'mean': [], 'stderr': []}
                    all_precision = []
                    all_recall = []
                    window_data = labels_results[label][model][labels_results[label][model]['age_window'] == window]
                    for run in window_data.columns:
                        if run.startswith('pred_'):
                            precision, recall, _ = precision_recall_curve(
                                window_data['label'].values,
                                window_data[run].values
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
                    pr_curves[label][f'{model}_window_{window}']['mean'] = list(zip(common_recall, mean_precision))
                    pr_curves[label][f'{model}_window_{window}']['stderr'] = list(zip(common_recall, std_error_precision))
            else:
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
    parser.add_argument('-b', '--bars', action='store_true', help='plot bars instead of curves')
    parser.add_argument('-c', '--cfgs', nargs='+', type=str, default=['default', 'invariant_float', 'age_predictor'],
                        help='configurations to plot')
    parser.add_argument('-w', '--age_windows', type=int, default=0,
                        help='Divide classifications in n equidistant age windows')
    parser.add_argument('--set', type=str, default='val', help='set to plot evaluations from (val or test)')
    args = parser.parse_args()
    results_path = Path(EVALUATION_PATH, args.dataset, args.set)

    plot(results_path, args.cfgs, args.targets, args.bars, args.age_windows)
