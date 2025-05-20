from pathlib import Path
from scripts.constants import EVALUATION_PATH
from scipy.interpolate import interp1d
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import precision_recall_curve, auc
from scripts.utils import load_predictions, get_age_windows, compute_metrics, metrics_to_df


def plot(results_path, cfgs, target_labels, bars, age_windows):
    labels_predictions, evaluated_cfgs = load_predictions(target_labels, cfgs, results_path)
    age_windows_ranges = get_age_windows(labels_predictions, target_labels, evaluated_cfgs, age_windows)

    if bars:
        metrics = compute_metrics(labels_predictions, target_labels, evaluated_cfgs)
        plot_bar_plots(metrics, evaluated_cfgs, results_path)
    else:
        roc_curves = build_roc_curves(labels_predictions, target_labels, evaluated_cfgs, age_windows)
        pr_curves = build_precision_recall_curves(labels_predictions, target_labels, evaluated_cfgs, age_windows)
        plot_data(roc_curves, evaluated_cfgs, '', 'ROC-AUC', (0.4, 1.0), False, 25,
                  results_path / 'roc_aucs.png', age_windows_ranges, type='auc')
        plot_data(pr_curves, evaluated_cfgs, '', 'PR-AUC', (0.4, 1.0), False, 25,
                  results_path / 'pr_aucs.png', age_windows_ranges, type='auc')


def plot_bar_plots(metrics, evaluated_cfgs, results_path):
    sns.set_theme(font_scale=1.5)
    target_labels = list(metrics.keys())
    fig, axs = plt.subplots(1, len(target_labels), figsize=(4 * len(target_labels), 6))
    axs = axs.flat if len(target_labels) > 1 else [axs]

    for ax, label in zip(axs, target_labels):
        data = metrics_to_df(metrics, label)
        if 'MAE' in data['Metric'].values:
            metric = 'MAE'
        elif 'MSE' in data['Metric'].values:
            metric = 'MSE'
        elif 'Correlation' in data['Metric'].values:
            metric = 'Correlation'
        else:
            metric = 'Accuracy'
        data_metric = data[data['Metric'] == metric]
        bars = sns.barplot(x='Model', y='Value', hue='Model', data=data_metric, ax=ax, errorbar=None,
                           width=1.0, alpha=1.0, dodge=False)
        for bar, model, error in zip(bars.patches, data_metric['Model'], data_metric['Error']):
            if model == 'Random':
                color = bar.get_facecolor()
                bar.set_facecolor('none')
                bar.set_edgecolor(color)
            else:
                ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), yerr=error, fmt='none', c='black')

        if metric == 'MSE':
            heights_drawn = []
            offset, text_height, separation = 0.02, 0.002, 0.05
            for i, bar in enumerate(ax.patches):
                for j in range(i + 1, len(ax.patches)):
                    model1, model2 = data.iloc[i]['Model'], data.iloc[j]['Model']
                    if model1 == 'Random' or model2 == 'Random':
                        continue
                    significance = metrics[label][model1][f'{model2}_significance']
                    max_height = max(bar.get_height(), ax.patches[j].get_height())
                    height_found = True
                    while height_found:
                        if max_height in heights_drawn:
                            max_height += separation
                        else:
                            height_found = False
                    plot_significance_against(ax, [i, j], max_height, significance, text_height=text_height,
                                              offset=offset, ns=True)
                    heights_drawn.append(max_height)
            fig.set_figheight(6 + max(heights_drawn))

        ax.set_title(label)
        ax.set_ylabel(metric)
        ax.set_xlabel('')
        ax.grid(False)
        ax.set_xticklabels([])

    fig.tight_layout()
    fig.patch.set_alpha(0)
    colors = sns.color_palette(n_colors=len(evaluated_cfgs))
    ncols = len(evaluated_cfgs) if len(evaluated_cfgs) < 6 else len(evaluated_cfgs) // 2
    fig.legend(handles=[plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               labels=evaluated_cfgs, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=ncols,
               fontsize='large')
    fig.savefig(results_path / 'bar_plots.png', format='png', bbox_inches='tight', transparent=True, dpi=150)
    plt.subplots_adjust(wspace=0.8)
    plt.show()


def plot_data(data, evaluated_cfgs, xlabel, ylabel, ylim, identity_line, fontsize, filename, age_windows_ranges, type):
    sns.set_theme()
    has_windows = any(age_windows_ranges.values())
    fig, axs = create_subplots(1, len(data.keys()), figsize=(18, 7), sharey=True)
    colors = sns.color_palette(n_colors=len(evaluated_cfgs))
    handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]

    for i, label in enumerate(data):
        if has_windows:
            n_columns = len(age_windows_ranges[label].keys())
            fig, axs = create_subplots(1, n_columns, figsize=(18, 7), sharey=True)
            fig.suptitle(f'{label.upper()}', fontsize=fontsize)
            label_age_ranges = age_windows_ranges[label]
            filename = filename.parent / f'age_{filename.stem}_{label}{filename.suffix}'
            for window in range(n_columns):
                for model in data[label]:
                    if f'window_{window}' in model:
                        model_name = model.split('_')[0]
                        if type == 'curve':
                            plot_mean(data[label][model]['mean'], data[label][model]['stderr'], model_name, axs[window])
                        else:
                            plot_violin(data[label], 'aucs', axs[window], colors)

                window_age_range = label_age_ranges[f'window_{window}']
                window_title = f'Age {window_age_range[0]:.1f}-{window_age_range[1]:.1f}'
                configure_axes(axs[window], xlabel, ylabel, ylim, identity_line, fontsize, window_title, window == 0)
            show_plot(fig, (handles, evaluated_cfgs), fontsize, filename)
        else:
            if type == 'curve':
                for model in data[label]:
                    plot_mean(data[label][model]['mean'], data[label][model]['stderr'], model, axs[i])
            else:
                plot_violin(data[label], 'aucs', axs[i], colors)
            configure_axes(axs[i], xlabel, ylabel, ylim, identity_line, fontsize, label, i == 0)
    if not has_windows:
        show_plot(fig, (handles, evaluated_cfgs), fontsize, filename)


def plot_violin(data, results_label, ax, colors):
    results_df = DataFrame.from_dict(data, orient='index')
    significance_to_invariant = significance_against(results_df, results_label, base_model='Age-invariant')
    significance_to_baseline = significance_against(results_df, results_label, base_model='Age-agnostic')
    results_df = results_df.reset_index().rename(columns={'index': 'Model'})
    results_df = results_df.explode(results_label)[['Model', results_label]]
    results_df[results_label] = results_df[results_label].astype(float)
    sns.violinplot(x='Model', y=results_label, data=results_df, hue='Model', ax=ax, palette=colors)
    add_significance_asterisks(ax, results_df, results_label, significance_to_invariant,
                               reference_model='Age-invariant')
    add_significance_to_baseline(ax, results_df, results_label, significance_to_baseline, reference_model='Age-aware',
                                 base_model='Age-agnostic')
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)


def add_significance_asterisks(ax, results_df, results_label, pvalues, reference_model):
    for i, model in enumerate(results_df['Model'].unique()):
        if model != reference_model:
            p_value = pvalues[model]
            if p_value < 0.05:
                y = results_df[results_df['Model'] == model][results_label].max()
                ax.annotate(significance_asterisks(p_value), xy=(i, y + 0.05), ha='center', va='center', fontsize=20)


def add_significance_to_baseline(ax, results_df, results_label, pvalues, reference_model, base_model):
    significance = pvalues[reference_model]
    reference_index = results_df[results_df['Model'] == reference_model].index[0]
    base_index = results_df[results_df['Model'] == base_model].index[0]
    y = max(results_df[results_df['Model'] == reference_model][results_label].max(),
            results_df[results_df['Model'] == base_model][results_label].max())
    plot_significance_against(ax, [base_index, reference_index], y, significance)


def plot_significance_against(ax, indices, y, p_value, text_height=0.005, offset=0.1, ns=False):
    asterisks = significance_asterisks(p_value, ns)
    if len(asterisks):
        y += offset
        ax.plot([indices[0], indices[0], indices[1], indices[1]], [y, y+text_height, y+text_height, y], lw=1.5,
                color='black')
        ax.text((indices[0] + indices[1]) / 2, y+text_height, asterisks, ha='center', va='bottom', color='black',
                fontsize=20)


def significance_asterisks(p, ns=False):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns' if ns else ''


def significance_against(results_df, results_label, base_model):
    models_significance = {}
    for model in results_df.index:
        if model != base_model:
            model_results = np.array(results_df.loc[model, results_label])
            base_results = np.array(results_df.loc[base_model, results_label])
            models_significance[model] = 1.0 - (model_results > base_results).sum() / len(model_results)
    return models_significance


def plot_mean(mean_data, stderr_data, model_label, ax):
    mean_fpr = [x[0] for x in mean_data]
    mean_tpr = [x[1] for x in mean_data]
    stderr_tpr = [x[1] for x in stderr_data]
    ax.plot(mean_fpr, mean_tpr, label=model_label)
    ax.fill_between(mean_fpr, np.array(mean_tpr) - np.array(stderr_tpr), np.array(mean_tpr) + np.array(stderr_tpr),
                    alpha=0.2)


def configure_axes(ax, xlabel, ylabel, ylim, identity_line, fontsize, label, is_first_column):
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if len(xlabel) == 0:
        ax.set_xticks([])
    if identity_line:
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    if ylim:
        ax.set_ylim(ylim)
    if is_first_column:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_title(label.upper(), fontsize=fontsize)


def create_subplots(nrows, ncolumns, figsize, sharey):
    fig, axs = plt.subplots(nrows, ncolumns, figsize=figsize, sharey=sharey)
    fig.patch.set_alpha(0)
    plt.subplots_adjust(wspace=0.03)
    return fig, axs


def show_plot(fig, handles_and_labels, fontsize, filename):
    handles, labels = handles_and_labels
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.00),
               ncol=len(labels), fontsize=fontsize)
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=150)
    plt.show()


def build_roc_curves(labels_results, labels, models, age_windows):
    thresholds = np.linspace(0, 1, 100)
    roc_curves = {label: {} for label in labels}
    print('**********ROC-AUCs values**********')
    for label in labels:
        print(f'{label.upper()}:')
        for model in models:
            if age_windows > 0:
                for window in range(age_windows):
                    window_data = labels_results[label][model][labels_results[label][model]['age_window'] == window]
                    mean_roc(window_data, thresholds, label, f'{model}_window_{window}', roc_curves)
            else:
                mean_roc(labels_results[label][model], thresholds, label, model, roc_curves)
    return roc_curves


def build_precision_recall_curves(labels_results, labels, models, age_windows):
    pr_curves = {label: {} for label in labels}
    common_recall = np.linspace(0, 1, 100)
    print('**********PR-AUCs values**********')
    for label in labels:
        print(f'{label.upper()}:')
        for model in models:
            model_preds = labels_results[label][model]
            if age_windows > 0:
                for window in range(age_windows):
                    window_data = model_preds[model_preds['age_window'] == window]
                    mean_pr(window_data, common_recall, label, f'{model}_window_{window}', pr_curves)
            else:
                mean_pr(model_preds, common_recall, label, model, pr_curves)
    return pr_curves


def mean_roc(data, thresholds, label, model_name, roc_curves):
    all_fpr, all_tpr, all_aucs = [], [], []
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
            all_aucs.append(auc(all_fpr[-1], all_tpr[-1]))
    mean_fpr, mean_tpr = np.mean(all_fpr, axis=0), np.mean(all_tpr, axis=0)
    stderr_tpr = np.std(all_tpr, axis=0) / np.sqrt(len(all_tpr))
    roc_curves[label][model_name] = {'mean': list(zip(mean_fpr, mean_tpr)), 'stderr': list(zip(mean_fpr, stderr_tpr)),
                                     'aucs': all_aucs}
    print(f'{model_name} {label} AUC: {np.median(all_aucs):.4f} '
          f'IQR: {np.percentile(all_aucs, 75) - np.percentile(all_aucs, 25):.4f}')


def mean_pr(data, common_recall, label, model_name, pr_curves):
    all_precision, all_recall, all_aucs = [], [], []
    for run in data.columns:
        if run.startswith('pred_'):
            precision, recall, _ = precision_recall_curve(data['label'].values, data[run].values)
            all_precision.append(precision)
            all_recall.append(recall)
            all_aucs.append(auc(recall, precision))
    interpolated_precisions = []
    for precision, recall in zip(all_precision, all_recall):
        interp_func = interp1d(recall, precision, bounds_error=False, fill_value=(0, 0))
        interp_prec = interp_func(common_recall)
        interpolated_precisions.append(interp_prec)
    interpolated_precisions = np.array(interpolated_precisions)
    mean_precision = np.mean(interpolated_precisions, axis=0)
    std_error_precision = np.std(interpolated_precisions, axis=0) / np.sqrt(len(interpolated_precisions))
    pr_curves[label][model_name] = {'mean': list(zip(common_recall, mean_precision)),
                                    'stderr': list(zip(common_recall, std_error_precision)),
                                    'aucs': all_aucs}
    print(f'{model_name} {label} AUC: {np.median(all_aucs):.4f} '
          f'IQR: {np.percentile(all_aucs, 75) - np.percentile(all_aucs, 25):.4f}')


def count_tp_fp_tn_fn(predictions, labels, threshold):
    tp, fp, tn, fn = 0, 0, 0, 0
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
    parser.add_argument('-c', '--cfgs', nargs='+', type=str, default=['age_invariant', 'age_agnostic', 'age_aware'],
                        help='configurations to plot')
    parser.add_argument('-w', '--age_windows', type=int, default=0,
                        help='Divide classifications in n equidistant age windows')
    parser.add_argument('--set', type=str, default='val', help='set to plot evaluations from (val or test)')
    args = parser.parse_args()
    results_path = Path(EVALUATION_PATH, args.dataset, args.set)

    plot(results_path, args.cfgs, args.targets, args.bars, args.age_windows)
