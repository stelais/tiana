from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import file_organizers.file_reading as fr
import threshold_metrics as tm


def confusion_matrix_plotter(true_labels, predictions,
                             inference_path, type_, test_split,
                             threshold_value,
                             should_normalize_=None,
                             labels_=None,
                             show_plot=False, small_size=False):
    if labels_ is None:
        labels_ = ['Not \n Microlensing', 'Microlensing']

    disp = ConfusionMatrixDisplay.from_predictions(true_labels,
                                                   predictions,
                                                   display_labels=labels_,
                                                   cmap=plt.cm.Blues,
                                                   normalize=should_normalize_)

    plt.title(f'Confusion Matrix - Threshold: {threshold_value} | {type_} {test_split}')

    if small_size:
        plt.rcParams['figure.figsize'] = [4, 4]
        plt.tight_layout()
        if should_normalize_ == 'true':
            plt.savefig(f'{inference_path}/{type_}_inference_plots/normalized_confusion_matrix_'
                        f'{threshold_value}_ts{test_split}_small.png', dpi=300)
        else:
            plt.savefig(f'{inference_path}/{type_}_inference_plots/confusion_matrix_'
                        f'{threshold_value}_ts{test_split}_small.png', dpi=300)
    else:
        plt.tight_layout()
        if should_normalize_ == 'true':
            plt.savefig(f'{inference_path}/{type_}_inference_plots/normalized_confusion_matrix_'
                        f'{threshold_value}_ts{test_split}.png', dpi=300)
        else:
            plt.savefig(f'{inference_path}/{type_}_inference_plots/confusion_matrix_'
                        f'{threshold_value}_ts{test_split}.png', dpi=300)
    if show_plot:
        plt.show()
    plt.close()


def inference_cumulative_distribution(inference_with_threshold, show_plot=False):
    """
    This function plots the cumulative distribution of the inference
    :param inference_with_threshold:
    :return:
    """
    microlensing_df = inference_with_threshold[inference_with_threshold['true_label'] == 1]
    non_microlensing_df = inference_with_threshold[inference_with_threshold['true_label'] == 0]

    # Scores in ascending order:
    microlensing_scores = microlensing_df.sort_values('Score')['Score']
    non_microlensing_scores = non_microlensing_df.sort_values('Score')['Score']

    # Inserting 0 and 1s
    microlensing_scores = np.insert(microlensing_scores, 0, 0)
    microlensing_scores = np.insert(microlensing_scores, len(microlensing_scores), 1)
    non_microlensing_scores = np.insert(non_microlensing_scores, 0, 0)
    non_microlensing_scores = np.insert(non_microlensing_scores, len(non_microlensing_scores), 1)

    # Defining CDF values
    step_microlensing = 1/ (len(microlensing_scores) - 1)
    microlensing_cdf = np.arange(0, 1+step_microlensing/2, step_microlensing)
    step_non_microlensing = 1/(len(non_microlensing_scores) - 1)
    non_microlensing_cdf = np.arange(0, 1+step_non_microlensing/2, step_non_microlensing)

    fig, ax = plt.subplots()
    ax.step(microlensing_scores, microlensing_cdf, label=f'Microlensing: {len(microlensing_scores) - 2}')
    ax.step(non_microlensing_scores, non_microlensing_cdf, label=f'Not Microlensing: {len(non_microlensing_scores) - 2}', color='orange')
    ax.set(xlabel='Neural Network Confidence', ylabel='Cumulative Distribution', title=f'Cumulative Distribution')
    ax.legend()
    if show_plot:
        plt.show()
    plt.close()



if __name__ == '__main__':
    type_ = '550k'
    test_split = 0
    threshold_value = 0.5

    inference_folder = '/Users/sishitan/Documents/Scripts/qusi_project/qusi/inferences/'
    inference_file = f'results_ts{test_split}_{type_}_with_tags.csv'

    inference_df = fr.read_inference_with_tags_and_labels(inference_folder + inference_file)
    inference_with_threshold = tm.threshold_prediction_setter(inference_df, threshold_value)
    true_labels = inference_with_threshold['true_label']
    predictions = inference_with_threshold['prediction']

    confusion_matrix_plotter(true_labels, predictions, inference_folder, type_, test_split, threshold_value)
    confusion_matrix_plotter(true_labels, predictions, inference_folder, type_, test_split, threshold_value,
                             should_normalize_='true')

    true_positives, false_positives, true_negatives, false_negatives = tm.performance_calculator(true_labels, predictions)
    print('True Positives: ', true_positives)
    print('False Positives: ', false_positives)
    print('True Negatives: ', true_negatives)
    print('False Negatives: ', false_negatives)

    inference_cumulative_distribution(inference_with_threshold, show_plot=True)
