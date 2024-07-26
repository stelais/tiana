import file_reading as fr
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


def tag_finder(moa_intern_name_, sumi_dataframe_):
    """
    This function finds the Sumi tag of a given lightcurve name
    :param moa_intern_name_:
    :param sumi_dataframe_:
    :return:
    """
    what_tag = sumi_dataframe_[sumi_dataframe_['lightcurve_name'] == moa_intern_name_]['sumi_tag'].values
    return what_tag[0]


def true_label_definer(what_tag_):
    """
    This function defines the true label of a given Sumi tag
    :param what_tag_:
    :return:
    """
    if what_tag_ == 'c' or what_tag_ == 'cf' or what_tag_ == 'cp' or what_tag_ == 'cw' or what_tag_ == 'cs' or what_tag_ == 'cb':
        return 1
    elif what_tag_ == 'v' or what_tag_ == 'n' or what_tag_ == 'nr' or what_tag_ == 'm' or what_tag_ == 'j':
        return 0
    else:
        return 0


def light_curve_name_to_true_label(lightcurve_name_, sumi_dataframe_):
    """
    This function returns the true label of a given lightcurve name
    :param lightcurve_name_:
    :param sumi_dataframe_:
    :return:
    """
    tag = tag_finder(lightcurve_name_, sumi_dataframe_)
    true_label = true_label_definer(tag)
    return true_label


def pure_inference_to_inference_with_tags_and_labels(pure_inference_dataframe, sumi_dataframe):
    """
    This function return the pure inference dataframe with the Sumi tag and the true label
    :param pure_inference_dataframe:
    :param sumi_dataframe:
    :return:
    """
    merged_df = pd.merge(pure_inference_dataframe, sumi_dataframe[['lightcurve_name', 'sumi_tag']],
                         on='lightcurve_name', how='left')
    merged_df['true_label'] = merged_df['sumi_tag'].progress_apply(true_label_definer)
    return merged_df


def combine_all_inferences(inference_folder):
    """
    This function combines all the inferences into one dataframe
    :param inference_folder:
    :return:
    """
    for index in range(0, 10):
        inference_name = f'results_550k_{index}'
        complete_inference_path = inference_folder + inference_name + '_with_tags' + '.csv'
        inference_with_tags_and_labels_df = fr.read_inference_with_tags_and_labels(complete_inference_path)
        if index == 0:
            combined_inference_df = inference_with_tags_and_labels_df
        else:
            combined_inference_df = pd.concat([combined_inference_df, inference_with_tags_and_labels_df])
    # Reset the index of the DataFrame
    combined_inference_df.reset_index(drop=True, inplace=True)
    return combined_inference_df


if __name__ == '__main__':
    inference_folder = '/Users/sishitan/Documents/Scripts/qusi_project/qusi/inferences/'
    for index in range(0, 10):
        inference_name = f'results_550k_{index}'
        inference_path = inference_folder + inference_name + '.csv'
        complete_inference_path = inference_folder + inference_name + '_with_tags' + '.csv'

        pure_inference_dataframe = fr.read_pure_inference_output(inference_path)
        sumi_dataframe = fr.read_sumi_nine_year_label()
        inference_with_tags_and_labels_df = pure_inference_to_inference_with_tags_and_labels(pure_inference_dataframe,
                                                                                             sumi_dataframe)
        inference_with_tags_and_labels_df.to_csv(complete_inference_path)

    combined_inference_df = combine_all_inferences(inference_folder)
    combined_inference_df.to_csv(inference_folder + 'results_550k_combined_with_tags.csv')

# moa_intern_name = 'gb5-R-9-3-41366'
# sumi_dataframe = fr.read_sumi_nine_year_label()
# tag = tag_finder(moa_intern_name, sumi_dataframe)
# print(tag)
# true_tag = true_label_definer(tag)
# print(true_tag)
