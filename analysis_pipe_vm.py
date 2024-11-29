import ast
import pandas as pd
from ephys_analysis import funcs_for_results_tables, funcs_plotting_raw_traces
from ephys_analysis import funcs_human_characterisation as hcf

def main():
    '''
    Main function to run the analysis pipeline.
    '''
    intr_data_path = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/results/human/data/' + \
        'summary_data_tables/intrinsic_properties/2024-07-04_collected.xlsx'
    event_type = 'spontan'
    df_merged = prep_event_data(intr_data_path, event_type)
    plot_chosen_traces(df_merged, 0, 5, 4)

    merged_with_comments = pd.read_excel('/Users/verjim/laptop_D_17.01.2022/Schmitz_lab' + \
                                         '/results/human/data/summary_data_tables/events/' + \
                                            '2024-09-16_merged_spontan_intrinsic.xlsx')
    merged_with_comments.dropna(subset=['trace_quality_eval'], inplace = True, ignore_index = True )

    okey, okey_tr, nice, nice_tr, perfect, perfect_tr, exclude = [], [], [], [], [], [], []
    for i, com in enumerate(merged_with_comments.trace_quality_eval):
        if 'okey' in com:
            okey.append(i)
            okey_tr.append(merged_with_comments['treatment'][i])
        elif 'nice' in com:
            nice.append(i)
            nice_tr.append(merged_with_comments['treatment'][i])
        elif 'perfect' in com:
            perfect.append(i)
            perfect_tr.append(merged_with_comments['treatment'][i])
        elif 'exclude' in com:
            exclude.append(i)

    return merged_with_comments.iloc[perfect, :]

# pylint: disable=line-too-long
def prep_event_data(intr_data_path, event_type):
    '''
    Merges event data with intrinsic data and returns a DataFrame.
    Parameters:
        intr_data_path (str) : Path to the intrinsic data file.
        event_type (str) : 'minis' or 'spontan'
    '''
    # collecting the event data
    df = funcs_for_results_tables.collect_events_dfs(event_type)
    df = df.rename(columns={'Cut sweeps first part (ms)': \
                            'cut_sweeps_to_datapoint'})
    df['Channels to use'] = df['Channels to use'].astype(int)

    # removing rows with empty swps_to_analyse
    df = df[(df['swps_to_analyse'] != '[]')].reset_index(drop = True)
    # dropping columns
    df['treatm_old'] = df['treatment'].values
    df['cell_ID_old'] = df['cell_ID'].values
    df.drop(columns=['treatment', 'cell_ID'], inplace = True)

    # loading and cleaning intrinsic data
    df_intrinsic_data = pd.read_excel(intr_data_path)
    drop_cols_intr = ['Unnamed: 27', 'Unnamed: 0.1', 'comments', 'Unnamed: 0.1.1','TH', \
                    'AP_heigth','rheos_ramp', ' ', 'max_repol', 'rheo_ramp', 'Unnamed: 0', \
                        'max_spikes', 'AP_halfwidth','Rheobse_ramp', 'Rheobase', 'max_depol']
    df_intrinsic = df_intrinsic_data.drop(columns = drop_cols_intr)

    cell_IDs_new_s, cell_IDs_new_intr = [], []

    for j in range(len(df)):
        fn = df['Name of recording'][j]
        slic = df['slice'][j]
        chans = [df['Channels to use'][j]]
        patcher = df['patcher'][j]
        cell_IDs_new_s.append(hcf.get_new_cell_IDs(fn, slic, chans, patcher)[0])
    df['cell_ID_new'] = cell_IDs_new_s

    for i in range(len(df_intrinsic)):
        fn = df_intrinsic['filename'][i]
        slic = df_intrinsic['slice'][i]
        chans = [df_intrinsic['cell_ch'][i]]
        patcher = df_intrinsic['patcher'][i]
        cell_IDs_new_intr.append(hcf.get_new_cell_IDs(fn, slic, chans, patcher)[0])
    df_intrinsic['cell_ID_new'] = cell_IDs_new_intr

    repeats_s = df['cell_ID_new'].value_counts()
    repeats_s = set(repeats_s[repeats_s > 1].index)
    df_no_repeats = df[~df['cell_ID_new'].isin(repeats_s)].reset_index(drop = True)

    repeats_i = df_intrinsic['cell_ID_new'].value_counts()
    repeats_i = set(repeats_i[repeats_i > 1].index)
    df_intrinsic_no_repeats = df_intrinsic[~df_intrinsic['cell_ID_new'].isin(repeats_i)]

    df_merged_all = df_no_repeats.merge(df_intrinsic_no_repeats, left_on = ['cell_ID_new'],right_on  = ['cell_ID_new'], validate = '1:1')

    # df_merged = df.merge(df_intrinsic, left_on = ['cell_ID_old'],right_on  = ['cell_ID'], validate = '1:1')
    # df_merged = df.merge(df_intrinsic, left_on = ['OP', 'slice', 'Channels to use','patcher'],\
    #  #                           right_on  = ['OP', 'slice', 'cell_ch', 'patcher'], validate = '1:1')
    # differences_tr = df_merged[df_merged['treatment'] != df_merged['treatm_old']]
    # differences_cell_id = df_merged[df_merged['cell_ID'] != df_merged['cell_ID_old']]

    # print('Fix treatment for OPs', differences_tr['OP'])
    # print('Fix cell_ID for OPs', differences_cell_id['OP'])

    return df_merged_all

def plot_chosen_traces(df_merged, indx_start, indx_stop, step, human_dir = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/'):
    '''
    Plots selected traces from a merged DataFrame based on specified parameters.
    Parameters:
        df_merged (DataFrame): A pandas DataFrame containing merged data with columns:
            - 'patcher': Identifier for the patcher.
            - 'OP': surgery name
            - 'Name of recording': Name of the recording file.
            - 'cell_ch': Channel number for the recording.
            - 'swps_to_analyse': Sweeps to analyze, stored as a string representation of a list.
            - 'cut_sweeps_to_datapoint': The point to cut the sweeps for plotting.
        indx_start (int): The starting index for plotting.
        indx_stop (int): The stopping index for plotting.
        step (int): The step size for plotting.
        human_dir (str): Base directory path for human data recordings. Default is set to 
                         '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/'.
    Returns:
        None: The function displays plots and waits for user input to continue.
    '''
    patcher_dict = {'Verji':'data_verji/', 'Rosie':'data_rosie/'}
    for i in range(indx_start, indx_stop, step):
        if i == 183:
            continue
        patcher = df_merged.patcher[i]
        op = df_merged.OP[i] + '/'

        fn = human_dir + patcher_dict[patcher] + op + df_merged['Name of recording'][i]
        chan = df_merged.cell_ch.values[i]
        swps_keep = ast.literal_eval(df_merged.swps_to_analyse.values[i])  # type: ignore #eval()
        first_point_keep = df_merged['cut_sweeps_to_datapoint'][i]

        funcs_plotting_raw_traces.plt_trace_select_swps_cut_parts(fn = fn, chan = chan, swps_keep = swps_keep, \
                                                                start_point_cut = 0, end_point_cut = first_point_keep)
        input('Press Enter to continue...')

if __name__ == "__main__":
    main()
