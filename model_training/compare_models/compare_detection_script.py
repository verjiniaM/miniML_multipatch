import ast # type: ignore
import os
import shutil
import random
import sys
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('../miniML_multipatch/core/')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../core/'))
from miniML import MiniTrace, EventDetection # type: ignore
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

def main():
    '''
    Function to run the comparison of the two models
    '''
    if sys.argv[1] == 'create':
        create_data(int(sys.argv[2]), int(sys.argv[3])) # file_index, swp_number
        # copy_folder_contents('/alzheimer/verjinia/data/model_comparison/data/', \
        #                      '/Users/verjim/miniML_data/data/')
        print('Data copied to the local folder: ' + '/Users/verjim/miniML_data/data/')
    elif sys.argv[1] == 'plot':
        plot_model_comparison(False)

def create_data (file_index, swp_number = 'all',
                 output_folder = '/alzheimer/verjinia/data/model_comparison/data/'):
    '''
    Function to create data for the comparison of the two models
    file_index: int, index of the file in the metadata table with perfect traces
    '''
    # server paths
    models_path = '/alzheimer/verjinia/miniML_multipatch/models/transfer_learning/human_pyramids_L2_3/'
    model1 =  models_path + '2024_Oct_29_lstm_transfer.h5'
    model2 = models_path + '2024_lstm_transfer_dec_2024.h5'

    events_df = pd.read_excel('/alzheimer/verjinia/data/metadata_tables/2024-09-16_merged_spontan_intrinsic_copy_used_for_model_evl.xlsx')
    data_path = '/alzheimer/verjinia/data/recordings/'
    filename  = data_path + events_df['Name of recording'][file_index]

    chan = events_df.cell_ch.values[file_index]
    swps_keep = ast.literal_eval(events_df.swps_to_analyse.values[file_index])

    if isinstance(swp_number, int) and swp_number <= len(swps_keep):
        swps_keep = ast.literal_eval(events_df.swps_to_analyse.values[file_index])[:swp_number]
    swps_delete = list(set(range(30)) - set(swps_keep))

    scaling = 1
    unit = 'pA'
    win_size = 3000
    threshold = 0.5
    event_direction = 'negative'
    training_direction = 'negative'

    trace = MiniTrace.from_axon_file_MP(filepath = filename,
                                        channel = chan,
                                        scaling = scaling,
                                        unit = unit,
                                        sweeps_delete = swps_delete,
                                        first_point = 0,
                                        last_point = 5500)

    detection1 = EventDetection(data = trace,
                            model_path = model1,
                            window_size = win_size,
                            model_threshold = threshold,
                            batch_size = 512,
                            event_direction = event_direction,
                            training_direction = training_direction,
                            compile_model = True)

    detection1.detect_events(eval = True, convolve_win = 20, resample_to_600 = True)

    detection2 = EventDetection(data = trace,
                           model_path = model2,
                           window_size = win_size,
                           model_threshold = threshold,
                           batch_size = 512,
                           event_direction = event_direction,
                           training_direction = training_direction,
                           compile_model = True)

    detection2.detect_events(eval=True, convolve_win=20, resample_to_600=True)

    date_prefix = datetime.now().strftime("%Y_%b_%d")
    with h5py.File(output_folder + date_prefix + events_df['Name of recording'][file_index] + \
                   str(chan) + '_model_comparison.h5', 'w') as f:

        model_1 = f.create_group('model_1')
        model_1.create_dataset('prediction1', data = detection1.prediction)
        model_1.create_dataset('event_peaks1', data = detection1.event_peak_times)
        model_1.create_dataset('peak_locs1', data = trace.data[detection1.event_peak_locations])

        model_2 = f.create_group('model_2')
        model_2.create_dataset('prediction2', data = detection2.prediction)
        model_2.create_dataset('event_peaks2', data = detection2.event_peak_times)
        model_2.create_dataset('peak_locs2', data = trace.data[detection2.event_peak_locations])

        trace_ = f.create_group('trace_')
        trace_.create_dataset('trace_time', data = trace.time_axis)
        trace_.create_dataset('trace_data', data = trace.data)

        model_1.attrs['model_1_name'] = model1[model1.rfind('/') + 1 :-3]
        model_2.attrs['model_2_name'] = model2[model2.rfind('/') + 1 :-3]
        trace_.attrs['filename'] = events_df['Name of recording'][file_index]
        trace_.attrs['chan'] = str(chan)

    print(f"Data for the comparison of {model1[models_path.rfind('/') + 1:]} and " + \
          f"{model2[models_path.rfind('/') + 1:]} data saved in {output_folder}")

def plot_model_comparison(latest = True,
                          plots_folder = '/Users/verjim/miniML_data/plots/',
                          data_folder = '/Users/verjim/miniML_data/data/'):
    '''
    Function to plot the latest comparison of the two models
    '''
    files = os.listdir(data_folder)
    files = [os.path.join(data_folder, f) for f in files]  # Get full paths
    files.sort(key=os.path.getctime)
    if latest:
        data_file = files[-1]
    else:
        for file_ in files:
            print(file_[file_.rfind('/')+1:])
        data_file = data_folder + input('Enter the name of the data file to be plotted: ')

    with h5py.File(data_file, 'r') as f:
        # Access model_1 group and its datasets
        model_1 = f['model_1']
        prediction_1 = model_1['prediction1'][:]
        event_peaks_1 = model_1['event_peaks1'][:]
        peak_locs_1 = model_1['peak_locs1'][:]
        model_1_name = model_1.attrs['model_1_name']

        # Access model_2 group and its datasets
        model_2 = f['model_2']
        prediction_2 = model_2['prediction2'][:]
        event_peaks_2 = model_2['event_peaks2'][:]
        peak_locs_2 = model_2['peak_locs2'][:]
        model_2_name = model_2.attrs['model_2_name']

        # Access trace group and its datasets
        trace_group = f['trace_']
        trace_time = trace_group['trace_time'][:]
        trace_data = trace_group['trace_data'][:]
        fn = trace_group.attrs['filename']
        chan = trace_group.attrs['chan']

    _, axs = plt.subplots(3, 1, sharex = True)

    axs[0].plot(trace_time[:len(prediction_1)], prediction_1, c='k', alpha=0.7, label= model_1_name)
    axs[0].plot(trace_time[:len(prediction_2)], prediction_2, c='b', alpha=0.7, label= model_2_name)
    axs[0].set_title('Predictions')
    axs[0].legend()

    axs[1].plot(trace_time, trace_data, c='k')
    axs[1].scatter(event_peaks_1, peak_locs_1, c='orange', zorder=2)
    axs[1].set_title(model_1_name)

    axs[2].plot(trace_time, trace_data, c='b')
    axs[2].scatter(event_peaks_2, peak_locs_2, c='orange', zorder=2)
    axs[2].set_title(model_2_name)
    _.suptitle(fn +' chan ' + chan)
    plt.savefig(plots_folder + fn[:-4] + '_' + chan + '_comparison.png')
    plt.show()

def print_h5_structure(file_path):
    """
    Function to print the structure of an HDF5 file.
    Args:
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")

        f.visititems(print_structure)

def create_eval_data(eval_purpose, num_files):
    '''
    Function to create data for the evaluation of the two models

    eval_purpose: str; clarification wht were those files used to, evaluation or training
    num_files: int; number of files to be used
    '''
    df_path = '/alzheimer/verjinia/data/metadata_tables/2024-0 9-16_merged_spontan_intrinsic_copy_used_for_model_evl.xlsx'
    events_df = pd.read_excel(df_path)

    # Filter indices where 'use' is NaN
    nan_indices = events_df[events_df['use'].isna()].index
    random_file_indx = random.sample(list(nan_indices), num_files)
    # Update the 'use' column at the selected random indices
    events_df.loc[random_file_indx, 'use'] = eval_purpose
    events_df.to_excel(df_path, index=False)

    for file in random_file_indx:
        swps_keep = ast.literal_eval(events_df['swps_to_analyse'].iloc[file]) 
        random_swp = random.choice(swps_keep)
        create_data(file, swp_number = random_swp)

    # copy_folder_contents('/alzheimer/verjinia/data/model_comparison/data/', \
    #                     '/Users/verjim/miniML_data/data/')

def copy_folder_contents(src_dir, dst_dir):
    """
    Function to copy all contents from one folder to another, 
    only if they do not already exist in the destination folder.
    
    Args:
        src_dir (str): Path to the source directory.
        dst_dir (str): Path to the destination directory.
    """
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Copy all contents from the source directory to the destination directory
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        # Check if the item already exists in the destination folder
        if not os.path.exists(dst_path):
            shutil.copytree(src_path, dst_path)

if __name__ == "__main__":
    main()