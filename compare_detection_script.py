import sys
sys.path.append('../miniML_multipatch/core/')

from miniML import MiniTrace, EventDetection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import ast
from datetime import datetime

file_index = 13

def create_data (file_index, swp_number = 'all', 
                 output_folder = '/alzheimer/verjinia/data/out_model_comparison/'):
    '''
    Function to create data for the comparison of the two models
    file_index: int, index of the file in the metadata table with perfect traces
    swp_number: int, number of the sweeps to be analysed. If all, all the cleam sweeps will be analysed    

    '''
    # server paths
    default_model = './models/GC_lstm_model.h5'
    trained_model = '/alzheimer/verjinia/miniML_multipatch/model_training/out_model_traininig/lstm_transfer_oct_2024.h5'
    perfect_traces_df = pd.read_excel('/alzheimer/verjinia/data/metadata_tables/perfect_traces.xlsx')
    data_path = '/alzheimer/verjinia/data/recordings/'
    filename  = data_path + perfect_traces_df['Name of recording'][file_index]

#     # local paths
#     default_model = './models/GC_lstm_model.h5'
#     trained_model = './models/transfer_learning/human_pyramids_L2_3/lstm_transfer_oct_2024.h5'
#     perfect_traces_df = pd.read_excel('/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/results/human/data/summary_data_tables/events/perfect_traces.xlsx')
#     patcher_dict = {'Verji':'data_verji/', 'Rosie':'data_rosie/'}
#     data_path = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/'
#     filename  = data_path + patcher_dict[perfect_traces_df.patcher[file_index]] + perfect_traces_df['OP'][file_index] + '/' + perfect_traces_df['Name of recording'][file_index]

    chan = perfect_traces_df.cell_ch.values[file_index]
    swps_keep = ast.literal_eval(perfect_traces_df.swps_to_analyse.values[file_index])

    if isinstance(swp_number, int) and swp_number <= len(swps_keep):
        swps_keep = ast.literal_eval(perfect_traces_df.swps_to_analyse.values[file_index])[:swp_number]

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


    default_detection = EventDetection(data = trace,
                            model_path = default_model,
                            window_size = win_size,
                            model_threshold = threshold,
                            batch_size = 512,
                            event_direction = event_direction,
                            training_direction = training_direction,
                            compile_model = True)

    default_detection.detect_events(eval = True, convolve_win = 20, resample_to_600 = True)

    trained_detection = EventDetection(data = trace,
                           model_path = trained_model,
                           window_size = win_size,
                           model_threshold = threshold,
                           batch_size = 512,
                           event_direction = event_direction,
                           training_direction = training_direction,
                           compile_model = True)

    trained_detection.detect_events(eval = True, convolve_win = 20, resample_to_600 = True)

    prediction_x = np.arange(0, len(default_detection.prediction)) * default_detection.stride_length * trace.sampling

    date_pre = datetime.now().strftime("%Y_%b_%d")
    with h5py.File(output_folder + date_pre + '_default_trained.h5', 'w') as f:
        f.create_dataset('prediction_x', data = prediction_x)

        f.create_dataset('prediction_default', data = default_detection.prediction)
        f.create_dataset('event_peaks_default', data = default_detection.event_peak_times)
        f.create_dataset('peaks_locs_default', data = trace.data[default_detection.event_peak_locations])

        f.create_dataset('prediction_trained', data = trained_detection.prediction)
        f.create_dataset('event_peaks_trained', data = trained_detection.event_peak_times)
        f.create_dataset('peaks_locs_trained', data = trace.data[trained_detection.event_peak_locations])

        f.create_dataset('trace_time', data = trace.time_axis)
        f.create_dataset('trace_data', data = trace.data)

print('Data for the comparison of the two models saved in ' + output_folder)


def plot_model_comparison(outout_folder):
    '''
    Function to plot the latest comparison of the two models
    '''

    with h5py.File('/Users/verjim/miniML_multipatch/out_model_training/detection_results.h5', 'r') as f:
        prediction_x = f['prediction_x'][:]

        prediction_old_model = f['prediction_default'][:]
        event_peaks_old = f['event_peaks_default'][:]
        peaks_locs_old = f['peaks_locs_default'][:]

        prediction_new_model = f['prediction_trained'][:]
        event_peaks_new = f['event_peaks_trained'][:]
        peeaks_new = f['peaks_locs_trained'][:]

        trace_time = f['trace_time'][:]
        trace_data = f[' trace_data'][:]

fig, axs = plt.subplots(3, 1, sharex = False)

axs[0].plot(prediction_x, prediction_old_model, c='k', alpha=0.7, label='old')
axs[0].plot(prediction_x, prediction_new_model, c='b', alpha=0.7, label='new')
axs[0].legend()

axs[1].plot(trace_time, trace_data, c='k')
axs[1].scatter(event_peaks_old, peaks_locs_old, c='orange', zorder=2)

axs[2].plot(trace_time, trace_data, c='b')
axs[2].scatter(event_peaks_new, peeaks_new, c='orange', zorder=2)
plt.show()

if server:
    plt.savefig('/alzheimer/verjinia/data/out_model_comparison/compare_detection.png')
    plt.close()