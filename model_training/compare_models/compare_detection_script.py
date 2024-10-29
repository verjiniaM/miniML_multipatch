import ast # type: ignore
import os
import sys
from datetime import datetime
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#sys.path.append('../miniML_multipatch/core/')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../core/'))
from miniML import MiniTrace, EventDetection # type: ignore
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

def main():
    '''
    Function to run the comparison of the two models
    '''
    if sys.argv[1] == 'create':
        create_data(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == 'plot':
        plot_model_comparison()

def create_data (file_index, swp_number = 'all',
                 output_folder = '/alzheimer/verjinia/miniML_multipatch/model_training/compare_models/data/'):
    '''
    Function to create data for the comparison of the two models
    file_index: int, index of the file in the metadata table with perfect traces
    swp_number: int, number of the sweeps to be analysed. If all, all the cleam sweeps will be analysed
    '''
    # server paths
    models_path = '/alzheimer/verjinia/miniML_multipatch/models/'
    default_model =  models_path + 'GC_lstm_model.h5'
    trained_model = models_path + 'transfer_learning/human_pyramids_L2_3/2024_Oct_29_lstm_transfer.h5'

    perfect_traces_df = pd.read_excel('/alzheimer/verjinia/data/metadata_tables/perfect_traces.xlsx')
    data_path = '/alzheimer/verjinia/data/recordings/'
    filename  = data_path + perfect_traces_df['Name of recording'][file_index]

#     # local paths
#     perfect_traces_df = pd.read_excel('/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/results/human/data/summary_data_tables/events/perfect_traces.xlsx')
#     patcher_dict = {'Verji':'data_verji/', 'Rosie':'data_rosie/'}
#     data_path = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/'
#     filename  = data_path + patcher_dict[perfect_traces_df.patcher[file_index]] + \
# perfect_traces_df['OP'][file_index] + '/' + perfect_traces_df['Name of recording'][file_index]

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
    date_prefix = datetime.now().strftime("%Y_%b_%d")
    with h5py.File(output_folder + date_prefix + '_default_trained.h5', 'w') as f:
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

def plot_model_comparison(latest = True,
                          plots_folder = './model_training/compare_models/plots/',
                          data_folder = './model_training/compare_models/data/'):
    '''
    Function to plot the latest comparison of the two models
    '''
    if latest:
        files = os.listdir(data_folder)
        files.sort()
        data_file = files[-1]
    else:
        data_file = input('Enter the name of the data file to be plotted: ')

    with h5py.File(data_folder + data_file, 'r') as f:
        prediction_x = f['prediction_x'][:]

        prediction_default = f['prediction_default'][:]
        event_peaks_default = f['event_peaks_default'][:]
        peaks_locs_default = f['peaks_locs_default'][:]

        prediction_trained = f['prediction_trained'][:]
        event_peaks_trained = f['event_peaks_trained'][:]
        peaks_locs_trained = f['peaks_locs_trained'][:]

        trace_time = f['trace_time'][:]
        trace_data = f[' trace_data'][:]

    _, axs = plt.subplots(3, 1, sharex = False)

    axs[0].plot(prediction_x, prediction_default, c='k', alpha=0.7, label='old')
    axs[0].plot(prediction_x, prediction_trained, c='b', alpha=0.7, label='new')
    axs[0].legend()

    axs[1].plot(trace_time, trace_data, c='k')
    axs[1].scatter(event_peaks_default, peaks_locs_default, c='orange', zorder=2)

    axs[2].plot(trace_time, trace_data, c='b')
    axs[2].scatter(event_peaks_trained, peaks_locs_trained, c='orange', zorder=2)
    plt.show()
    plt.savefig(plots_folder + data_file[:8] + '_comparison.png')

if __name__ == "__main__":
    main()
