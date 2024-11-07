import ast # type: ignore
import os
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
        create_data(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == 'plot':
        plot_model_comparison()

def create_data (file_index, swp_number = 'all',
                 output_folder = '/alzheimer/verjinia/miniML_multipatch/model_training/compare_models/data/'):
    '''
    Function to create data for the comparison of the two models
    file_index: int, index of the file in the metadata table with perfect traces
    '''
    # server paths
    models_path = '/alzheimer/verjinia/miniML_multipatch/models/'
    model1 =  models_path + 'GC_lstm_model.h5'
    model2 = models_path + 'transfer_learning/human_pyramids_L2_3/2024_Oct_29_lstm_transfer.h5'

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

    detection2.detect_events(eval = True, convolve_win = 20, resample_to_600 = True)
    # prediction_x = np.arange(0, len(detection1.prediction)) * detection1.stride_length * trace.sampling
    date_prefix = datetime.now().strftime("%Y_%b_%d")
    with h5py.File(output_folder + date_prefix + '_model_comparison.h5', 'w') as f:

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
        trace_.attrs['filename'] = perfect_traces_df['Name of recording'][file_index]
        trace_.attrs['chan'] = str(chan)

    print('Data for the comparison of the two models saved in ' + output_folder)

def plot_model_comparison(latest = False,
                          plots_folder = '/Users/verjim/miniML_multipatch/model_training/compare_models/plots/',
                          data_folder = '/Users/verjim/miniML_multipatch/model_training/compare_models/data/'):
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
    
    _.suptitle(fn + chan)
    plt.show()
    plt.savefig(plots_folder + data_file[:8] + '_comparison.png')

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

if __name__ == "__main__":
    main()
