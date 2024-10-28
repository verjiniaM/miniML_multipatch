import sys
sys.path.append('../miniML_multipatch/core/')

from miniML import MiniTrace, EventDetection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# local
model_1 = './models/transfer_learning/models/GC_lstm_model.h5'
model_2 = './models/transfer_learning/human_pyramids_L2_3/lstm_transfer_oct_2024.h5'

# server
# model_1 = './models/GC_lstm_model.h5'
# model_2 = './models/human_pyramids_L2_3/lstm_transfer_oct_2024.h5'

perfect_traces_df = pd.read_excel('/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/results/human/data/summary_data_tables/events/perfect_traces.xlsx')
patcher_dict = {'Verji':'data_verji/', 'Rosie':'data_rosie/'}

path = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/'
scaling = 1 
unit = 'pA'

# training dataset includes traces until a = 12
a = 13

filename  = path + patcher_dict[perfect_traces_df.patcher[a]] + perfect_traces_df['OP'][a] + '/' + perfect_traces_df['Name of recording'][a]
chan = perfect_traces_df.cell_ch.values[a]

# keeping only the first sweep for tesing purposes
#swps_keep = ast.literal_eval(perfect_traces_df.swps_to_analyse.values[a])
swps_keep = [0]
swps_delete = list(set(range(30)) - set(swps_keep))

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


# detection_1 = EventDetection(data = trace,
#                            model_path = model_1,
#                            window_size = win_size,
#                            model_threshold = threshold,
#                            batch_size = 512,
#                            event_direction = event_direction,
#                            training_direction = training_direction,
#                            compile_model = True)

# detection_1.detect_events(eval = True, convolve_win = 20, resample_to_600 = True)

# detection_2 = EventDetection(data = trace,
#                            model_path = model_2,
#                            window_size = win_size,
#                            model_threshold = threshold,
#                            batch_size = 512,
#                            event_direction = event_direction,
#                            training_direction = training_direction,
#                            compile_model = True)

# detection_2.detect_events(eval = True, convolve_win = 20, resample_to_600 = True)

# fig, axs = plt.subplots(3, 1, sharex=True)

# prediction_x = np.arange(0, len(detection_1.prediction)) * detection_1.stride_length * trace.sampling


# axs[0].plot(prediction_x, detection_1.prediction, c='k', alpha=0.7, label=f'{model_1}')
# axs[0].plot(prediction_x, detection_2.prediction, c='b', alpha=0.7, label=f'{model_1}')
# axs[0].legend()

# axs[1].plot(trace.time_axis, trace.data, c='k')
# axs[1].scatter(detection_1.event_peak_times, trace.data[detection_1.event_peak_locations], c='orange', zorder=2)

# axs[2].plot(trace.time_axis, trace.data, c='b')
# axs[2].scatter(detection_2.event_peak_times, trace.data[detection_2.event_peak_locations], c='orange', zorder=2)
# plt.show()
 