import sys
sys.path.append('./core/')
from miniML import MiniTrace, EventDetection
import matplolist.pyplot as plt

filename = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/data_verji/OP230808/23808033.abf' #'./example_data/gc_mini_trace.h5'
unit = 'pA'

# get from h5 file
trace = MiniTrace.from_axon_file(filepath=filename,
                                 channel = 2,
                                 unit=unit,
                                 first_point = 0, 
                                 last_point = 5000,
                                 sweeps_delete = [range(25)])


trace.plot_trace()
plt.show()

win_size = 900
stride = int(win_size/30)
direction = 'negative'

detection = EventDetection(data = trace,
                           model_path = '/Users/verjim/miniML_multipatch/model_training/kaggle_output_model_training_dataset1/lstm_transfer_dataset1.h5',
                           window_size = win_size,
                           model_threshold = 0.5,
                           batch_size = 512,
                           event_direction = direction,
                           training_direction = 'negative',
                           compile_model = True)

print('stwarting detection')
detection.detect_events(stride=stride,
                        eval=True,
                        verbose=True,
                        convolve_win = 15,
                        resample_to_600 = True)

print('plotting')
detection.plot_prediction(include_data=True, plot_filtered_prediction=True, plot_filtered_trace=True)
detection.plot_events()
detection.plot_event_histogram(plot='amplitude', cumulative=False)

print('saving the results')
detection.save_to_pickle(filename='./example_data/results/gc_mini_trace_results.pickle', 
                         include_prediction=True, 
                         include_data=False)

detection.save_to_csv(path='./example_data/results', overwrite=True)