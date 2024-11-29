
import sys
sys.path.append('./core/')
from miniML import MiniTrace, EventDetection
import tensorflow as tf
print(tf.__version__)

filename = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/data_verji/OP230808/23808033.abf' #'./example_data/gc_mini_trace.h5'
chan = 4
swps_delete = list(range(1, 29))
scaling = 1
unit = 'pA'
win_size = 750
stride = int(win_size/30)

trace = MiniTrace.from_axon_file_MP(filepath = filename,
                                    channel = chan,
                                    scaling = scaling,
                                    unit = unit,
                                    sweeps_delete = swps_delete,
                                    first_point = 0,
                                    last_point = 5500)

trace.plot_trace()

detection = EventDetection(data = trace,
                        model_path='./models/transfer_learning/human_pyramids_L2_3/2024_Oct_29_lstm_transfer.h5',
                        window_size = win_size,
                        model_threshold = 0.5,
                        batch_size = 512,
                        event_direction = 'negative',
                        training_direction = 'negative',
                        compile_model = True)

detection.detect_events(stride=stride,
                        eval = True, #
                        convolve_win = 20, #the Hann wondow for filtering. presserving the peaks?
                        resample_to_600 = True) #always true for transfer learning
