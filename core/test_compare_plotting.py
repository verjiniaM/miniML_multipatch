import matplotlib.pyplot as plt
from miniML import MiniTrace
import manual_results_correction_imbroscb as funcs_man_eval
from scipy import signal

def hann_filter(data, filter_size):
    win = signal.windows.hann(filter_size)
    return signal.convolve(data, win, mode = 'same') / sum(win)

fn_path = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/data_spontan/recordings/24d12007.abf'
chan = 5
swps_delete = list(range(5,29))
analysis_end = 1_200_000
convolve_win = 20

trace = MiniTrace.from_axon_file_MP(filepath = fn_path,
                                    channel = chan,
                                    scaling = 1,
                                    unit = 'mV',
                                    sweeps_delete = swps_delete,
                                    first_point = 0,
                                    last_point = 5500)


signal_no_filter = trace.data[:analysis_end,]
signal_lp = funcs_man_eval.lowpass(signal_no_filter, 800, order=1)
signal_hann = hann_filter(signal_no_filter, filter_size = convolve_win)

fig, ax = plt.subplots(3,1, figsize = (15,10), sharex = True)
ax[0].plot(signal_no_filter, color = 'black')
ax[0].set_title('No filter')
ax[1].plot(signal_lp, color = 'blue')
ax[1].set_title('Lowpass filter 800Hz')
ax[2].plot(signal_hann, color = 'red')
ax[2].set_title('Hanning filter 20ms')

plt.show()