import ast # type: ignore
import numpy as np
import sys
import os
import pandas as pd
from datetime import date
sys.path.append('../miniML_multipatch/core/')
sys.path.append(os.path.join(os.path.dirname(__file__), '../../core/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

#
# ONLY when running interactively
# import importlib
# sys.path.append('/Users/verjim/miniML_multipatch/core/')
# sys.path.append('/Users/verjim/miniML_multipatch/model_training/core/')

# keep this
from miniML import MiniTrace # type: ignore
import manual_results_correction_imbroscb as funcs_man_eval # type: ignore

# defining paths
human_dir = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/'
dest_dir_files = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/data_spontan/recordings/'
results_dir = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/results/human/paper_figs_collected_checked/events/'

meta_df = pd.read_excel(results_dir + 'spontan_slice_all_manual_review.xlsx')

# save the current version with todays date. whenever it's analyzed, there will be an older version with the date of analysis
today = date.today().strftime("%y_%m_%d_")
if os.path.exists(results_dir + 'sum_results.xlsx'):
    sum_df = pd.read_excel(results_dir + 'sum_results.xlsx')
    sum_df.to_excel(results_dir + 'archive/' + today +'sum_results.xlsx')
else:
    sum_df = pd.DataFrame(columns = ['filename', 'channel', 'sampl_rate_Hz',
                                     'analysis_start_swp_num', 'num_analyzed_swsp',
                                     'cut_swp_last_part_dp', 'analysis_len_sec',
                                     'avg_IEI_ms', 'avg_amp_pA', 'avg_abs_amp_pA',
                                     'avg_rise_times_10_90_ms_barb', 'avg_decay_times_10_90_ms_barb',
                                     'raw_mean_bl_signal_pA', 'hann_mean_bl_signal_pA',
                                     'raw_stdv_bl_signal_pA', 'hann_stdv_bl_signal_pA',
                                     'man_revised', 'time_analysis_end'])
    sum_df.to_excel(results_dir + 'sum_results.xlsx', sheet_name='Summary results')

# funcs_man_eval.copy_event_files(meta_df, dest_dir_files)

num_swps_to_analyze = 5
aprox_std_val = 5

# for i, fn  in enumerate(meta_df['Name of recording']):
#     if i >=12:
#         break
i = 9 # start with
fn = meta_df['Name of recording'][i]

# load the latest summary df
sum_df = pd.read_excel(results_dir + 'sum_results.xlsx')
# remove all unnamed columns
sum_df = sum_df.loc[:, ~ sum_df.columns.str.contains('Unnamed')]
sum_df.to_excel(results_dir + 'archive/' + today +'sum_results.xlsx')

chan = meta_df['Channels to use'][i]
first_sweep = meta_df['Analysis start at sweep (number)'][i]
last_point = meta_df['Cut sweeps last part (ms)'][i]
swps_to_analyse_read = meta_df['swps_to_analyse'][i]
fs = meta_df['Sampling rate (Hz)'][i]
# analysis_end = meta_df['Analysis length (min)'][i] * fs # datapoints
analysis_end = 1_200_000
fn_path = dest_dir_files + fn
# 20 seconds - length each swp; 5500 (datapoints) / 20 (sampling rate (kHz)) * 0.001 (miliseconds to seconds)
analysis_length_sec = (20 - ((5500 / 20) * (0.001))) * num_swps_to_analyze

swps_keep = ast.literal_eval(meta_df.swps_to_analyse.values[i])
if num_swps_to_analyze <= len(swps_keep):
    swps_keep = ast.literal_eval(meta_df.swps_to_analyse.values[i])[:num_swps_to_analyze]
swps_delete = list(set(range(30)) - set(swps_keep))

# update sum_df
if (fn, chan) not in list(zip(sum_df['filename'].values, sum_df['channel'].values)):
    data_add = pd.DataFrame({'filename' : fn,
                            'channel': chan, 
                            'sampl_rate_Hz':fs,
                            'analysis_start_swp_num': first_sweep,
                            'num_analyzed_swsp': len(swps_keep),
                            'cut_swp_last_part_dp': 5500,
                            'analysis_len_sec': analysis_length_sec,
                            'avg_IEI_ms':[np.nan],
                            'avg_amp_pA':[np.nan],	
                            'avg_abs_amp_pA':[np.nan],
                            'avg_rise_times_10_90_ms_barb':[np.nan], 
                            'avg_decay_times_10_90_ms_barb':[np.nan],
                            'raw_mean_bl_signal_pA':[np.nan],
                            'hann_mean_bl_signal_pA':[np.nan],
                            'Manually revised':'no',
                            'raw_stdv_bl_signal_pA':[np.nan],
                            'hann_stdv_bl_signal_pA':[np.nan],
                            'time_analysis_end':[np.nan],
                            'man_revised':'no'})
    sum_df = pd.concat([sum_df, data_add], ignore_index=True)
    sum_df.to_excel(results_dir + 'sum_results.xlsx', sheet_name='Summary results')

# load the trace
trace = MiniTrace.from_axon_file_MP(filepath = fn_path,
                                    channel = chan,
                                    scaling = 1,
                                    unit = 'mV',
                                    sweeps_delete = swps_delete,
                                    first_point = 0,
                                    last_point = 5500)


data_analysis = trace.data[:analysis_end,] # keep only as much data as needed
# signal_lp = funcs_man_eval.lowpass(data_analysis, 800, order=1)
time = np.linspace(0,round((data_analysis.shape[0] - 1) / (fs / 1000)),
                data_analysis.shape[0]) # seconds
# std_approx = funcs_man_eval.lowpass(data_analysis - aprox_std_val, 3, order=4)  # 3 Hz instead of 800

if os.path.exists(results_dir + 'individ_dfs/' + fn[:fn.rfind('.abf')] + '_' + str(chan) + '_individual.xlsx'):
    individ_df = pd.read_excel(results_dir + 'individ_dfs/' + fn[:fn.rfind('.abf')] + '_' + str(chan) +'_individual.xlsx')
    if len(individ_df.columns) == 0:
        x, y = [], []
    else:
        x = individ_df['x (dp)'][individ_df['x (dp)'].values < analysis_end].values
        y = individ_df['y (pA)'][:len(x)].values
else:
    individ_df = pd.DataFrame()
    individ_df.to_excel(results_dir + 'individ_dfs/' + fn[:fn.rfind('.abf')] + '_' + str(chan) +'_individual.xlsx')
    x, y = [], []

org_vals = np.array(list(zip(x, y)), dtype='object')

funcs_man_eval.MiniSpontProof(sum_df_path = results_dir + 'sum_results.xlsx', num_swps = len(swps_keep), sampling = trace.sampling,
                            individ_df_path = results_dir + 'individ_dfs/' + fn[:-4] + '_' +str(chan) + '_individual.xlsx',
                            time_ = time, signal_lp = data_analysis, original_values = org_vals, fs = fs)

print(f'complete with the analysis of i {i}, {fn} channel {chan}')
