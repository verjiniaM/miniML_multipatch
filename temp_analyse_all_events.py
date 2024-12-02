import ast
import sys
sys.path.append('./core/')
from miniML import MiniTrace, EventDetection
import tensorflow as tf
print(tf.__version__)
import pandas as pd

# Spontan data

OUT_DIR = '/alzheimer/verjinia/data/analysed_events/'

data_path = '/alzheimer/verjinia/data/recordings/'
events_df = pd.read_excel('/alzheimer/verjinia/data/metadata_tables/' +  \
                            '2024-09-16_merged_spontan_intrinsic_copy_used_for_model_evl.xlsx')

analysed_df = pd.read_excel(OUT_DIR + 'all_analyzed_events.xlsx')
# start_indx = len(analysed_df)
start_indx = 0
num_swps_to_analyze = 4 # takes the first 4 decent sweeps

scaling = 1
unit = 'pA'
win_size = 750
stride = int(win_size/30)

# result_df = pd.DataFrame(columns = ['OP','patcher', 'patient_age','file_name', 'channel', 'slice', \
#                                     'cell_ID', 'day', 'treatment', 'hrs_incubation', 'repatch', \
#                                     'Rs', 'Rin', 'resting_potential', 'high K concentration', \
#                                     'analysis_len (sec)', 'event count', 'amplitude mean', 'amplitude std', \
#                                     'amplitude median', 'charge mean', 'risetime mean (10 - 90)', \
#                                     'halfdecaytime mean', 'frequency (Hz)', 'comment'])

result_df = analysed_df
for i, fn in enumerate(events_df['Name of recording'][start_indx:]):
    i = i + start_indx

    if events_df.resting_potential.values[i] in analysed_df.resting_potential.values:
        continue

    if i == 671 or i == 673 or i == 954: # skipped files, can't solve problem
        continue
    file_path  = data_path + fn
    chan = events_df.cell_ch.values[i] 
    swps_keep = ast.literal_eval(events_df.swps_to_analyse.values[i])

    save_individ_dir = OUT_DIR + 'indivd_model_retrain1_Oct/' +f"{fn[:-4]}_ch{str(chan)}.csv"

    if num_swps_to_analyze <= len(swps_keep):
        swps_keep = ast.literal_eval(events_df.swps_to_analyse.values[i])[:num_swps_to_analyze]
    swps_delete = list(set(range(30)) - set(swps_keep))

    trace = MiniTrace.from_axon_file_MP(filepath = file_path,
                                        channel = chan,
                                        scaling = scaling,
                                        unit = unit,
                                        sweeps_delete = swps_delete,
                                        first_point = 0,
                                        last_point = 5500)
    

    detection = EventDetection(data = trace,
                            model_path='./models/transfer_learning/human_pyramids_L2_3/2024_Oct_29_lstm_transfer.h5',
                            window_size = win_size,
                            model_threshold = 0.5,
                            batch_size = 512,
                            event_direction = 'negative',
                            training_direction = 'negative',
                            compile_model = True)

    detection.detect_events(stride=stride,
                            eval = True,
                            convolve_win = 20,
                            resample_to_600 = True)

    df_meta = pd.DataFrame({'OP': events_df.OP.values[i],
                            'patcher': events_df.patcher.values[i], 
                            'patient_age': events_df.patient_age.values[i],
                            'file_name' : fn,
                            'channel' : str(chan), 
                            'slice' : events_df.slice.values[i], 
                            'cell_ID': events_df.cell_ID.values[i], 
                            'day' : events_df.day.values[i], 
                            'treatment': events_df.treatment.values[i], 
                            'hrs_incubation': events_df.hrs_incubation_x.values[i],  # what are the x and y for???
                            'repatch': events_df.repatch.values[i],  
                            'Rin': events_df.Rin.values[i], 
                            'resting_potential': events_df.resting_potential.values[i], 
                            'high K concentration': events_df['high K concentration'].values[i],
                            'analysis_len (sec)': len(trace.data / 20_000)}, index = [0])
    
    if len(detection.events) == 0:
        result_df = pd.concat([result_df, df_meta], axis = 0)
        continue

    individual, avgs = detection.save_to_csv(save_individ_dir)

    df_to_add = pd.concat([df_meta, avgs.T], axis = 1)
    result_df = pd.concat([result_df, df_to_add], axis = 0)

    if (i + 1) % 10 == 0: 
        result_df.to_excel(OUT_DIR + 'all_analyzed_events.xlsx', index=False)
    




# %%

# check if the incubation times x nad y are the samedf_unique1 = events_df.drop_duplicates()
import pandas as pd
import matplotlib.pyplot as plt

# df_unique = results_df.drop_duplicates()
df_no_duplicates = pd.read_excel('/alzheimer/verjinia/data/analysed_events/no_repetitions_all_analyzed_events.xlsx')

df_slice = df_no_duplicates[df_no_duplicates['repatch'] == 'no']
df_repatch = df_no_duplicates[df_no_duplicates['repatch'] == 'yes']

df_repatch = df_repatch[df_repatch['amplitude mean'] > -2500]

# chose the df
df = df_slice

plt.figure(figsize=(10, 6))

# Plot the scatter points with jitter using np.linspace
for k, treatment in enumerate(df['treatment'].unique()):
    if treatment == 'TTX':
        continue
    for c, day in enumerate(df['day'].unique()):
        d = k + c
        day_treatment_data = df[(df['day'] == day) & (df['treatment'] == treatment)]
        mean = day_treatment_data['amplitude mean'].mean()
        jitter = np.linspace(-0.1, 0.1, len(day_treatment_data))  # Add jitter using np.linspace
        plt.scatter(d + jitter, day_treatment_data['amplitude mean'], label=f'Treatment {treatment}, Day {day}')
        plt.scatter(d, mean, color='black', s=100, edgecolor='black', linewidth=1)

# Customize the plot
plt.xlabel('Day')
plt.ylabel('Amplitude Mean')
plt.title('Repatch Mean Amplitude by Day and Treatment')
plt.legend()
plt.grid(True)
plt.show()

# Print the values of the means by day and treatment
day_treatment_means = df.groupby(['day', 'treatment'])['amplitude mean'].mean()
print("Means by Day and Treatment:")
print(day_treatment_means)