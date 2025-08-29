# # get the updated x, y values
# # if 'a' not pressed before, updated = org
# updated_values = org_vals
# x = updated_values[:, 0].astype('float64')
# y = updated_values[:, 1].astype('float64')

# # sort the updated x, y values (for ascending x)
# x, y = zip(*sorted(zip(x, y)))
# x = np.array(x)
# y = np.array(y)

# # get the datapoint from the ms
# x_dp = (x).astype('int')

# # calculate the interevent interval
# x_pd = pd.Series(x)
# interevent_interval = x_pd.shift(-1) - x_pd
# average_interevent_interval = np.nanmean(interevent_interval)

# # instantiate parameters
# parameters = ParametersCalculator(signal_lp, x_dp, y, fs)

#     # calculate the amplitude
# amplitude = parameters.amplitude_stdev()[0]
# amplitude[interevent_interval < 45] = np.nan
# average_amplitude = np.nanmean(amplitude)

# # calculate the mean rise (10-90%) and the mean decay time (90-10%)
# mean_rise_time = parameters.rise_time()[0]
# mean_decay_time = parameters.decay_time()[0]

# # calculate the stdev of the baseline signal
# stdev = parameters.amplitude_stdev()[1]
# average_stdev = np.nanmean(stdev)

#%%
import numpy as np
import pandas as pd
from scipy import signal
from miniML import MiniTrace
from miniML_functions import (get_event_peak, get_event_baseline, get_event_onset, get_event_risetime,
                              get_event_halfdecay_time, get_event_charge)

def init_dicts(attr_names, shape, dtype):
    ''' initialize multiple 1d ndarrays with given shape containing NaNs '''
    empty_d = {}
    for label in attr_names:
        value = -1 if 'int' in str(dtype) else np.NaN
        empty_d[label] = np.full(int(shape), value, dtype=dtype)
    return empty_d

def hann_filter(data, filter_size):
    win = signal.windows.hann(filter_size)
    return signal.convolve(data, win, mode = 'same') / sum(win)

def get_event_properties(data_def, event_locations, sampling,
            window_size = 750,convolve_win = 20, filter: bool = True):
    '''
    Find more detailed event location properties required for analysis. Namely, baseline, event onset,
    peak half-decay and 10 & 90% rise positions. Also extracts the actual event properties, such as
    amplitude or half-decay time.
    
    Parameters
    ------
    filter: bool
        If true, properties are extracted from the filtered data.
    '''
    ### Prepare data
    diffs = np.diff(event_locations, append = len(data_def)) # Difference in points between the event locations
    add_points = int(window_size/3)
    after = window_size + add_points
    positions = event_locations
    
    ### Set parameters for charge calculation
    factor_charge = 4
    num_combined_charge_events = 1
    calculate_charge = False # will be set to True in the loop if double event criteria are fulfilled; not a flag for charge

    if np.any(positions - add_points < 0) or np.any(positions + after >=  len(data_def)):
        raise ValueError('Cannot extract time windows exceeding input data size.')
    
    if filter:
        mini_trace = hann_filter(data = data_def, filter_size = convolve_win)
    else:
        mini_trace = data_def

    mini_trace *= -1 # because the event_direction is negative

    temp_dict = init_dicts(['event_peak_locations', 'event_start', 'min_positions_rise', 'max_positions_rise'], positions.shape[0], dtype = np.int64)
    params_dict = init_dicts(['event_peak_values', 'event_bsls', 'decaytimes', 'charges', 'risetimes', 'half_decay'], positions.shape[0], dtype = np.float64)

    for ix, position in enumerate(positions):
        indices = position + np.arange(-add_points, after)
        data = mini_trace[indices]
            
        if filter:
            data_unfiltered = data_def * -1 # negative event direction
        else:
            data_unfiltered = data
        
        event_peak = get_event_peak(data = data,event_num = ix,add_points = add_points,window_size = window_size,diffs = diffs)
        
        temp_dict['event_peak_locations'][ix] = int(event_peak)
        params_dict['event_peak_values'][ix] = data[event_peak]
        peak_spacer = int(window_size/100)
        params_dict['event_peak_values'][ix] = np.mean(data_unfiltered[int(event_peak-peak_spacer):int(event_peak+peak_spacer)])
        
        baseline, baseline_var = get_event_baseline(data = data, event_num = ix, diffs = diffs, add_points = add_points,
                                                    peak_positions = temp_dict['event_peak_locations'], positions = positions)
        params_dict['event_bsls'][ix] = baseline
        
        onset_position = get_event_onset(data = data, peak_position = event_peak, baseline = baseline,
                                            baseline_var = baseline_var)
        temp_dict['event_start'][ix] = onset_position
        
        risetime, min_position_rise, max_position_rise = get_event_risetime(data = data, peak_position = event_peak,
                                                                            onset_position = onset_position)
        params_dict['risetimes'][ix] = risetime
        temp_dict['min_positions_rise'][ix] = min_position_rise
        temp_dict['max_positions_rise'][ix] = max_position_rise

        level = baseline + (data[event_peak] - baseline) / 2
        if diffs[ix] < add_points: # next event close; check if we can get halfdecay
            right_lim = diffs[ix]+add_points # Right limit is the onset of the next event
            test_arr =  data[event_peak:right_lim]
            if test_arr[test_arr<level].shape[0]: # means that event goes below 50% ampliude before max rise of the next event; 1/2 decay can be calculated
                halfdecay_position, halfdecay_time = get_event_halfdecay_time(data = data[0:right_lim],
                                                                                peak_position = event_peak, baseline = baseline)
            else:
                halfdecay_position, halfdecay_time = np.nan, np.nan
        else:  
            halfdecay_position, halfdecay_time = get_event_halfdecay_time(data = data, peak_position = event_peak, 
                                                                            baseline = baseline)

        params_dict['half_decay'][ix] = halfdecay_position
        params_dict['decaytimes'][ix] = halfdecay_time
        
        # calculate params_dict['charges']
        ### For charge; multiple event check done outside function.
        if ix < positions.shape[0]-1:
            if num_combined_charge_events ==  1: # define onset position for charge calculation
                onset_in_trace = positions[ix] - (add_points-temp_dict['event_start'][ix])
                baseline_for_charge = params_dict['event_bsls'][ix]

            if np.isnan(params_dict['half_decay'][ix]):
                num_combined_charge_events += 1
            
            else:
                ### Get distance from peak to next event location.
                peak_in_trace = positions[ix] + (temp_dict['event_peak_locations'][ix] - add_points)
                next_event_location = positions[ix+1]
                delta_peak_location = next_event_location - peak_in_trace
                
                # determine end of area calculation based on event decay
                endpoint = int(temp_dict['event_peak_locations'][ix] + \
                                factor_charge*(int(params_dict['half_decay'][ix]) - temp_dict['event_peak_locations'][ix]))
                delta_peak_endpoint = endpoint-temp_dict['event_peak_locations'][ix]

                if delta_peak_location > delta_peak_endpoint: # Next event_location further away than the charge window; calculate charge
                    calculate_charge = True
                else:
                    num_combined_charge_events += 1

            if calculate_charge:
                endpoint_in_trace = positions[ix] + (temp_dict['event_peak_locations'][ix] - add_points) + delta_peak_endpoint
                charge = get_event_charge(data = mini_trace, start_point = onset_in_trace, end_point = endpoint_in_trace,
                                           baseline = baseline_for_charge, sampling = sampling)
                
        else: # Handle the last event
            if num_combined_charge_events ==  1: # define onset position for charge calculation
                onset_in_trace = positions[ix] - (add_points - temp_dict['event_start'][ix])
                baseline_for_charge = params_dict['event_bsls'][ix]
            
            peak_in_trace = positions[ix] + (temp_dict['event_peak_locations'][ix] - add_points)
            endpoint = int(temp_dict['event_peak_locations'][ix] + factor_charge*(int(params_dict['half_decay'][ix]) - temp_dict['event_peak_locations'][ix]))
            delta_peak_endpoint = endpoint-temp_dict['event_peak_locations'][ix]
            endpoint_in_trace = positions[ix] + (temp_dict['event_peak_locations'][ix] - add_points) + delta_peak_endpoint
            
            if endpoint_in_trace > mini_trace.shape[0]:
                endpoint_in_trace = mini_trace.shape[0]

            charge = get_event_charge(data = mini_trace, start_point = onset_in_trace, end_point = endpoint_in_trace, baseline = baseline_for_charge, sampling = sampling)
            calculate_charge = True
        if calculate_charge: # Charge was caclulated; check how many potentially overlapping events contributed.
            charge = [charge/num_combined_charge_events]*num_combined_charge_events
            for ix_adjuster in range(len(charge)):
                params_dict['charges'][ix-ix_adjuster] = charge[ix_adjuster]
                        
            # Reset values after calculation
            calculate_charge = False
            num_combined_charge_events = 1

    ## Convert units
    params_dict['event_peak_values'] *=  -1
    params_dict['event_bsls'] *=  -1
    params_dict['risetimes'] *=  sampling
    params_dict['decaytimes'] *=  sampling
    params_dict['charges'] *=  -1

    ## map indices back to original         
    for ix, position in enumerate(positions):
        temp_dict['event_peak_locations'][ix] = int(temp_dict['event_peak_locations'][ix] + event_locations[ix] - add_points)
        temp_dict['event_start'][ix] = int(temp_dict['event_start'][ix] + event_locations[ix] - add_points)
        temp_dict['min_positions_rise'][ix] = int(min_position_rise + event_locations[ix] - add_points)
        temp_dict['max_positions_rise'][ix] = int(max_position_rise + event_locations[ix] - add_points)            
        
        if not np.isnan(params_dict['half_decay'][ix]):
            params_dict['half_decay'][ix] = int(params_dict['half_decay'][ix] + event_locations[ix] - add_points)
    return temp_dict, params_dict


fn_path = '/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/data/human/data_spontan/recordings/24d12007.abf'
analyzed_df = pd.read_excel('/Users/verjim/laptop_D_17.01.2022/Schmitz_lab/results/human/paper_figs_collected_checked/events/individ_dfs/24d12007_5_individual.xlsx')
chan = 5
swps_delete = [a for a in range(5,29)]
analysis_end = 1_200_000
convolve_win = 20
sampl_rate_kHz = 20

trace = MiniTrace.from_axon_file_MP(filepath = fn_path,
                                    channel = chan,
                                    scaling = 1,
                                    unit = 'mV',
                                    sweeps_delete = swps_delete,
                                    first_point = 0,
                                    last_point = 5500)


signal_no_filter = trace.data[:analysis_end,]

# for func_dev
data_def = signal_no_filter
event_locations = analyzed_df['x (location)'].to_numpy(int) * sampl_rate_kHz
window_size = 750
convolve_win = 20
sampling = trace.sampling

event_dict, params_dict = get_event_properties(data_def, event_locations, sampling)