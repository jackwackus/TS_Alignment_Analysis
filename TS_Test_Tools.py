"""
Author: Jack Connor
Date Created: 11/16/21
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import find_peaks, square

def read_config(read_file):
    """
    Reads a configuration file and writes data to a dictionary
    Args:
        read_file (str): path to configuration text file
    Returns:
        config_dic (dict): dictionary containing configuration data
    """
    config_dic = {}
    with open(read_file) as f:
        for line in f:
            if line.find('\n') > 0:
                line = line[:-1]
            sep = line.find("=")
            object_name = line[0:sep]
            if len(object_name) < 1:
                continue
            object_value = line[sep+1:]
            config_dic[object_name] = object_value
    return config_dic

def read_parameter_list():
    """
    Reads a list of instrument parameter names from a text file. Writes them to a list.
    Returns:
        parm_list (list): list of parameter names
    """
    fname = "Parameter List.txt"
    parm_list = []
    with open(fname, 'r') as f:
        for line in f:
            if line.find('\n') > 0:
                parameter = line[:-1]
            else:
                parameter = line
            parm_list += [parameter]
    return parm_list

def ingest_pulse_log(test_name, ext, f_suffix = 'log'):
    """
    Reads in log of on/off signals to control solenoid.
    Args:
        test_name (str): name used to label test. formatted yyyymmdd
        ext (str): extension to log filename suffix. present in filename if multiple subtests have been run
        f_suffix (str): filename suffix following the test name
    Returns:
        pulse_log (pandas.DataFrame): data table representing pulse log
    """
    working_dir = os.getcwd()
    file_path = f'{working_dir}\\data\\{test_name}\\Pulse_Log\\{test_name}_{f_suffix}{ext}.csv'
    pulse_log = pd.read_csv(file_path)
    return pulse_log

def generate_condensed_data(data_dic, pulse_log):
    """
    Generates a dictionary containing instrument specific dataframes, each cut to the start and end times indicated by the pulse log.
    Args:
        data_dic (dict): dictionary containing instrument specific dataframes
        pulse_log (pandas.DataFrame): dataframe delineating the start and end times of the sample pulses
    Returns:
        condensed_data_dic (dict): dictionary containing instrument specific dataframes, each cut to the start and end times indicated by the pulse log
    """
    condensed_data_dic = {}
    start_time = min(pulse_log['Timestamp'])
    end_time = max(pulse_log['Timestamp'])
    for instrument in data_dic:
        data = data_dic[instrument]
        cut_data = data[data['Timestamp'] >= start_time]
        condensed_data = cut_data[cut_data['Timestamp'] <= end_time]
        condensed_data_dic[instrument] = condensed_data.sort_values('Timestamp').reset_index(drop = True)
    return condensed_data_dic

def plot_condensed_data(condensed_data_dic):
    """
    Generates a stack of time series plots, one plot for each instrument in the condensed_data_dic.
    Args:
         condensed_data_dic (dict): dictionary containing instrument specific dataframes, each cut to the start and end times indicated by the pulse log
    Returns:
        fig (matplotlib.pyplot.figure): figure containing stacked subplots for instrument time series
    """
    n_plots = len(condensed_data_dic)
    fig = plt.figure()
    i = 1
    plot_dict = {}
    for instrument in condensed_data_dic:
        plot_dict[instrument] = fig.add_subplot(n_plots, 1, i)
        data = condensed_data_dic[instrument]
        [x, y] = list(data)
        plot_dict[instrument].plot(data[x], data[y])
        plot_dict[instrument].set_ylabel(instrument)
        i += 1
    return fig

def ProcessDataForFindPeaks(data):
    """
    Pulls time and data values from instrument specific data frame. Converts datetime values to time in seconds. Places time and data values in numpy arrays.
    Args:
        data (pandas.DataFrame): dataframe containing two columns: time in datetime, and data
            time must be in column 0, data in column 1
    Returns:
        start_time (datetime.datetime): first datetime value in the dataset
        np.array(T) (numpy.array): 1 dimensional numpy array of time in seconds
        np.array(D) (numpy.array): 1 dimensional array of data
    """
    i = 0
    T = []
    D = []
    [X, Y] = list(data)
    for x, y in zip(data[X], data[Y]):
        if i == 0:
            start_time = x
            T += [0]
        else:
            dt = (x - data[X][i-1]).seconds
            t = T[i-1] + dt
            T += [t]
        D += [float(y)]
        i += 1
    return start_time, np.array(T), np.array(D)

def find_data_peaks(condensed_data, pulse_log, pulse_time):
    """
    Uses square wave best-fit method to identify peak times in dataset of periodic sample pulses.
    Iterates over all possible offset times (seconds) in a period and generates square waves with those offset times.
    For each offset time, takes the inner product of the square wave and the data. Identifies the offset time with the largest inner product.
    Determines the peak times of the square wave with that offset time.
    Args:
        condensed_data (pandas.DataFrame): dataframe containing two columns: time in datetime, and data
            time must be in column 0, data in column 1
        pulse_log (pandas.DataFrame): dataframe delineating the start and end times of the sample pulses
        pulse_time (int): timespan of sample pulses in seconds (half of the pulse cycle period)
    Returns:
        peak_times (list): list of datetime values representing the times of instrument response to each sample pulse
        t_offset (int): time in seconds representing offset time of best-fit square wave
        probs (numpy.array): array of all best-fit model inner product values tested
    """
    start_time, t, data = ProcessDataForFindPeaks(condensed_data)

    T = 2*pulse_time
    f = 1/T
    dt = 1
    t_offset = np.arange(0,T,dt)
    num_guesses = t_offset.size
    probs = np.zeros((num_guesses))

    count = 0
    for i,t0 in enumerate(t_offset):
        #model is guaranteed to be same size as data, since model is a function of t
        #also, since model is defined by t, which only reflects available timestamps, model is only evaluated at available timestamps
        model = 0.1*square(2*np.pi* f *(t-t0))
        prob = np.sum(model*data)
        probs[i] = prob

    best_t0 = t_offset[np.argmax(probs)]

    num_peaks = len(pulse_log[pulse_log['System State'] == 'On']['Timestamp'])
    alt_peaks = [best_t0 + i*T for i in range(num_peaks)]

    best_model = square(2*np.pi*f*(t -best_t0))

    peak_times = [start_time + timedelta(seconds = int(t)) for t in alt_peaks]

    """
    plt.figure() 
    plt.plot(t_offset, probs) 

    plt.figure()
    plt.plot(t, data)
    plt.plot(t, best_model)

    plt.figure()
    plt.plot(t,data)
    plt.scatter(alt_peaks, [1]*len(alt_peaks), color='k')
    plt.show()
    """
    return peak_times, t_offset, probs

def find_model_peaks(condensed_data, pulse_log, pulse_time):
    """
    Uses square wave best-fit method to identify peak times in dataset of periodic sample pulses.
    Iterates over all possible offset times (seconds) in a period and generates square waves with those offset times.
    For each offset time, takes the inner product of the square wave and the data. Identifies the offset time with the largest inner product.
    Determines the indices at which peaks occur in the data arrays, as well as their times in seconds from the start of the sample period.
    Args:
        condensed_data (pandas.DataFrame): dataframe containing two columns: time in datetime, and data
            time must be in column 0, data in column 1
        pulse_log (pandas.DataFrame): dataframe delineating the start and end times of the sample pulses
        pulse_time (int): timespan of sample pulses in seconds (half of the pulse cycle period)
    Returns:
        peak_times (list): list containing indices at which peaks occur in the data array
        alt_peaks (list): list of times at which peaks occur in seconds elapsed over the sampling period
    """
    start_time, t, data = ProcessDataForFindPeaks(condensed_data)

    T = 2*pulse_time
    f = 1/T
    dt = 1
    t_offset = np.arange(0,T,dt)
    num_guesses = t_offset.size
    probs = np.zeros((num_guesses))

    count = 0
    for i,t0 in enumerate(t_offset):
        #model is guaranteed to be same size as data, since model is a function of t
        #also, since model is defined by t, which only reflects available timestamps, model is only evaluated at available timestamps
        model = 0.1*square(2*np.pi* f *(t-t0))
        prob = np.sum(model*data)
        probs[i] = prob

    best_t0 = t_offset[np.argmax(probs)]

    num_peaks = len(pulse_log[pulse_log['System State'] == 'On']['Timestamp'])
    alt_peaks = [best_t0 + i*T for i in range(num_peaks)]
    peak_indices = [i for i, time in enumerate(t) if time in alt_peaks]

    return peak_indices, alt_peaks

def find_data_peaks_2(condensed_data, pulse_log):
    """
    Uses square wave best-fit method to identify peak times in dataset of periodic sample pulses.
    Iterates over all possible offset times (seconds) in a period and generates square waves with those offset times.
    For each offset time, takes the inner product of the square wave and the data. Identifies the offset time with the largest inner product.
    Determines the best-fit square wave for the data (with it's corresponding offset time).
    Args:
        condensed_data (pandas.DataFrame): dataframe containing two columns: time in datetime, and data
            time must be in column 0, data in column 1
        pulse_log (pandas.DataFrame): dataframe delineating the start and end times of the sample pulses
    Returns:
        best_model (numpy.array): numpy array of best-fit model for the data
        t (numpy.array): 1 dimensional numpy array of time values in seconds corresponding to data
        data (numpy.array): 1 dimensional numpy array of data values
    """

    start_time, t, data = ProcessDataForFindPeaks(condensed_data)

    T = 120
    f = 1/T
    dt = 1
    t_offset = np.arange(0,T,dt)
    num_guesses = t_offset.size
    probs = np.zeros((num_guesses))
    data -= np.mean(data)

    count = 0
    for i,t0 in enumerate(t_offset):
        model = 0.1*square(2*np.pi* f *(t-t0))
        prob = np.sum(model*data)
        probs[i] = prob

    best_t0 = t_offset[np.argmax(probs)]

    num_peaks = len(pulse_log[pulse_log['System State'] == 'On']['Timestamp'])
    alt_peaks = [best_t0 + i*T for i in range(num_peaks)]

    best_model = square(2*np.pi*f*(t -best_t0))

    peak_times = [start_time + timedelta(seconds = int(t)) for t in alt_peaks]

    """
    plt.figure() 
    plt.plot(t_offset, probs) 

    plt.figure()
    plt.plot(t, data)
    plt.plot(t, best_model)

    plt.figure()
    plt.plot(t,data)
    plt.scatter(alt_peaks, [1]*len(alt_peaks), color='k')
    plt.show()
    """
    return best_model, t, data

def plot_model_matching(condensed_data, pulse_log, padding = None, figsize = (8,6)):
    """
    Ingests G2401 timestamp test data. Runs square wave best-fit method.
    Generates a series of plots illustrating the sampling of square waves from offset=0 to the best-fit offset.
    Args:
        condensed_data (pandas.DataFrame): dataframe containing two columns: time in datetime, and data
            time must be in column 0, data in column 1
        pulse_log (pandas.DataFrame): dataframe delineating the start and end times of the sample pulses
        padding (float): spacing on left and right of plot
            must be a positive number less than 0.5
        figsize (tuple): tuple containing width and height of figure, in inches
    Returns:
        saves a series of plots numbered and named with zero padding (up to 3 digits) to be compatible with ffmpeg 
    """
    left = padding
    right = 1 - padding

    start_time, t, data = ProcessDataForFindPeaks(condensed_data)

    T = 120
    f = 1/T
    dt = 1
    t_offset = np.arange(0,T,dt)
    num_guesses = t_offset.size
    probs = np.zeros((num_guesses))
    data -= np.mean(data)

    count = 0
    for i,t0 in enumerate(t_offset):
        model = 0.1*square(2*np.pi* f *(t-t0))
        prob = np.sum(model*data)
        probs[i] = prob

    best_t0 = t_offset[np.argmax(probs)]

    num_peaks = len(pulse_log[pulse_log['System State'] == 'On']['Timestamp'])
    alt_peaks = [best_t0 + i*T for i in range(num_peaks-1)]
    data_alt_peaks = [data[t] for t in alt_peaks]

    best_model = square(2*np.pi*f*(t -best_t0))

    peak_times = [start_time + timedelta(seconds = int(t)) for t in alt_peaks]

    frame_n = 1
    for t0 in range(best_t0):

        model = square(2*np.pi*f*(t-t0))
       
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(1,1,1)
        ax.plot(t, data, color = 'b')
        twin_ax = ax.twinx()
        twin_ax.plot(t, model, color = 'r', alpha = .7)

        ax.set_ylabel('G2401 Data', color = 'b')
        ax.tick_params(axis='y', colors='b')
        twin_ax.set_ylabel('Model', color = 'r')
        twin_ax.spines['left'].set_color('blue')
        twin_ax.spines['right'].set_color('r')
        twin_ax.tick_params(axis='y', colors='r')
        
        plt.subplots_adjust(left = left, right = right)
        #plt.show()
        filename = f'demo_vid\\{frame_n:02}.png'
        fig.savefig(filename)
        plt.close()
        frame_n += 1


    fig_2 = plt.figure(figsize = figsize)
    ax_2 = fig_2.add_subplot(1,1,1)
    ax_2.plot(t, data, color = 'b')
    twin_ax2 = ax_2.twinx()
    twin_ax2.plot(t, best_model, color = 'r', alpha = .7)
    ax_2.scatter(alt_peaks, data_alt_peaks, color='g')

    ax_2.set_ylabel('G2401 Data', color = 'b')
    ax_2.tick_params(axis='y', colors='b')
    twin_ax2.set_ylabel('Model', color = 'r')
    twin_ax2.spines['left'].set_color('blue')
    twin_ax2.spines['right'].set_color('r')
    twin_ax2.tick_params(axis='y', colors='r')

    plt.subplots_adjust(left = left, right = right)
    #plt.show()
    filename = f'demo_vid\\{frame_n:02}.png'
    fig_2.savefig(filename)
    plt.close()

    return

def plot_data_with_peaks(condensed_data_dic, pulse_log, pulse_time):
    """
    Plots timestamp test data for all instruments in data dictionary. Identifies peak times and indicates them with red dots.
    Plots probability distributions of response times for all instruments in data dictionary.
    Args:
        condensed_data_dic (dict): dictionary containing instrument specific dataframes, each cut to the start and end times indicated by the pulse log       
        pulse_log (pandas.DataFrame): dataframe delineating the start and end times of the sample pulses
        pulse_time (int): timespan of sample pulses in seconds (half of the pulse cycle period)
    Returns:
        fig (matplotlib.pyplot.figure): figure with subplots illustrating timestamp test data and peak times for all instruments
        fig_2 (matplotlib.pyplot.figure): figure with subplots illustrating probability distributions of response times for all instruments
        data_peaks_dic (dict): dictionary containing list of peak times for each instrument
    """
    n_plots = len(condensed_data_dic)
    fig = plt.figure()
    fig_2 = plt.figure()
    i = 1
    plot_dict = {}
    plot_dict_2 = {}
    data_peaks_dic = {}
    for instrument in condensed_data_dic:
        plot_dict[instrument] = fig.add_subplot(n_plots, 1, i)
        data = condensed_data_dic[instrument]
        data_peaks, t_offset, probs = find_data_peaks(data, pulse_log, pulse_time)
        [x, y] = list(data)
        plot_dict[instrument].plot(data[x], data[y])
        for time in data_peaks:
            df = data[data['Timestamp'] == time]
            if len(df['Timestamp']) > 0:
                plot_dict[instrument].scatter(df[x], df[y], color = 'r')
            else:
                try:
                    t_1 = time - timedelta(seconds = 1)
                    t_2 = time + timedelta(seconds = 1)
                    df1 = data[data['Timestamp'] == t_1]
                    df2 = data[data['Timestamp'] == t_2]
                    value = (df1[y].iloc[0] + df2[y].iloc[0])/2
                    plot_dict[instrument].scatter(time, value, color = 'r')
                except:
                    pass
        plot_dict[instrument].set_ylabel(instrument)
        data_peaks_dic[instrument] = data_peaks
        plot_dict_2[instrument] = fig_2.add_subplot(n_plots, 1, i)
        plot_dict_2[instrument].plot(t_offset, probs)
        plot_dict_2[instrument].set_ylabel(instrument)
        i += 1
    fig.suptitle('Test Data and Peak Times')
    fig_2.suptitle('Response Time Probability Distributions')
    return fig, fig_2, data_peaks_dic

def plot_data_with_peaks_2(condensed_data_dic, pulse_log):
    """
    For all instruments, plots timestamp test data with best-fit square waves overlayed.
    Args:
        condensed_data_dic (dict): dictionary containing instrument specific dataframes, each cut to the start and end times indicated by the pulse log       
        pulse_log (pandas.DataFrame): dataframe delineating the start and end times of the sample pulses
    Returns:
        fig (matplotlib.pyplot.figure): figure containing subplots illustrating instrument and best-fit data
    """
    n_plots = len(condensed_data_dic)
    fig = plt.figure()
    i = 1
    plot_dict = {}
    data_peaks_dic = {}
    for instrument in condensed_data_dic:
        plot_dict[instrument] = fig.add_subplot(n_plots, 1, i)
        data = condensed_data_dic[instrument]
        best_model, t, data = find_data_peaks_2(data, pulse_log)
        plot_dict[instrument].plot(t, data)
        plot_dict[instrument+'_2'] = plot_dict[instrument].twinx()
        plot_dict[instrument+'_2'].plot(t, best_model, color = 'orange')
        plot_dict[instrument].set_ylabel(instrument)
        i += 1
    return fig

def plot_data_for_clicks(condensed_data_dic):
    """
    Generate plots of timestamp test data whose x-axes are in seconds rather than datetime.
    These plots can be used to write x values [in seconds] of clicked points to a text file.
    Args:
        condensed_data_dic (dict): dictionary containing instrument specific dataframes, each cut to the start and end times indicated by the pulse log       
    Returns:
        fig (matplotlib.pyplot.figure): figure containing subplots illustrating instrument data with x values in seconds
    """
    n_plots = len(condensed_data_dic)
    fig = plt.figure()
    i = 1
    plot_dict = {}
    print('\nNote, start times are shown for manual peak picking purposes, and have nothing to do with method validation.')
    for instrument in condensed_data_dic:
        plot_dict[instrument] = fig.add_subplot(n_plots, 1, i)
        data = condensed_data_dic[instrument]
        start_time, t, d = ProcessDataForFindPeaks(data)
        print(f'{instrument} start time is {start_time}.')
        plot_dict[instrument].plot(t, d)
        plot_dict[instrument].set_ylabel(instrument)
        i += 1
    return fig

def create_response_times_table(data_peaks_df, pulse_log):
    """
    Generates a table listing the response times for all instruments.
    Args:
        data_peaks_df (pandas.DataFrame): table containing lists of peak times (in datetime) for each instrument       
        pulse_log (pandas.DataFrame): table delineating the start and end times of the sample pulses
    Returns:
        pd.DataFrame(response_time_dict) (pandas.DataFrame): table containing response time of each instrument
    """
    first_pulse_time = datetime.strptime(
            pulse_log[pulse_log['System State'] == 'On']['Timestamp'].iloc[0],
            '%Y-%m-%d %H:%M:%S'
            )
    response_time_dict = {'Instrument': [], 'Response Time': []}
    for instrument in list(data_peaks_df):
        response_time = (data_peaks_df[instrument].iloc[0] - first_pulse_time).seconds
        response_time_dict['Instrument'] += [instrument]
        response_time_dict['Response Time'] += [response_time]
    return pd.DataFrame(response_time_dict)

def convert_compiled_df_to_data_dic(compiled_df, parameters_list):
    """
    Makes compiled data compatible with the functions in this module.
    Ingests complete datafile produced by Mobile Data Compiler. Samples parameter specific dataframes as specified by parameters_list.
    Stores parameter specific dataframes in a data dictionary in a format compatible with other TS_Test_Tools functions.
    Args:
        compiled_df (pandas.DataFrame): datafile generated by Mobile Data Compiler, read into python using pandas
        parameters_list (list): list of parameters to pull from compiled_df
    Returns:
        data_dic (dict): dictionary of parameter specific dataframes, keyed by parameter
    """
    data_dic = {}
    compiled_df = compiled_df.rename(columns = {'DateTime': 'Timestamp'})
    for parameter in parameters_list:
        data_dic[parameter] = pd.concat([compiled_df['Timestamp'], compiled_df[parameter]], axis = 1).dropna()
    return data_dic

def plot_best_fit_and_t0(condensed_data_dict, pulse_log, pulse_time):
    """
    Plot data with:
    Square wave best-fit superimposed,
    Best-fit response times indicated by red dots.
    Args:
        condensed_data_dict (dict): dictionary containing condensed dataframes for each parameter, keyed by parameter name
            condensed dataframes only contain datetime and parameter values, and are condensed to timespan of pulse log
        pulse_log (pandas.DataFrame): dateframe containing timestamped log of solenoid on/off switch times
        pulse_time (int): number of seconds in a sample pulse (1/2 of pulse period)
    Returns:
        fig (matplotlib.pyplot.figure): figure containing features in description above
    """
    def handle_missing_data(peak_times, data, t):
        """
        For each peak time, checks if time is in data's time array, t:
            If peak time is not in t, finds nearest available time values before and after, pulls data values at these times,
                Computes average of time and data values and saves in respective arrays.
            If peak time is in t, saves t and corresponding data value in respective arrays.
        Args:
            peak_times (list): list of peak times
            data (numpy.array): vector of data values
            t (numpy.array): vector of times corresponding to data
        Returns:
            new_times (list): list of times resulting from procedure described above
            new_vals (list): list of data values resulting from procedure described above
        """
        new_times = []
        new_vals = []
        peak_times = peak_times.copy()[:-1]
        for peak_time in peak_times:
            if peak_time not in t:
                lower_times = [time for time in t if time < peak_time]
                higher_times = [time for time in t if time > peak_time]
                closest_time_low = min(lower_times, key = lambda x: abs(x - peak_time))
                closest_time_high = min(higher_times, key = lambda x: abs(x - peak_time))
                for i, time in enumerate(t):
                    if time == closest_time_low:
                        closest_index_low = i
                    elif time == closest_time_high:
                        closest_index_high = i
                new_times += [(closest_time_high + closest_time_low)/2]
                new_vals += [(data[closest_index_high] + data[closest_index_low])/2]
            else:
                for i, time in enumerate(t):
                    if time == peak_time:
                        index = i
                new_times += [peak_time]
                new_vals += [data[index]]
        return new_times, new_vals

    relabel_dict = read_config('Relabel Dictionary.txt')

    dateparser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    pulse_tuples = []
    for i, row in pulse_log.iterrows():
        if row['System State'] == 'On':
            t0 = dateparser(row['Timestamp'])
            try:
                tf = dateparser(pulse_log.iloc[i+1]['Timestamp'])
                pulse_tuples += [(t0, tf)]
            except:
                pass
    n_plots = len(condensed_data_dict)
    fig = plt.figure()
    plot_dict = {}
    i = 1
    for parameter in condensed_data_dict:
        start_time, t, data = ProcessDataForFindPeaks(condensed_data_dict[parameter])
        plot_dict[parameter] = fig.add_subplot(n_plots, 1, i)
        plot_dict[parameter].plot(t, data, color = 'blue')
        best_model, t, data_2 = find_data_peaks_2(condensed_data_dict[parameter], pulse_log)
        plot_dict[parameter+'_2'] = plot_dict[parameter].twinx()
        plot_dict[parameter+'_2'].plot(t, best_model, color = 'orange')
        plot_dict[parameter+'_2'].set_yticks([])
        plot_dict[parameter+'_2'].set_ylabel('Best-Fit', fontsize = 'small', color = 'orange')
        model_peak_indices, peak_times = find_model_peaks(condensed_data_dict[parameter], pulse_log, pulse_time)
        if len(peak_times) - len(model_peak_indices) == 1:
            plot_dict[parameter].plot(t[model_peak_indices], data[model_peak_indices], "o", color = 'r', ms = 6, zorder = 3)
        else:
            new_times, new_vals = handle_missing_data(peak_times, data, t)
            plot_dict[parameter].plot(new_times, new_vals, "o", color = 'r', ms = 6, zorder = 3)
        plot_dict[parameter].set_yticks([])
        if i < 6:
            plot_dict[parameter].set_xticks([])
        else:
            plot_dict[parameter].set_xlabel('Time (seconds)')
        plot_dict[parameter].set_ylabel(relabel_dict[parameter], fontsize = 'small', color = 'blue')
        #plot_dict[parameter].set_zorder(plot_dict[parameter+'_2'].get_zorder()+1)
        #plot_dict[parameter].patch.set_visible(False)
        i += 1
    fig.suptitle('Test Data, Best-Fits, and Peak Times')
    return fig
