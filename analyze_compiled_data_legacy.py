"""
Author:	Jack Connor
Date Created: 8/13/2021
"""

import os
import TS_Test_Tools as ttt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

# Main function calls routines.
def main():
    """
    Runs square wave best-fit response time analysis on timestamp test data compiled by Mobile Data Compiler.
    Uses command line arguments to ingest test name and differentiate between sub-tests.
        Test names are dates formatted YYYYMMDD. Sub-test pulse log suffixes can be input using the 'subtest_extension' argument.
        Pulse logs must be stored in the following folder within the working directory: data/{test_name}/Pulse_Log.
            Pulse logs are named {test_name}_log{subtest_extension}.csv
        Data is read from local Mobile Data Compiler directory, identified by test_name.
        The names of instrument parameters that will be analyzed are read from the "Parameters List.txt" file in the working directory. 
    Generates the following figures:
        1) figure with subplots illustrating timestamp test data and peak times for all instruments,
        2) figure with subplots illustrating probability distributions of response times for all instruments,
        3) figure with subplots containing data with square wave best-fit superimposed and best-fit response times indicated by red dots,
        4) figure with subplots containing data with square wave best-fit superimposed and best-fit response times indicated by red dots,
            peak maxima indicated by black crosses, and 90% maxima indicated by green crosses.
    Writes a csv file listing the response times (by square wave best-fit) for all instruments.
        Writes file in the following folder within the working directory: data/{test_name}.
    """
    #NOTE all TS_Test_Tools (ttt) functions are contained in the TS_Test_Tools.py library.

    import argparse

    # Structure and process command line arguments.
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--username', type=str, help='BAAQMD username [used to set file paths]', required=True)
    parser.add_argument('-n', '--test_name', type=str, help='Test Name (YYYYMMDD)', required=True)
    parser.add_argument('-e', '--subtest_extension', type=str, help='Subtest Name Extension', default='')
    parser.add_argument('-t', '--pulse_time', type=int, help='Pulse Time [half of pulse period] (seconds)', default=60)
    args = parser.parse_args()

    test_name = args.test_name
    ext = args.subtest_extension
    pulse_time = args.pulse_time
    user = args.username

    # Establish directory path for datafiles. 
    working_dir = os.getcwd()
    data_dir = f'C:\\Users\\{user}\\OneDrive - Bay Area Air Quality Management District\\python_apps\\Mobile Data Compiler'

    # The following function is not called in the default implementation of this script. It is used to log click locations on plots.
    def log_click(event):
        """
        Ingests click events from matplotlib plots and logs the data to a text file.
        Args:
            event (matplotlib.backend_bases.Event): object representing a clicking event on a matplotlib plot, carrying corresponding positional data
        Returns:
            writes a row detailing click data to a file in the data/test_name folder of the working directory
        """
        writeFile = f'{working_dir}\\data\\{test_name}\\{ext}_click_log.txt'
        row = f'{event.inaxes},{event.xdata}\n'
        with open(writeFile, 'a') as f:
            f.write(row)

    # Define a dateparser for interpreting date strings in input files.
    dateparser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    # Define a file name for writing response time data. 
    write_file = f'{working_dir}\\data\\{test_name}\\{ext}_peak_times.csv'

    # Define a file name to read test data from.
    data_path = f'{data_dir}\\{test_name}\\Results\\Complete.csv'
    compiled_df = pd.read_csv(data_path, parse_dates = ['DateTime'], date_parser = dateparser, low_memory = False) 
    print('Data Read.\n')

    # Read and process list of instrument parameters.
    parameter_list = ttt.read_parameter_list()
    data_dict = ttt.convert_compiled_df_to_data_dic(compiled_df, parameter_list)

    # Read pulse log.
    pulse_log = ttt.ingest_pulse_log(test_name, ext)
    # Cut instrument data to time period specified by pulse log.
    condensed_data_dict = ttt.generate_condensed_data(data_dict, pulse_log)
    print('Data condensed to pulse period.\n')

    # Run analysis and make plots.

    # Plot data and nothing else. Skipped in default implementation.
    #fig1 = ttt.plot_condensed_data(condensed_data_dict)

    # Run square wave best fit analysis and generate lists of peak times for each instrument.
    # Plot data with peak times indicated. Plot probability distributions of response times.
    fig2, fig3, data_peaks_dict = ttt.plot_data_with_peaks(condensed_data_dict, pulse_log, pulse_time)

    # Plot data for manual peak picking (by clicking). Skipped in default implementation.
    #fig4 = ttt.plot_data_for_clicks(condensed_data_dict)
    #cid = fig4.canvas.mpl_connect('button_press_event', log_click)
    
    # Plot square wave best-fit data.
    fig5 = ttt.plot_best_fit_and_t0(condensed_data_dict, pulse_log, pulse_time)
    # Plot square wave best-fit and 90% peak maxima data.
    fig6 = ttt.plot_90perc_peaks_2(condensed_data_dict, pulse_log, pulse_time)

    # Print and write response time data.
    data_peaks_df = pd.DataFrame(data_peaks_dict) 
    response_time_df = ttt.create_response_times_table(data_peaks_df, pulse_log)
    print('Square Wave Method Response Times')
    print(response_time_df)
    response_time_df.to_csv(write_file, index = False)
    print(f'\nResponse times table written to {write_file}.')

    # Show plots.
    plt.show()

# Run main function when this script is main module.
if __name__ == '__main__':
    main()
