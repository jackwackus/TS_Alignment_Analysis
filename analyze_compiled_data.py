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
    Generates a figure with subplots for each instrument displaying
        test data with square wave best-fit superimposed over data and best-fit response times indicated by red dots.
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

    # Run square wave best fit analysis and generate lists of peak times for each instrument.
    # Plot data with peak times indicated. Plot probability distributions of response times.
    fig01, fig02, data_peaks_dict = ttt.plot_data_with_peaks(condensed_data_dict, pulse_log, pulse_time)

    # Close the plots because we don't need them.
    plt.close()
    plt.close()

    # Plot square wave best-fit data.
    fig = ttt.plot_best_fit_and_t0(condensed_data_dict, pulse_log, pulse_time)

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
