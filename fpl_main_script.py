import pandas as pd
from datetime import datetime
from general_helpers import safe_read_csv, get_year_month_day_string, get_year_month_day_hour_minute_second
import sys
import time

def main():    
    try:
        # for automation purposes, only run once per day since our choice function includes this
        if sys.argv[-1] == 'automated':
            from private_versions.constants import DROPBOX_PATH
            date_info = get_year_month_day_hour_minute_second()
            date, time = date_info[:3], date_info[3:]
            print('Local Time: ', date[0], ' ', date[1], ' ', date[2], ' ', end=",, ")
            print(time[0], ' ', time[1], ' ', time[2])

            # above for printing to stdout, here for saving to csv
            # seperation is historic, not logical
            date = get_year_month_day_string()
            
            # record all runs of script
            verifying_action = safe_read_csv(DROPBOX_PATH + 'verified.csv')
            new_row_df = pd.DataFrame([[date, time]], columns=['date', 'time'])
            verifying_action = pd.concat([verifying_action, new_row_df], axis=0, ignore_index=True)
            verifying_action.to_csv(DROPBOX_PATH +"verified.csv")

            # record only successes
            main_script_df = safe_read_csv(DROPBOX_PATH + 'automated_action_taken.csv')
            if main_script_df.shape[0] > 0 and main_script_df.loc[main_script_df['date']==date].shape[0] > 0:
                print("exiting so gracefully")
                return 
            else:
                print("Actually Computing")
                new_row_df = pd.DataFrame([[date, 'yes']], columns=['date', 'success'])
                main_script_df = pd.concat([main_script_df, new_row_df], axis=0, ignore_index=True)
                
        # often running on a different computer
        else:
            print('hello backup computer')
            print('still yet to figure out how to properly do this')
            print('what we are aiming to do is allow the variable to be changed based on a sysargv arg')
            print('but the variable we are importing in different files')
            #import private_versions.malleable_constants 
            #private_versions.malleable_constants.change_computer_username(private_versions.malleable_constants.BACKUP_COMPUTER_WHO_RUNS_MAIN_SCRIPT)
            #private_versions.malleable_constants.change_c_entry(private_versions.malleable_constants.BACKUP_COMPUTER_OS_RUNS_MAIN_SCRIPT)
            #print(private_versions.malleable_constants.C_ENTRY, private_versions.malleable_constants.COMPUTER_USERNAME)
            
        
        # main chunk
        from Overseer import FPL_AI
        from private_versions.Personalities import personalities_to_run
        for pers in personalities_to_run:
            ai = FPL_AI(**pers)
            ai.make_moves()

        # suceeded without shutting off 
        if sys.argv[-1] == 'automated':
            main_script_df.to_csv(DROPBOX_PATH + 'automated_action_taken.csv')

    except Exception as e:
        print('Hit Exception: ', e)





if __name__ == '__main__':
    start = time.time()
    main()
    print('RunTime = ', round((time.time() - start)/60, 2), ' minutes')