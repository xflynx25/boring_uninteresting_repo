from constants import DROPBOX_PATH
import pandas as pd
from datetime import datetime
from general_helpers import safe_read_csv
import sys

def main():
    
    # for automation purposes, only run once per day since our choice function includes this
    if sys.argv[-1] == 'automated':
        print('smooth')
        date = datetime.utcnow().strftime('%m %d %Y')
        print(date)
        time = datetime.utcnow().strftime('%H:%M:%S') # remember, UTC
        print(time)
        
        # record all runs of script
        verifying_action = safe_read_csv(DROPBOX_PATH + 'verified.csv')
        new_row_df = pd.DataFrame([[date, time]], columns=['date', 'time'])
        print(verifying_action, new_row_df)
        verifying_action = pd.concat([verifying_action, new_row_df], axis=0, ignore_index=True)
        print('\n\n look at it now \n ', verifying_action)
        verifying_action.to_csv(DROPBOX_PATH +"verified.csv")


        # record only successes
        main_script_df = safe_read_csv(DROPBOX_PATH + 'automated_action_taken.csv')
        if main_script_df.shape[0] > 0 and main_script_df.loc[main_script_df['date']==date].shape[0] > 0:
            raise Exception("exiting so gracefully")
        else:
            new_row_df = pd.DataFrame([[date, 'yes']], columns=['date', 'success'])
            main_script_df = pd.concat([main_script_df, new_row_df], axis=0, ignore_index=True)
            
    
    # main chunk
    from Overseer import FPL_AI
    from Personalities import personalities_to_run
    for pers in personalities_to_run:
        ai = FPL_AI(**pers)
        ai.make_moves()

    # suceeded without shutting off 
    if sys.argv[-1] == 'automated':
        main_script_df.to_csv(DROPBOX_PATH + 'automated_action_taken.csv')





if __name__ == '__main__':
    main()