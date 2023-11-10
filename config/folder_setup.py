import pandas as pd
def initialize_overseer_folders(folder):
    df = pd.DataFrame()
    for file in ('chips', 'deltas', 'made_moves'):
        df.to_csv(folder + f'{file}.csv')