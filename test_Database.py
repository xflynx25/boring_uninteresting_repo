# page with tester functions to verify the database is working properly. 
# if they fail the test, we return info, else, we return Falsey (as in they didn't fail)
from operator import index
from constants import DROPBOX_PATH
import pandas as pd

def test_columns_are_of_single_type(df):
    bad_cols = []
    for name, col in df.T.iterrows():
        if len(set(col.apply(lambda x: type(x)).unique())) != 1:
            bad_cols.append(name)
    return bad_cols

def test_which_gameweeks_have_na(df, include_new_guys = True):
    if not(include_new_guys):
        starting_gws = {element:int(df.loc[df['element']==element]['gw'].to_list()[0]) for element in df['element'].unique()}
        print(list(starting_gws.items())[0])

        bad_indices = []
        for i, row in df.iterrows():
            if row['element'] in starting_gws and row['gw'] in list(range(starting_gws[row['element']], starting_gws[row['element']]+7)):
                bad_indices.append(i)
        df = df.drop(labels=bad_indices, axis=0)

    bad_gws = {}
    for gw in range(1,39):
        n = 0
        for _, row in df.loc[df['gw']==gw].iterrows():

            if row.isna().any():
                if gw not in (1,2,3,4,5,6,34,35,36,37,38):
                    print('joinwk: ', starting_gws[row['element']], '  -- ', row['gw'], ': ', row['name'], '  -- team:', row['team'])
                n += 1
        if n != 0:
            bad_gws[gw] = n
    return bad_gws

def test_column_is_all_nan(df):
    bads = []
    for col in df.columns:
        if df[col].isna().all():
            bads.append(col)
    return bads

if __name__ == '__main__':
    old_db = DROPBOX_PATH + "updated_training_dataset.csv"
    new_db = DROPBOX_PATH + "2020-21/Processed_Dataset_2021.csv"

    #for db_path in (old_db, new_db):
    #    db = pd.read_csv(db_path, index_col = 0)
    #    print("\nThis database has mixed columns: ", test_columns_are_of_single_type(db))

    for season in (1617, 1718, 1819, 2021):
        century = 20
        hypenated_season = f'{century}{str(season)[:2]}-{str(season)[2:]}'
        db_path = DROPBOX_PATH + f"Our_Datasets/{hypenated_season}/Processed_Dataset_{season}.csv"
        df = pd.read_csv(db_path, index_col=0)
        #nans = test_which_gameweeks_have_na(df, include_new_guys=False)
        nans = test_column_is_all_nan(df)
        print(f'Season {season}: {nans}')