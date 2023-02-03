# 25 Helper Functions
import pandas as pd
import math
import numpy as np
from selenium import webdriver
import time
from bs4 import BeautifulSoup

import importlib 
import Requests 
importlib.reload(Requests)
from Requests import * #helper functions
from private_versions.constants import LAST_GAMEWEEK, VASTAAV_ROOT, DATAHUB, CORE_STATS, NEW_STATS, BOOLEAN_STATS,\
    MANUAL_VASTAAV_ROOT, DROPBOX_PATH, MANUALLY_REDO_WEEKS, VASTAAV_NO_RESPONSE_WEEKS, VERBOSITY
from private_versions.constants import accountant_core as CORE_META 
from private_versions.constants import accountant_team as TEAM_META
from collections import Counter
from general_helpers import difference_in_days, get_columns_containing, drop_columns_containing,\
    safe_read_csv, get_current_day





###################################################################
'''rename columns by adding a suffix to each except for gw'''
def new_cols(old_cols, suffix):
    new_cols = []
    for col in old_cols:
        if col == 'gw':
            new_cols.append(col)
        else:
            new_cols.append(col + suffix)
    return new_cols

def new_cols_prefix(old_cols, prefix):
    '''rename columns by adding a prefix to each except for gw'''
    new_cols = []
    for col in old_cols:
        if col == 'gw':
            new_cols.append(col)
        else:
            new_cols.append(prefix + col)
    return new_cols

# turns that into gw indexed, giving blank weeks all 0
# stats must include round# which is range of absolute numbers
# last_gameweek_stats: is what the last gameweek should choose, will use if blank gw
def prev_stats_round_to_gw(gws, final_columns, new_df, last_gameweek_stats_if_blank):
    gws = gws.reset_index(drop=True)
    new_df = new_df.reset_index(drop=True)
    weeks = []
    entered_league = int(min(gws))
    for gw in range(entered_league,LAST_GAMEWEEK+1):
        matches = pd.Series()
        try_gw = gw
        stats = 'not initialized'
        '''blank control'''
        while matches.shape[0] == 0: #keeps going in case of blank
            matches = gws[gws==try_gw]
            try_gw += 1
            if try_gw == LAST_GAMEWEEK + 2: #prevents infinite runoff if blank last gw
                last_week_blank = last_gameweek_stats_if_blank
                last_week_blank['gw'] = gw
                stats = pd.DataFrame(last_week_blank).T
                break

        '''main functionality'''
        if type(stats)==str: # == 'not initialized':
            index = matches.index[:1]
            stats = new_df.loc[index] #first game of gw
            stats['gw'] = gw

        stats = stats.drop('round#', axis=1)
        weeks.append(stats)
            
    #weeks.columns = final_columns
    return pd.concat(weeks, axis=0, ignore_index=True)

def gw_round_split(df):
    # add a round# column for use in this problem only
    final_columns = df.columns 
    temp_df = df.copy()
    temp_df['round#'] = list(range(1,df.shape[0]+1))

    # gets last n games
    gws = df['gw']
    temp_df = temp_df.drop('gw', axis=1)

    return gws, final_columns, temp_df

def drop_duplicate_columns(df):
    return df.loc[:,~df.columns.duplicated()]
##################################################################
###################### END HELPERS ###############################


################### ORIGINALLY IN DATABASE MAIN ############################


# goes from name (datahub) to id (vaastav)
def team_to_id(row, converter, opponent=False):
    if opponent:
        name = 'opponent'
    else:
        name = 'team'
    return converter.loc[converter['team_name']==row[name]]['team'].to_numpy()[0]
 
# returns df with (season, id, element_type, team)
def online_element_position_team(season):
    url = VASTAAV_ROOT + season + "/players_raw.csv"
    players = Requests.get_df_from_internet(url)
    df = players[['id', 'element_type', 'team']]
    df.columns = ['element', 'position', 'team']
    return df
    
def online_element_position_team404(gw_df):
    return gw_df[['element', 'position', 'team']]

def join_minimeta(row, minimeta_df):
    player = minimeta_df.loc[minimeta_df['element']==row['element']]
    minimeta = player[['position', 'team']].iloc[0]
    df = pd.concat([row, minimeta], axis=0)
    return df

# to process teams we need to convert round# into gameweek, get this from picking 
# six players that plays all the gameweeks, and taking the most frequently occuring list of gw (bcz people could be transfered)
# @return: single column dataframe 'gw' with N-weeks rows 
def get_gameweeks(team, raw_players):
    all_players_one_team = raw_players.loc[raw_players['team']==team]
    first_week = min(all_players_one_team['gw'])
    good_players = all_players_one_team.loc[all_players_one_team['gw']==first_week]['element'].iloc[0:6] #look through first six incase any transfer action
    possible_gws = []
    for player in good_players:
            wks = raw_players.loc[raw_players['element']==player]['gw'].to_list()
            possible_gws.append(tuple(wks))
    gws = Counter(possible_gws).most_common()[0][0]
    return np.array(gws)


# @return gw starting at first and ending at LAST_GAMEWEEK
# with blanks, return season name element and zeroes
# with doubles, return the first occurance
def online_make_player_metadata_df(player_df):
    df = player_df[CORE_META]
    element = df['element'].to_numpy()[0]
    team = df['team'].to_numpy()[0]
    position = df['position'].to_numpy()[0]

    entered_league = int(min(df['gw']))
    metas = []
    for wk in range(entered_league, LAST_GAMEWEEK+1):
        gwks = df.loc[(df['gw']==wk)]
        if gwks.shape[0] == 0: #no matches
            # value is just based on previous gw cost,  can't do any better
            if wk == 1:
                value = np.nan
            else:
                value = df.loc[df['gw']<wk]['value'].to_numpy()[-1]
            blank = pd.Series([wk, element, team, position, value, 0,0,0,0], index=CORE_META)
            metas.append(blank)
        else: #normal gw or double gw
            metas.append(gwks.iloc[0,:])
    return pd.concat(metas, axis=1, ignore_index=True).T



#  The difference here is opponent will be 0 if blank
# the id if single gw, and first_opp*20+second_opp if two opponents
# ['gw', 'team', 'opponent', 'day', 'hour', 'was_home', 'oddsW', 'oddsD', 'oddsL']
def online_make_team_metadata_df(team_df):
    df = team_df[TEAM_META]
    team = df['team'].to_numpy()[0]

    entered_league = 1
    metas = []
    for wk in range(entered_league, LAST_GAMEWEEK+1):
        gwks = df.loc[(df['gw']==wk)]
        if gwks.shape[0] == 0: #no matches
            blank = pd.Series([wk, team, 0, 0, 0, 0, 0,0,0], index=TEAM_META)
            metas.append(blank)
        else:
            opponents = list(gwks['opponent'])
            normal = gwks.iloc[0,:].copy() # to avoid warnings
            if len(opponents) == 1:
                opp = opponents[0]
            elif len(opponents) == 2:
                opp = opponents[0]*20 + opponents[1]
            elif len(opponents) == 3:
                opp = opponents[0]*20 + opponents[1] # FOR NOW SINCE NEVER BEFORE TRIPLE WE JUST PICK DOUBLE
            else: 
                raise Exception('More than 3 opponents this gameweek?')
            normal.loc['opponent'] = opp
            metas.append(normal) #could be double
    return pd.concat(metas, axis=1, ignore_index=True).T


################## END DIRECT MAIN HELPERS ###########################
#######################################################################


# gets first game of the season, this tuple form should be fastest to access
def get_day0(season):
    url = season + '/gws/gw1.csv'
    times = get_df_from_internet(url)['kickoff_time']
    day0 = sorted(times.unique())[0]
    
    root_year = int(day0[:4])
    root_month = int(day0[5:7])
    root_day = int(day0[8:10])
    return root_year, root_month, root_day

# @param: from the patch db a time in format __ / __ / ____ where day and month sometimes only 1 digit
# @return: formatted like the above, yyyy-mm-dd
'''TURNS OUT WE DIDN'T NEED THIS'''
def format_patch_time(time):
    day, month, year = time.split('/')
    if len(month) == 1:
        month = '0'+month
    if len(day) == 1:
        day = '0' + day
    return year + '-' + month + '-' + day


# Makes single column dataframe of 4 digit season (integer)
def full_column(value, length, name):
    col = [[value]]*length
    df = pd.DataFrame(col)
    df.columns = [name]
    return df

#2019-08-10T11:30:00Z
def format_kickoff(row, day0):
    root_year, root_month, root_day = day0

    row = row[0]
    year = int(row[:4])
    month = int(row[5:7])
    day = int(row[8:10])
    hour = int(row[11:13])

    if abs(year - root_year) > 1: 
        raise Exception("Season went on for more than a year.")
    difference = difference_in_days(day0, [year, month, day])
    
    return pd.Series([difference, hour], index=['day', 'hour'])



'''returns _Ln, which is avg of previous n matches for all statistics'''
def last_games_stats(df, n):
    # returns series of the previous n gw aggregated, all 0 if first n gw's
    def row_stats(original_row, n, full_df):
        current = original_row['round#'] #current meaning current gw

        zero_list = [0] * full_df.shape[1]
        row = pd.Series(zero_list, index = full_df.columns)
        
        if current <= n:
            row = row.map(lambda x: np.nan)
        else:
            row = pd.Series(zero_list, index = full_df.columns)
            for prev in range(1, 1+n):
                some_week = full_df.loc[full_df['round#']==int(current-prev)]
                series_week = some_week.iloc[0]
                row = row.add(series_week)
            row = row / n #avg per match
        
        row['round#'] = current
        return row

    # get form scores by absolute game number, then convert to gw basis
    gws, final_columns, temp_df = gw_round_split(df) 
    new_df = temp_df.apply(lambda x: row_stats(x, n, temp_df), axis=1, result_type='expand')
    # so here we should simply have scores in last _L games 
    # we need to get a representation for the current gameweek in case it is a blank
    # THIS IS SUCH A HACKK 
    last_played_round = new_df['round#'].iloc[-1]
    last_gw_stats_if_blank = row_stats(pd.Series([last_played_round + 1], index=['round#']), n, temp_df)
    # now we need to convert to gw 
    new_df = prev_stats_round_to_gw(gws, final_columns, new_df, last_gw_stats_if_blank)

    # renaming columns
    suffix = '_L' + str(n)
    new_columns = new_cols(new_df.columns, suffix)
    new_df.columns = new_columns

    return new_df


'''returns _SAH and _SAA, aggregating all previous gw and dividing by number of previous gameweeks
    updated to also return _SAT, season average total'''
def season_averages(df):

    # location should be 'home', 'away', 'total'
    def season_avg(row, location, full_df):
        #games = full_df.loc[(full_df['round#'] < row['round#'])]
        #print(games)
        if location == 'home':
            games = full_df.loc[(full_df['was_home']==1) | (full_df['was_home']==True)]
        elif location == 'away':
            games = full_df.loc[(full_df['was_home']==0) | (full_df['was_home']==False)]
        else:
            games = full_df
        games = games.loc[(games['round#'] < row['round#'])]
        

        num_games = games.shape[0]
        if num_games == 0:
            return pd.Series([np.nan] * (full_df.shape[1]), index=full_df.columns)
        games_scaled = games / num_games
        result = games_scaled.sum(axis=0)
        result['round#'] = row['round#'] #reset the round don't wanna sum that 
        return result
        
    gws, final_columns, temp_df = gw_round_split(df)

    last_played_round = temp_df.iloc[-1,:]

    #for each, get season avg, get last_gw buffer, transfer to gw coordinates, and apply suffix
    home_avgs = temp_df.apply(lambda x: season_avg(x, 'home', temp_df), axis=1, result_type='expand') #gets avg of all previous home games
    home_last = season_avg(last_played_round, 'home', temp_df)
    home_avgs = prev_stats_round_to_gw(gws, final_columns, home_avgs, home_last).drop(['was_home'], axis=1)
    home_avgs.columns = new_cols(home_avgs.columns, '_SAH')

    away_avgs = temp_df.apply(lambda x: season_avg(x, 'away', temp_df), axis=1, result_type='expand') #gets avg of all previous away games
    away_last = season_avg(last_played_round, 'away', temp_df)
    away_avgs = prev_stats_round_to_gw(gws, final_columns, away_avgs, away_last).drop(['was_home'], axis=1)
    away_avgs.columns = new_cols(away_avgs.columns, '_SAA')

    total_avgs = temp_df.apply(lambda x: season_avg(x, 'total', temp_df), axis=1, result_type='expand') #gets avg of all previous games
    total_last = season_avg(last_played_round, 'total', temp_df)
    total_avgs = prev_stats_round_to_gw(gws, final_columns, total_avgs, total_last).drop(['was_home'], axis=1)
    total_avgs.columns = new_cols(total_avgs.columns, '_SAT')

    return home_avgs, away_avgs, total_avgs



def days_rest(row, df):
    gw = row['gw']
    if row['opponent'] == 0: #blank gw 
        return 0
    if gw == 1:
        return np.nan

    this_game_day = row['day']

    prev_game_day = 0
    week = gw-1 
    while week > 0: #keeps going back until find a week w game i.e. blank protection
        date =  df.loc[df['gw'] == week]['day'].to_numpy()[0]
        if date != np.nan and date != 0:
            prev_game_day = date
            break
        else:
            week -= 1
    return int(np.subtract(this_game_day, prev_game_day))


'''gets days rest that the opponent had'''
def online_opp_days_rest(df, team, gw, fixtures_df):
    opponent = fixtures_df.loc[(fixtures_df['team']==team)&(fixtures_df['gw']==gw)]['opponent']
    
    if opponent.shape[0] == 0: 
        return 0

    opponent = opponent.to_numpy()[0] 
    if opponent == np.nan:
        return np.nan 
    else: #will work for doubles because we just want the first one 
        dr = df.loc[(df['team']==opponent) & (df['gw']==gw)]['days_rest']
        days_rest = dr[dr.index[0]]
        return days_rest

''' This function returns the avg opponent's 22 OG features
    for all form lengths (ex. L1, L3, L6, SAT, SALOC)
    for all forward_pred_lengths (ex. FIX1, FIX2, FIX3,...FIX6)
    
    adds rows for the opponents in the next N gws, and divides by N
    this means that you could have 3-9 opponents realistically
    
    we also add on a column stating the number of rows that are being added = num_opponents

    @param: df contains the whole raw_season_stats, gw is start gw
'''
def online_opponent_statistics(df, team, gw, n, fixtures_df):
    # first get column names 
    prefix = 'FIX' + str(n) + '_'
    patterns = [['_L', 'SA'] if n == 1 else ['_L', 'SAT']][0] #only want opponent saloc if next game (bcz only guarantee on home/away)
    stat_cols = list(get_columns_containing(patterns, df).columns)
    end_cols = new_cols_prefix(stat_cols + ['num_opponents'], prefix)

    # avoid going out of index with late season nan
    if gw + n > 39:
        return pd.Series(np.nan, index=end_cols)

    num_opponents = 0
    opp_list = []
    for wk in range(gw, gw+n):
        week = df.loc[(df['gw'])==gw] #where get stats from
        opponent = fixtures_df.loc[(fixtures_df['team']==team)&(fixtures_df['gw']==wk)]['opponent']#THIS IS THE DIFF LINE FROM MAIN_DATABASE

        if opponent.shape[0] == 0: #blank
            pass  
        elif opponent.shape[0] == 1:
            opp = opponent.to_numpy()[0]
            stats = week.loc[week['team']==opp][stat_cols]
            num_opponents += 1
            #print('stats is ', stats)
            opp_list.append(stats)
        else: #double gw, use both
            new_opps_stats = [week.loc[week['team']==opp][stat_cols] for opp in opponent.to_numpy()]
            opp_list = opp_list + new_opps_stats
            num_opponents += 2
            
    # deal with only blank 
    if len(opp_list) == 0:
        return pd.Series(0, index=end_cols)
    # not end of season & at least one single or double
    all_opponents = pd.concat(opp_list, axis=0)
    total = all_opponents.sum(axis=0, skipna=False) / num_opponents #avg scores over the weeks
    total['num_opponents'] = num_opponents
    total.index = new_cols_prefix(total.index, prefix)
    return total


#gives back list of (home,away) for each gw so far
def locations_per_week(df, team_version=False):
    entered_league = int(min(df['gw']))
    if team_version:
        entered_league = 1
    loc_dict = {}
    for gw in range(entered_league, LAST_GAMEWEEK+1):
        week = df.loc[(df['gw']==gw)]
        if week.shape[0] == 0:
            loc_dict[gw] = (0,0) 
        else:
            homes = week.loc[week['was_home']==1].shape[0]
            aways = week.shape[0] - homes
            loc_dict[gw] = (homes, aways)
    return loc_dict


'''
if a team is home, returns only the SAH endings for self
and return only SAA for all stats that start with FIX1

we only choose to keep the location based stats for the next opponent
because we don't know the home/away of the other teams, so there is not
likely to be much of substance there that won't be captured in the full average
There could be in a very nonlinear sense some sort of if theres a high difference
between the opponents away form and current home form, might effect things, 
but for this model this area should be sacrificed to keep the features down
'''
# @params: home_away is dict mapping gw to (home, away) counts 
# @return: saloc columns, also num_opponents, and home (1,0.5,0)
def location_specific_columns(df, home_away):
    cols = df.columns
    #these return boolean lists
    SAA_columns = cols.str.contains('SAA')
    SAH_columns = cols.str.contains('SAH')
    #default column to set all to so concatenate well
    def_cols = cols[SAA_columns]

    entered_league = int(min(df['gw']))
    all_weeks = []
    for gw in range(entered_league, LAST_GAMEWEEK+1):
        row = df.loc[df['gw']==gw]
        loki = home_away[gw]
        home = loki[0] 
        away = loki[1] 
        
        if home>0 and away==0:
            team = row.loc[:, SAH_columns]
        elif away>0 and home==0:
            team = row.loc[:, SAA_columns]
        elif home>=1 and away>=1: #double or triples
            uno = away*row.loc[:, SAA_columns]
            dos = home*row.loc[:, SAH_columns]
            dos.columns = uno.columns
            team = uno.add(dos, fill_value=0) / (home + away)
        elif home==0 and away==0: #blank
            team = pd.DataFrame(0, index=[0],columns=def_cols)
        team.columns = def_cols
        all_weeks.append(team)
    
    team_all = pd.concat(all_weeks, axis=0, ignore_index=True)
    team_all.columns = [x[:-1] + 'LOC' for x in team_all.columns]
    team_all['gw'] = list(range(entered_league, LAST_GAMEWEEK+1))\

    return team_all

    

#@return gw  ppm column
def ppm_column(player_df):
    player_df = player_df.sort_values('gw').fillna(0)
    entered_league = int(min(player_df['gw']))
    all_weeks = []
    for gw in range(entered_league, LAST_GAMEWEEK+1):
        df = player_df.loc[player_df['gw']<=gw]
        total_pts, total_mins = df['total_points_L1'].sum(), df['minutes_L1'].sum()
        ppm = [0 if total_mins == 0 else float( df['total_points_L1'].sum() / df['minutes_L1'].sum() )][0]
        all_weeks.append([ppm])
    
    player_new = pd.DataFrame(all_weeks, columns=['ppmin'])
    player_new['gw'] = list(range(entered_league, LAST_GAMEWEEK+1))
    return player_new



# changes this to max 1 min -1
def online_change_columns_to_weekly_comparisons(full_df, patterns):

    df = get_columns_containing(patterns + ['gw', 'element'], full_df)
    all_weeks = []
    for gw in df['gw'].unique():
        week_df = df.loc[df['gw'] == gw]
        meta_columns = week_df[['gw', 'element']]
        gw_df = drop_columns_containing(['gw', 'element'], week_df)
        maximums = gw_df.max(axis=0)
        minimums = gw_df.min(axis=0)
        maximums.replace(to_replace=0, value=2**-10, inplace=True)
        minimums.replace(to_replace=0, value=2**-10, inplace=True)

        signs = gw_df.copy()
        signs = signs.where(signs>=0, other=-1)
        signs = signs.where(signs<=0, other=1)
        #signs[signs > 0] = 1
        #signs[signs < 0] = -1

        positives = (signs + 1) / 2 * gw_df / maximums #gw_df.div(maximums, axis=1)
        negatives = (signs - 1) / 2 * gw_df / minimums #gw_df.div(minimums, axis=1)
        full = pd.concat([meta_columns, positives+negatives], axis=1)
        all_weeks.append(full)

    new_df = drop_columns_containing(patterns, full_df)
    new_columns = pd.concat(all_weeks, axis=0, ignore_index=True)
    final_df = pd.merge(new_df, new_columns, how='left',
                        on=['gw', 'element'])
    return final_df

# applied on df, create stats for whatever is specified by constants.boolean_stats
def add_boolean_stats(row):
    minutes = row['minutes']
    points = row['total_points']

    decent_week = [1 if points >= 5 else 0][0]
    good_week = [1 if points >= 8 else 0][0]
    great_week = [1 if points >= 12 else 0][0]
    lasted_long = [1 if minutes >= 60 else 0][0]
    substitute = [1 if minutes <= 45 and minutes > 0 else 0][0]
    absent = [1 if minutes == 0 else 0][0]

    df = pd.Series([decent_week, good_week, great_week, lasted_long, substitute, absent], index=BOOLEAN_STATS)
    return df




""" Takes in row of dataframe, makes some switches to get data we need"""
# id --> element
# now_cost --> value
# if in home_teams --> was_home
# transfers_in_event, transfers_out_event --> transfers_in, transfers_out
# transfers_in - transfers_out --> transfers_balance
# gw --> round 
# selected_by_percent --> selected
def raw_data_fixup(x, gw, home_teams):
    was_home = [1 if x['team'] in home_teams else 0][0]
    element = x['id']
    value = x['now_cost']
    transfers_in = x['transfers_in_event']
    transfers_out = x['transfers_out_event']
    this_round = gw
    selected = x['selected_by_percent']
    transfers_balance = transfers_in - transfers_out
    position = x['element_type']

    row = drop_columns_containing(['transfers'], x)
    new_cols = [this_round, was_home, element, value, transfers_in, transfers_out, transfers_balance, selected, position]
    names = ['round', 'was_home', 'element', 'value', 'transfers_in', 'transfers_out', 'transfers_balance', 'selected','position']
    additions = pd.Series(new_cols, index=names)

    return additions


# we want to record scores and minutes, there will just be nans if this is past our last gameweek
# this is fine since we don't use nan rows for predictions anymore
def prediction_stats(df, n, feature):
    def points(gw, n, full_df):
        weeks_allowed = full_df.loc[(
            full_df['gw'] >= gw) & (full_df['gw'] < gw+n)]
        if gw + (n - 1) > LAST_GAMEWEEK:
            return np.nan
        elif weeks_allowed.shape[0] == 0:  # blank
            return 0
        else:
            return np.sum(weeks_allowed[feature])

    df = df[['gw', feature]]
    entered_league = int(min(df['gw']))
    scores = []
    for gw in range(entered_league, LAST_GAMEWEEK + 1):
        scores.append([gw,  points(gw, n, df)])
    new_df = pd.DataFrame(scores, columns=df.columns)

    '''renaming columns'''
    suffix = '_N' + str(n)
    new_columns = new_cols(new_df.columns, suffix)
    new_df.columns = new_columns
    return new_df