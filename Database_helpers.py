# 25 Helper Functions
import os
import pandas as pd
import numpy as np
from collections import Counter
from private_versions.constants import DATAHUB, LAST_GAMEWEEK, CORE_STATS, DROPBOX_PATH, NEW_STATS, BOOLEAN_STATS
from private_versions.constants import database_core as CORE_META
from private_versions.constants import database_team as TEAM_META
from general_helpers import get_columns_containing, drop_columns_containing

ROOT = DROPBOX_PATH + r"vaastav\data"
os.chdir(ROOT)

####################### HELPERS ###################################
###################################################################
def change_global_last_gameweek(gw):
    global LAST_GAMEWEEK
    LAST_GAMEWEEK = gw


'''rename columns by adding a suffix to each except for gw'''
def new_cols(old_cols, suffix):
    new_cols = []
    for col in old_cols:
        if col == 'gw':
            new_cols.append(col)
        else:
            new_cols.append(col + suffix)
    return new_cols


'''rename columns by adding a prefix to each except for gw'''
def new_cols_prefix(old_cols, prefix):
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
def prev_stats_round_to_gw(gws, final_columns, new_df, last_gameweek_stats):
    gws = gws.reset_index(drop=True)
    new_df = new_df.reset_index(drop=True)
    weeks = []
    entered_league = int(min(gws))
    for gw in range(entered_league, LAST_GAMEWEEK+1):
        matches = pd.Series()
        try_gw = gw
        stats = 'not initialized'
        '''blank control'''
        while matches.shape[0] == 0:  # keeps going in case of blank
            matches = gws[gws == try_gw]
            try_gw += 1
            if try_gw == LAST_GAMEWEEK + 2:  # prevents infinite runoff if blank last gw
                last_week_blank = last_gameweek_stats
                last_week_blank['gw'] = gw
                stats = pd.DataFrame(last_week_blank).T
                break

        '''main functionality'''
        if type(stats) == str:  # == 'not initialized':
            index = matches.index[:1]  # this covers double gw
            stats = new_df.loc[index]  # first game of gw
            stats['gw'] = gw

        stats = stats.drop('round#', axis=1)
        weeks.append(stats)

    return pd.concat(weeks, axis=0, ignore_index=True)


def gw_round_split(df):
    # add a round# column for use in this problem only
    final_columns = df.columns
    temp_df = df.copy()
    temp_df['round#'] = list(range(1, df.shape[0]+1))

    # gets last n games
    gws = df['gw']
    temp_df = temp_df.drop('gw', axis=1)

    return gws, final_columns, temp_df


def drop_duplicate_columns(df):
    return df.loc[:, ~df.columns.duplicated()]
##################################################################
###################### END HELPERS ###############################


################### ORIGINALLY IN DATABASE MAIN ############################

# Returns df with (season, team_name, team) ''last arg is an id''
# Relies that team_id are assigned alphabetically
def get_team_metaInfo(season):
    df = pd.read_csv(DATAHUB)
    df = df.loc[df['season'] == season]
    teams = df['team'].unique()
    teams = sorted(teams)

    meta = []
    for i in range(20):
        meta.append([season, teams[i], i+1])
    meta_df = pd.DataFrame(meta, columns=['season', 'team_name', 'team'])
    return meta_df

# goes from name (datahub) to id (vaastav)
def team_to_id(row, converter, opponent=False):
    if opponent:
        name = 'opponent'
    else:
        name = 'team'
    return converter.loc[converter['team_name'] == row[name]]['team'].iloc[0]

# returns df with (season, id, element_type, team)
def element_position_team(season):
    filename = season + r"\players_raw.csv"
    players = pd.read_csv(filename)
    df = players[['id', 'element_type', 'team']]
    df.columns = ['element', 'position', 'team']
    return df


def join_minimeta(row, minimeta_df):
    minimeta = minimeta_df.loc[minimeta_df['element']
                               == row['element']][['position', 'team']].iloc[0]
    df = pd.concat([row, minimeta], axis=0)
    return df

# to process teams we need to convert round# into gameweek, get this from picking
# six players that plays all the gameweeks, and taking the most frequently occuring list of gw (bcz people could be transfered)
# @return: single column dataframe 'gw' with N-weeks rows
def get_gameweeks(team, raw_players):
    all_players_one_team = raw_players.loc[raw_players['team'] == team]
    first_week = min(all_players_one_team['gw'])
    # look through first six incase any transfer action
    good_players = all_players_one_team.loc[all_players_one_team['gw']
                                            == first_week]['element'].iloc[0:6]
    possible_gws = []
    for player in good_players:
        wks = raw_players.loc[raw_players['element'] == player]['gw'].to_list()
        possible_gws.append(tuple(wks))
    gws = Counter(possible_gws).most_common()[0][0]
    return np.array(gws)

# @return gw starting at first and ending at LAST_GAMEWEEK
# with blanks, return season name element and zeroes
# with doubles, return the first occurance
def make_player_metadata_df(player_df):
    df = player_df[CORE_META]
    season = df['season'].iloc[0]
    name = df['name'].iloc[0]
    element = df['element'].iloc[0]
    team = df['team'].iloc[0]
    position = df['position'].iloc[0]

    entered_league = int(min(df['gw']))
    metas = []
    for wk in range(entered_league, LAST_GAMEWEEK+1):
        gwks = df.loc[(df['gw'] == wk)]
        if gwks.shape[0] == 0:  # no matches
            # value is just based on previous gw cost,  can't do any better
            if wk == 1:
                value = np.nan
            else: # assumes not two blanks in a row if use bottom two (OG)
                value = df.loc[df['gw']<wk]['value'].to_numpy()[-1] # should just use this 
                #val = df.loc[df['gw'] == wk-1]['value']
                #value = val[val.index[0]]
            blank = pd.Series([season, wk, name, element, team,
                               position, value, 0, 0, 0, 0, 0], index=CORE_META)
            metas.append(blank)
        else:  # normal gw or double gw
            metas.append(gwks.iloc[0, :])
    return pd.concat(metas, axis=1, ignore_index=True).T

# The difference here is opponent will be 0 if blank
# the id if single gw, and first_opp*20+second_opp if two opponents
def make_team_metadata_df(team_df):
    df = team_df[TEAM_META]
    season = df['season'].iloc[0]
    team = df['team'].iloc[0]

    entered_league = int(min(df['gw']))
    metas = []
    for wk in range(entered_league, LAST_GAMEWEEK+1):
        gwks = df.loc[(df['gw'] == wk)]
        if gwks.shape[0] == 0:  # no matches
            blank = pd.Series(
                [season, wk, team, 0, 0, 0, 0, 0, 0], index=TEAM_META)
            metas.append(blank)
        else:
            opponents = list(gwks['opponent'])
            normal = gwks.iloc[0, :]
            if len(opponents) == 1:
                opp = opponents[0]
            elif len(opponents) == 2:
                opp = opponents[0]*20 + opponents[1]
            else:
                raise Exception('More than 2 opponents this gameweek?')
            normal['opponent'] = opp
            metas.append(normal)  # could be double
    return pd.concat(metas, axis=1, ignore_index=True).T


################## END DIRECT MAIN HELPERS ###########################
#######################################################################

# gets first game of the season, this tuple form should be fastest to access
def get_day0(season):
    filename = season + '/gws/gw1.csv'
    times = pd.read_csv(filename, encoding='ISO-8859-1')['kickoff_time']
    day0 = sorted(times.unique())[0]

    root_year = int(day0[:4])
    root_month = int(day0[5:7])
    root_day = int(day0[8:10])
    return root_year, root_month, root_day

# Makes single column dataframe of 4 digit season (integer)
def season_column(season, length):
    season = int(season[2:4]+season[5:])
    season_col = [season]*length
    season_df = pd.DataFrame(season_col)
    season_df.columns = ['season']
    return season_df

# 2019-08-10T11:30:00Z
def format_kickoff(row, day0):
    root_year, root_month, root_day = day0

    row = row[0]
    year = int(row[:4])
    month = int(row[5:7])
    day = int(row[8:10])
    hour = int(row[11:13])

    def is_leap_year(year):
        if year % 400 == 0:
            return True
        elif year % 100 == 0:
            return False
        elif year % 4 == 0:
            return True
        else:
            return False
    # Check Leap Year
    leap_year = is_leap_year(root_year + 1)
    if leap_year:
        feb = 29
    else:
        feb = 28

    # month dict
    days_in_month = {
        1: 31,
        2: feb,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 31,
        10: 31,
        11: 30,
        12: 31,
    }
    # convert dates into days based on day0
    difference = 0
    if year == root_year:
        if month == root_month:
            difference = day - root_day
        else:
            difference += days_in_month[root_month] - root_day
            for m in range(root_month + 1, month):
                difference += days_in_month[m]
            difference += day
    elif year == root_year + 1:
        difference += days_in_month[root_month] - root_day
        for m in range(root_month + 1, 13):
            difference += days_in_month[m]
        for m in range(1, month):
            difference += days_in_month[m]
        difference += day
    else:
        raise Exception("Season went on for more than a year.")

    return pd.Series([difference, hour], index=['day', 'hour'])


'''returns _Ln, which is avg of previous n matches for all statistics'''
def last_games_stats(df, n):
    # returns series of the previous n gw aggregated, all 0 if first n gw's
    def row_stats(original_row, n, full_df):
        current = original_row['round#']  # current meaning current gw

        if current <= n:
            row = original_row.map(lambda x: np.nan)
            row['round#'] = current
        else:
            zero_list = [0] * full_df.shape[1]
            row = pd.Series(zero_list, index=full_df.columns)
            for prev in range(1, 1+n):
                some_week = full_df.loc[full_df['round#'] == int(current-prev)]
                series_week = some_week.iloc[0]
                row = row.add(series_week)
            row = row / n  # avg per match
            row['round#'] = current
        return row

    # get form scores by absolute game number, then convert to gw basis
    gws, final_columns, temp_df = gw_round_split(df)
    new_df = temp_df.apply(lambda x: row_stats(
        x, n, temp_df), axis=1, result_type='expand')
    # so here we should simply have scores in last _L games
    # we need to get a representation for the current gameweek in case it is a blank
    # THIS IS SUCH A HACKK
    last_played_round = new_df['round#'].iloc[-1]
    last_gw_stats = row_stats(
        pd.Series([last_played_round + 1], index=['round#']), n, temp_df)
    # now we need to convert to gw
    new_df = prev_stats_round_to_gw(gws, final_columns, new_df, last_gw_stats)

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
        games = full_df.loc[(full_df['round#'] < row['round#'])]
        if location == 'home':
            games = games.loc[(full_df['was_home'] == 1)]
        elif location == 'away':
            games = games.loc[(full_df['was_home'] == 0)]

        num_games = games.shape[0]
        if num_games == 0:
            return pd.Series([np.nan] * (full_df.shape[1]), index=full_df.columns)
        games_scaled = games / num_games
        result = games_scaled.sum(axis=0)
        # reset the round don't wanna sum that
        result['round#'] = row['round#']
        return result

    gws, final_columns, temp_df = gw_round_split(df)
    last_played_round = temp_df.iloc[-1, :]

    def get_avgs(location, suffix):
        loc_avgs = temp_df.apply(lambda x: season_avg(
            x, location, temp_df), axis=1, result_type='expand')  # gets avg of all previous home games
        loc_last = season_avg(last_played_round, location, temp_df)
        loc_avgs = prev_stats_round_to_gw(gws, final_columns, loc_avgs,
                                loc_last).drop(['was_home'], axis=1)
        loc_avgs.columns = new_cols(loc_avgs.columns, suffix)
        return loc_avgs
    
    home_avgs = get_avgs('home', '_SAH')
    away_avgs = get_avgs('away', '_SAA')
    total_avgs = get_avgs('total', '_SAT')
    return home_avgs, away_avgs, total_avgs


'''gets days rest that the team had between games'''
def days_rest(row, df):
    gw = row['gw']
    if row['opponent'] == 0:  # blank gw
        return 0
    if gw == 1:
        return np.nan

    this_game_day = row['day']

    prev_game_day = 0
    week = gw-1
    while week > 0:  # keeps going back until find a week w game i.e. blank protection
        date = df.loc[df['gw'] == week]['day'].iloc[0]
        if date != np.nan and date != 0:
            prev_game_day = date
            break
        else:
            week -= 1
    return int(np.subtract(this_game_day, prev_game_day))


'''gets days rest that the opponent had'''
def opp_days_rest(df, team, gw):
    opponent = df.loc[(df['team'] == team) & (
        df['gw'] == gw)]['opponent'].iloc[0]
    if opponent == np.nan or opponent == 0:
        return opponent
    elif opponent < 21:
        dr = df.loc[(df['team'] == opponent) & (df['gw'] == gw)]['days_rest']
        days_rest = dr[dr.index[0]]
        return days_rest
    else:  # double gw, return first
        opp = (opponent-1)//20
        dr = df.loc[(df['team'] == opp) & (df['gw'] == gw)]['days_rest']
        days_rest = dr[dr.index[0]]
        return days_rest


'''pts on next 1,3,6 ---  '_N1', '_N3', '_N6' '''
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


''' This function returns the avg opponent's 22 OG features
    for all form lengths (ex. L1, L3, L6, SAT, SALOC)
    for all forward_pred_lengths (ex. FIX1, FIX2, FIX3,...FIX6)
    
    adds rows for the opponents in the next N gws, and divides by N
    this means that you could have 3-9 opponents realistically
    
    we also add on a column stating the number of rows that are being added = num_opponents

    @param: df contains the whole raw_season_stats, gw is start gw
'''
def opponent_statistics(df, team, gw, n):
    # first get column names
    prefix = 'FIX' + str(n) + '_'
    patterns = [['_L', 'SA'] if n == 1 else ['_L', 'SAT']][0] #only want opponent saloc if next game (bcz only guarantee on home/away)
    stat_cols = list(get_columns_containing(patterns, df).columns)
    end_cols = new_cols_prefix(stat_cols + ['num_opponents'], prefix)

    # avoid going out of index with late season nan
    if gw + n > LAST_GAMEWEEK + 1:
        return pd.Series(np.nan, index=end_cols)

    num_opponents = 0
    opp_list = []
    for wk in range(gw, gw+n):
        week = df.loc[(df['gw']) == gw]  # where get stats from
        opponent = df.loc[(df['team'] == team) & (
            df['gw'] == wk)]['opponent'].iloc[0]  # where look for opp
        if opponent == 0:  # blank
            pass
        elif opponent < 21:
            stats = week.loc[week['team'] == opponent][stat_cols]
            num_opponents += 1
            opp_list.append(stats)
        else:  # double gw, return first
            first_opp = week.loc[week['team'] == (opponent-1)//20][stat_cols]
            if opponent % 20 == 0:
                second_opp = week.loc[week['team'] == 20][stat_cols]
            else:
                second_opp = week.loc[week['team'] == opponent % 20][stat_cols]
            num_opponents += 2
            opp_list.append(first_opp)
            opp_list.append(second_opp)

    # deal with only blank
    if num_opponents == 0:
        return pd.Series(0, index=end_cols)
    # not end of season & at least one single or double
    all_opponents = pd.concat(opp_list, axis=0)
    total = all_opponents.sum(axis=0, skipna=False) / \
        num_opponents  # avg scores over the weeks
    total['num_opponents'] = num_opponents
    total.index = new_cols_prefix(total.index, prefix)
    return total


# gives back list of (home,away) for each gw so far
def locations_per_week(df):
    entered_league = int(min(df['gw']))
    loc_dict = {}
    for gw in range(entered_league, LAST_GAMEWEEK+1):
        week = df.loc[(df['gw'] == gw)]
        if week.shape[0] == 0:
            loc_dict[gw] = (0, 0)
        else:
            homes = week.loc[week['was_home'] == 1].shape[0]
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
    # these return boolean lists
    SAA_columns = cols.str.contains('SAA')
    SAH_columns = cols.str.contains('SAH')
    # default column to set all to so concatenate well
    def_cols = cols[SAA_columns]

    entered_league = int(min(df['gw']))
    all_weeks = []
    for gw in range(entered_league, LAST_GAMEWEEK+1):
        row = df.loc[df['gw'] == gw]
        loki = home_away[gw]
        home = loki[0]
        away = loki[1]

        if home > 0 and away == 0:
            team = row.loc[:, SAH_columns]
        elif away > 0 and home == 0:
            team = row.loc[:, SAA_columns]
        elif home == 1 and away == 1:  # double
            uno = row.loc[:, SAA_columns]
            dos = row.loc[:, SAH_columns]
            dos.columns = uno.columns
            team = uno.add(dos, fill_value=0) / 2
        elif home == 0 and away == 0:  # blank
            team = pd.DataFrame(0, index=[0], columns=def_cols)
        team.columns = def_cols
        all_weeks.append(team)

    team_all = pd.concat(all_weeks, axis=0, ignore_index=True)
    team_all.columns = [x[:-1] + 'LOC' for x in team_all.columns]
    team_all['gw'] = list(range(entered_league, LAST_GAMEWEEK+1))\

    return team_all



'''returns df with any columns containing items in patterns to 
a percentage of the weekly top or bottom score for +/-     '''
# ex: 4, 5, -1, -2 --> .8, 1, -.5, -1
def change_columns_to_weekly_comparisons(full_df, patterns):

    df = get_columns_containing(patterns + ['season', 'gw', 'element'], full_df)
    all_weeks = []
    for season in df['season'].unique():
        for gw in df['gw'].unique():
            week_df = df.loc[(df['season'] == season) & (df['gw'] == gw)]
            meta_columns = week_df[['season', 'gw', 'element']]
            gw_df = drop_columns_containing(['season', 'gw', 'element'], week_df)
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
                        on=['season', 'gw', 'element'])
    return final_df



#return: max/avg 1-4 (8 total stats)
#we take average of player playing more than 45 minutes, but max of all
def get_opp_fantasy_points(gw, opponent, raw_players):
    answers = []
    opp = raw_players.loc[(raw_players['gw']==gw)&(raw_players['team']==opponent)]
    opp_starters = opp.loc[opp['minutes']>45]
    for pos in range(1,5):
        options = opp.loc[opp['position']==pos]['total_points']
        answer = [float(options.max()) if options.shape[0] != 0 else 2][0] #will just give the base 2pts if noone played
        answers.append(answer)
    for pos in range(1,5):
        options = opp_starters.loc[opp_starters['position']==pos]['total_points']
        answer = [float(options.mean()) if options.shape[0] != 0 else 2][0] #don't want to disadvantage based on teams w/o position
        answers.append(answer)
    return answers

# THIS STAT MEASURES HOW MUCH YOU ARE CONCEDING TO A SPECIFIC POSITION
#@return: df['gw', 'team', 'type_points_given_posx', for x=1,2,3,4 for type= max, avg]
## only considers players playing > 60 min for the avg. 
def database_make_team_fantasy_point_concessions(current_gw, raw_players):
    all_fixtures = raw_players.loc[raw_players['gw']<current_gw][['gw','team','opponent_team']].drop_duplicates()
    all_rows = []
    for _, fixture in all_fixtures.iterrows():
        gw, team, opponent = fixture

        opp_point_list = get_opp_fantasy_points(gw, opponent, raw_players)
        row = [gw, team, opponent] + opp_point_list 
        all_rows.append(row)

    col_names = ['gw','team', 'opponent'] + NEW_STATS

    final = pd.DataFrame(all_rows, columns=col_names)
    return final

# turn the many concession stats into just the one representing their position
def resolve_concessions(total):
    temp_og = get_columns_containing(['concession_pos_1'], total)
    og_column_names = get_columns_containing(['FIX'], temp_og).columns
    new_column_names = []
    for name in og_column_names:
        cutoff_index = name.index("concession") + 14 
        new_column_names.append( name[:cutoff_index] + name[cutoff_index+2:] )

    core = drop_columns_containing(['concession'], total)
    total = change_columns_to_weekly_comparisons(total, ['concession']) #big deal line here
    pos_columns = []
    for _, row in total.iterrows():
        pattern = 'concession_pos_' + str(int(row['position']))
        cols = get_columns_containing([pattern], row)
        opp_cols = get_columns_containing(['FIX'], cols)
        opp_cols.index = new_column_names
        pos_columns.append(opp_cols)
    pos_columns = pd.concat(pos_columns, axis=1).T
    final = pd.concat([core, pos_columns], axis=1)
    return final


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


#@if odds are 0 0 0 in this row, make the avg
def correct_odds_if_necessary(row, odds_df):
    sum_odds = row[['oddsW', 'oddsD', 'oddsL']].sum()
    if sum_odds <= 0:
        if sum_odds < 0:
            raise Exception("sum of odds is less than 0")
        else:
            avgs = odds_df.loc[odds_df['team']==row['team']][['oddsW', 'oddsD', 'oddsL']].mean(axis=0)
            row[['oddsW', 'oddsD', 'oddsL']] = avgs
    return row 

#Input= ['season', 'round#', 'day', 'team', 'opponent', 'was_home', 'FTgf'
# , 'FTga', 'pts', 'oddsW', 'oddsD', 'oddsL', 'Sf', 'Sa', 'STf', 'STa', 'Cf', 'Ca', 'Ff', 'Fa']
#Return= same except odds averaged for weeks with all 0s
def correct_bad_odds_team_stats(team_stats):
    odds_rows = [0,0,0]
    for i, name in enumerate(list(team_stats.columns)):
        if name == 'oddsW':
            odds_rows[0] = i
        if name == 'oddsD':
            odds_rows[1] = i
        if name == 'oddsL':
            odds_rows[2] = i

    df = team_stats[['round#', 'oddsW', 'oddsD', 'oddsL']]
    for index, row in df.iterrows():
        gw = row['round#']
        sum_odds = row[['oddsW', 'oddsD', 'oddsL']].sum() 
        if sum_odds <= 0:
            if sum_odds < 0:
                raise Exception("sum of odds is less than 0")
            else:
                avgs = df.loc[df['round#']<gw][['oddsW', 'oddsD', 'oddsL']].mean(axis=0)
                team_stats.iloc[index, odds_rows] = avgs
    return team_stats 


# add one hot, and pts_goal, pts_clean_sheet
def add_alternate_position_representations(df):
    pts_dict = {
        1: 6, 2:6, 3:5, 4:4
    }
    cs_dict = {
        1: 4, 2:4, 3:1, 4:0
    }
    new_col_names = ['is_pos_1', 'is_pos_2', 'is_pos_3', 'is_pos_4', 'pts_goal', 'pts_cs']
    new_columns = []
    for _,row in df.iterrows():
        stats = [0]*6
        pos = int(row['position'])
        stats[pos-1] = 1 
        stats[4] = pts_dict[pos]
        stats[5] = cs_dict[pos]
        new_columns.append(stats)

    new = pd.DataFrame(new_columns, columns=new_col_names)
    final_df = pd.concat([df, new], axis=1)
    return final_df