################# SUMMARY #################
# Maintains 3 Databases : Odds, Player, Team
# Also creates 2 df for Overseer: Health & Names
#
# Main Functions:
#  Create Health/Name df (2)
#  Update FixtureList 
#  Update Odds
#  Update previous/current for Player/Team (4) 
#  Synthesis of the above to create weekly data
#  Post-Season storage of the yearly data
#  Update db on specific user situation chips/delta (2)
#  Logging activity databases
###########################################
import time
from datetime import datetime
import importlib 
import Accountant_helpers 
importlib.reload(Accountant_helpers)
from Accountant_helpers import * #helper functions

PLAYER_DB = DROPBOX_PATH + r'\player_raw.csv'
TEAM_DB = DROPBOX_PATH + r'\team_raw.csv'
ODDS_DB = DROPBOX_PATH + r'\odds.csv'
PLAYER_HEALTH_DB = DROPBOX_PATH + r'\player_health.csv'
BACKUP_ODDS = DROPBOX_PATH + r"\2020_odds_online.csv"

def redefine_globals(prefix):
    global PLAYER_DB, TEAM_DB, ODDS_DB, PLAYER_HEALTH_DB, BACKUP_ODDS
    PLAYER_DB = prefix + r'\player_raw.csv'
    TEAM_DB = prefix + r'\team_raw.csv'
    ODDS_DB = prefix + r'\odds.csv'
    PLAYER_HEALTH_DB = prefix + r'\player_health.csv'
    BACKUP_ODDS = prefix + r"\2020_odds_online.csv"

# returns df with ids and names so we can understand what players they looking at
def make_name_df():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = proper_request("GET", url, headers=None)
    players_raw = pd.DataFrame(response.json()['elements'])
    name_df = players_raw[['id','web_name']]
    name_df.columns = ['element','name']
    return name_df
    
# @return: df w/ a,d,i,u statuses
def make_health_df():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = proper_request("GET", url, headers=None)
    players_raw = pd.DataFrame(response.json()['elements'])
    health_df = players_raw[['id','status']]
    health_df.columns = ['element','status']
    return health_df

# @return: df w/ a,d,i,u statuses
def make_pseudohealth_df_offline(elements):
    all = []
    for elem in elements:
        all.append([elem, 'a'])
    health_df = pd.concat(all, axis=0, ignore_index=True)
    health_df.columns = ['element','status']
    return health_df

''' # RUNTIME # :: Near Instantaneous
@param: '2020-21'
@return: df with gw, team, opponent, was_home, day, hour, kickoff_time
        where blanks are moved to gw38 till rescheduled
        also returns the current_gw we are on
'''
def make_fixtures_df(season, ignore_gwks=[]):
    relevant = ['event', 'team_h', 'team_a', 'kickoff_time']
    url = 'https://fantasy.premierleague.com/api/fixtures/'
    response = proper_request("GET", url, headers=None)
    df_raw = pd.DataFrame(response.json())
    df = df_raw[relevant]
    df.columns = ['gw'] + relevant[1:]
    last_date = df.loc[df['gw']==38]['kickoff_time'].iloc[-1] #for any tbd

    fixtures = []
    for row in df.iterrows():
        row = row[1]
        if math.isnan(row['gw']): #'schedule' for last day of season 
            row['gw'] = 38
            row['kickoff_time'] = last_date
        team1 = row[row.index]
        team1.index = ['gw', 'team', 'opponent', 'kickoff_time']
        team1['was_home'] = 1
        team2 = row[row.index]
        team2.index = ['gw', 'opponent', 'team', 'kickoff_time']
        team2['was_home'] = 0
        fixtures.append(pd.concat([team1, team2], axis=1, ignore_index=True, sort=True).T)
    fixtures_df = pd.concat(fixtures, axis=0, ignore_index=True)

     # day and time columns
    day0 = sorted(df_raw['kickoff_time'].dropna().unique())[0]
    root_year = int(day0[:4])
    root_month = int(day0[5:7])
    root_day = int(day0[8:10])
    day0 = root_year, root_month, root_day

    day_time = pd.DataFrame(fixtures_df['kickoff_time']).apply(lambda x: format_kickoff(x, day0), axis=1, result_type='expand')
    day_time.columns = ['day', 'hour']
    final_df =  pd.concat([fixtures_df, day_time], axis=1, sort=True)

    # Adjust if we don't want to plan with a week in mind
    final_df = hide_fixtures(final_df, ignore_gwks)

    try:
        current_gw = int( min( df_raw.loc[df_raw['started']==False]['event'] ) )
    except: #just for testing after season ends, come back and delete
        current_gw = 38
    return final_df, current_gw

''' 2 api calls to get all available odds for the league, adds to the csv'''
# uses additional api calls if the big call doesn't take care of all those in fixtures_df before current_gw
def update_odds_df(fixtures_df, current_gw, patch=False):
    SEASON = 2020
    premier_league = get_premier_league_id(SEASON)
    week_odds = get_bet365_odds(premier_league)

    available_odds = []
    for match in week_odds.items():
        group = [ match[0],match[1][0],match[1][1],match[1][2] ]
        available_odds.append(pd.Series(group, index=['fixture_id','oddsH', 'oddsD', 'oddsA']))
    week = pd.concat(available_odds, axis=1, ignore_index=True).T

    odds_df =  pd.read_csv(ODDS_DB, index_col=0)
    try:
        odds_df = odds_df.loc[~odds_df['fixture_id'].isin(week['fixture_id'])] #replaces already entered
    except KeyError:
        pass #just the first time opening it 
    final_odds = pd.concat([odds_df, week], axis=0, sort=True).reset_index(drop=True)

    '''look at ones this week in case general func didn't get them'''
    first_patch = False
    if fixtures_df.loc[fixtures_df['gw']<=current_gw].shape[0] // 2 > final_odds.shape[0]: #not all the odds , first term has both representations of the week
        first_patch = True
    if first_patch:
        clutch_odds = individual_game_odds(premier_league, final_odds, fixtures_df, current_gw) #from helpers
        final_odds = pd.concat([final_odds, clutch_odds], axis=0, sort=True).reset_index(drop=True)
    
    '''do a soft check to see if were missing things, this might not catch if we have recorded doubles" odds'''
    if fixtures_df.loc[fixtures_df['gw']<=current_gw].shape[0] // 2 > final_odds.shape[0]: #not all the odds , first term has both representations of the week
        patch = True
    if patch:
        #clutch_odds = individual_game_odds(premier_league, final_odds, fixtures_df, current_gw, what_weeks='upto')
        #final_odds = pd.concat([final_odds, clutch_odds], axis=0, sort=True).reset_index(drop=True)
        relevant = ['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
        database = pd.read_csv(BACKUP_ODDS, index_col=0)[relevant]
        final_odds = patch_odds(final_odds, database, fixtures_df, current_gw)        
    
    final_odds.to_csv(ODDS_DB)

# Update Player
### Vaastav Gws, (redundancy here if miss week it will recover)
### 
### We only save the weeks that vaastav has - He will almost always have 
###   the previous week up before the deadline. However, if he doesn't, 
###   we get the data directly from the website and process it ourselves. 
###
### If something fails, one can manually input the weeks to be fixed in constants
###
### Top line was for previous season different representation for vaastav
###
### @return: the full prev_week player df which includes those weeks no vastaav data available
def update_player_previous(season, current_gw):
    '''
    position_dict = {
        'GK': 1, 'DEF':2,'MID':3,'FWD':4
    }
    team_dict = teamname_to_id_converter(season)
    '''

    if current_gw == 1:
        return
    players = safely_get_database(PLAYER_DB, current_gw)
    try:
        last_active_gw = max(players['gw'])
        players = players.loc[players['gw']<last_active_gw] #remove last weeks stuff
        gws = players['gw'].unique() 
        need_filling = missing_numbers(gws, current_gw) #set 
        for x in MANUALLY_REDO_WEEKS:
            need_filling.add(x)
    except KeyError: #first occurance
        need_filling = [x+1 for x in range(current_gw)][:-1]
    
    earliest_replacement = min(need_filling)
    players = players.loc[players['gw']<earliest_replacement] #remove all weeks after earliest week needs filling
    


    for gw in range(earliest_replacement, current_gw):
        player_gw = online_raw_player_gw(season, gw)
        if gw in VASTAAV_NO_RESPONSE_WEEKS: 
            without_transfers = drop_columns_containing(['transfers'], player_gw)
            previous_week_transfers = get_columns_containing(['element', 'transfers'], players.loc[players['gw']==gw-1])
            player_gw = pd.merge(without_transfers, previous_week_transfers, how='left')
        '''
        player_gw['position'] = player_gw.apply(lambda x: position_dict[x['position']], axis=1) 
        player_gw['team'] = player_gw.apply(lambda x: team_dict[x['team']], axis=1)
        '''
        players = pd.concat([players, player_gw], axis=0, sort=True)

    # only save the ones that we got from the trusted mothersource, vaastav
    good_data_weeks = players.loc[~(players['gw'].isin(VASTAAV_NO_RESPONSE_WEEKS))]
    good_data_weeks.to_csv(PLAYER_DB)
    
    print('size of previous week players df: ', players.shape)
    return players 


# Next Week Player
### FPL API bootstrap --> elements  
# gw, element, team, position, value, 'transfers_in','transfers_out','transfers_balance','selected','health'
def update_player_current(prev_week_players, current_gw):
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = proper_request("GET", url, headers=None)
    resp = response.json()
    total_players = resp['total_players']
    players_raw = pd.DataFrame(resp['elements'])

    relevant = ['id','team','element_type','now_cost','transfers_in_event', 'transfers_out_event', 'selected_by_percent']
    df = players_raw[relevant]
    df.columns = ['element', 'team', 'position', 'value', 'transfers_in','transfers_out','selected']

    regularized_transfer_info = transfer_regularizer_for_current(df[['transfers_in', 'transfers_out', 'selected']], current_gw, total_players)
    df = pd.concat([df.drop(['transfers_in', 'transfers_out', 'selected'], axis=1), regularized_transfer_info],axis=1).reset_index(drop=True)

    gw_col = full_column(current_gw, df.shape[0], 'gw')
    df = pd.concat([gw_col, df], axis=1).reset_index(drop=True) #adding gw_column
    print('size of current week players df: ', df.shape)
    updated = pd.concat([prev_week_players, df], axis=0, ignore_index=True, sort=True)
    return updated


# Update Team 
### Fixtures DF for 'gw', 'day', 'hour', 'team', 'opponent', 'was_home'
### API for 14 stats for all games w gw - 1 
def update_team_previous(current_gw, fixtures_df):
    if current_gw == 1:
        return
    odds_df = pd.read_csv(ODDS_DB, index_col=0)
    teams = safely_get_database(TEAM_DB, current_gw)

    try:
        last_active_gw = max(teams['gw'])
        teams = teams.loc[teams['gw']<last_active_gw] #remove last weeks stuff
        gws = teams['gw'].unique()
        need_filling = missing_numbers(gws, current_gw)
    except KeyError: #first time we doing this season
        need_filling = [x+1 for x in range(current_gw) if x != current_gw - 1]

    for gw in need_filling:
        relevant = ['gw', 'day', 'hour', 'team', 'opponent', 'was_home', 'kickoff_time']
        df = fixtures_df[relevant]
        df = df.loc[df['gw']==gw]

        SEASON = 2020
        season = '2020-21'
        premier_league = get_premier_league_id(SEASON) 
        vastaav_converter = id_to_teamname_converter(season)# db id --> namestring
        api_converter = api_id_to_teamname_converter(premier_league) #teamname_to_id_converter(season) # api id --> namestring
        super_converter = api_to_db_id_converter(api_converter, vastaav_converter) #api id --> db id

        new_rows = []
        dates = df['kickoff_time'].apply(lambda x: x[:10]).unique()
        for date in dates:
            fixtures = get_fixture_ids(premier_league, date)
            for fix in fixtures: #(id, h, a, home_g, away_g, status)
                if fix[5] != 'Match Finished':
                    continue #match has not taken place yet 

                fixture_id = fix[0]
                raw_odds = odds_df.loc[odds_df['fixture_id']==fixture_id].reset_index(drop=True)
                if raw_odds.shape[0] == 0:
                    raise Exception("Do not have odds stored for this fixture")
                stats = get_match_stats(fixture_id) # Sh, Sa, ST, F, C dict (8 items)

                home_id = super_converter[fix[1]]
                away_id = super_converter[fix[2]]
                home_goals = fix[3]
                away_goals = fix[4]
                for row in bake_previous_stats_from_raw(home_id, away_id, home_goals, away_goals, stats, raw_odds):
                    row = correct_odds_if_necessary(row, odds_df, teams)
                    new_rows.append(row)

        stat_cols = pd.concat(new_rows, axis=1).T
        df = df[['gw', 'day', 'hour', 'team', 'opponent', 'was_home']]
        full_gw = pd.merge(df, stat_cols, how='left', left_on=['team', 'opponent'], right_on = ['team', 'opponent'])
        teams = pd.concat([teams, full_gw], axis=0, ignore_index=True, sort=True)
    
    print('size of previous weeks week teams df: ', teams.shape)
    teams.to_csv(TEAM_DB)
    return teams


# Next Week Team
### Use Fixtures DF to get 'gw', 'day', 'hour', 'team', 'opponent', 'was home'
### Use API to get oddsW, oddsD, oddsL for all games w gw, store odds for matches in seperate df
def update_team_current(current_gw, fixtures_df):
    odds_df = pd.read_csv(ODDS_DB, index_col=0)
    teams = safely_get_database(TEAM_DB, current_gw)

    df = fixtures_df[['gw', 'day', 'hour', 'team', 'opponent', 'was_home', 'kickoff_time']]
    df = df.loc[df['gw']==current_gw]

    SEASON = 2020
    season = '2020-21'
    premier_league = get_premier_league_id(SEASON) 
    vastaav_converter = id_to_teamname_converter(season)# db id --> namestring
    api_converter = team_id_converter_api(premier_league) # api id --> namestring
    super_converter = api_to_db_id_converter(api_converter, vastaav_converter) #api id --> db id

    new_rows = []
    dates = df['kickoff_time'].apply(lambda x: x[:10]).unique()
    for date in dates:
        print('date in dates= ', date)
        fixtures = get_fixture_ids(premier_league, date)
        for fix in fixtures: #(id, h, a, home_g, away_g, status)
            if fix[5] == 'Match Finished':
                #raise Exception("Match has already taken place!!!")
                pass

            fixture_id = fix[0]
            raw_odds = odds_df.loc[odds_df['fixture_id']==fixture_id].reset_index(drop=True)
            if raw_odds.shape[0] == 0:
                print('This is a bad fixture id we don"t have odds for: ', fixture_id)
                # we should probably fix this later and make it have to do with the actual average
                default_raw_odds = pd.DataFrame([[fixture_id, 3,3,3]], columns=['fixture_id','oddsH', 'oddsD', 'oddsA'])
                raw_odds = default_raw_odds
                #raise Exception("Do not have odds stored for this fixture")

            home_id = super_converter[fix[1]]
            away_id = super_converter[fix[2]]
            for row in bake_current_stats_from_raw(home_id, away_id, raw_odds):
                new_rows.append(row)

    stat_cols = pd.concat(new_rows, axis=1).T
    df = df[['gw', 'day', 'hour', 'team', 'opponent', 'was_home']]
    full_gw = pd.merge(df, stat_cols, how='left', left_on=['team', 'opponent'], right_on = ['team', 'opponent'])
    
    print('size of current week teams df: ', full_gw.shape)
    teams = pd.concat([teams, full_gw], axis=0, ignore_index=True, sort=True)
    teams.to_csv(TEAM_DB)
    # need to rewrite odds to include the team id because we need to associate with fixtures_df!!!!!!!


## @ param: self explanatory vars passed in from other functions
##          getting_postseason: if this is being called and want _N1 to be added 
## @ Summary: Processing of the raw data
def processing_overseer(current_gw, form_lengths, forward_pred_lengths, fixtures_df, raw_players, getting_postseason = False):
    team_concessions = make_team_fantasy_point_concessions(current_gw, raw_players, fixtures_df)
    team_concessions.to_csv(DROPBOX_PATH + "team_concessions.csv")
    raw_teams = pd.read_csv(TEAM_DB, index_col=0)
    raw_teams = pd.merge(raw_teams, team_concessions, how='left', on=['gw','team','opponent']) ## new addition

    players = online_processed_player_season(raw_players, form_lengths, forward_pred_lengths, getting_postseason=getting_postseason)
    print('proccessed players', players.shape, 'expect more than half 205 features, bcz 1,2,3,6')
    teams = online_processed_team_season(raw_teams, form_lengths, forward_pred_lengths)
    teams = online_opponent_datacollect(teams, forward_pred_lengths, fixtures_df)
    total = pd.merge(players, teams, how='left', left_on=['team','gw'], right_on = ['team','gw'])

    total.to_csv(DROPBOX_PATH + "resolve_concessions_testing.csv")
    total = resolve_concessions(total)
    total = add_alternate_position_representations(total)

    print('proccessed teams', teams.shape, total.shape, 'teams and total shape')
    return total 


# @return: current weekly stats which will be used for regression
def current_week_full_stats(season, form_lengths, forward_pred_lengths, ignore_gwks=[]):
    fixtures_df, current_gw = make_fixtures_df(season, ignore_gwks=ignore_gwks)
    print('currently thinks it is gameweek ', current_gw, ' because that is when the first unplayed game was')
    update_odds_df(fixtures_df, current_gw) 

    raw_players = update_player_previous(season, current_gw) 
    raw_players = update_player_current(raw_players, current_gw)
    update_team_previous(current_gw, fixtures_df)
    update_team_current(current_gw, fixtures_df)
    print('updated all tables') # full raw dataset
    
    total = processing_overseer(current_gw, form_lengths, forward_pred_lengths, fixtures_df, raw_players)
    current = total.loc[total['gw']==current_gw]
    current = current.replace([np.inf, -np.inf], 0) #infinities are treated as 0
    return current #~(8694x473) 


# This will save into the season folder the end year results, provided you moved
# preliminary odds, fixtures, player raw, & team raw
def finish_recording_season_gw38(season):
    prefix = DROPBOX_PATH + f"{season}/"
    redefine_globals(prefix)
    fixtures_df = pd.read_csv(prefix + "fix_df.csv", index_col=0)
    current_gw = 39
    update_player_previous(season, current_gw) 
    update_team_previous(current_gw, fixtures_df)
    print('updated all tables')

# The way to create the yearly datasets post 2019
def get_season_data_and_save(season, form_lengths, forward_pred_lengths):
    prefix = DROPBOX_PATH + f"{season}/"
    redefine_globals(prefix)
    print(PLAYER_DB)
    current_gw = 38
    fixtures_df = pd.read_csv(prefix + "fix_df.csv", index_col=0)
    raw_players = pd.read_csv(PLAYER_DB, index_col=0)
    current = processing_overseer(current_gw, form_lengths, forward_pred_lengths, fixtures_df, raw_players, getting_postseason = True)
    int_season = int(season[2:4]) * 100 + int(season[5:7])
    current['season'] = int_season
    current.to_csv(prefix + f"Processed_Dataset_{int_season}.csv")

#@param: folder with trailing slash, max_hit is pos int multiple of four
#@return dictionary with keys 1-n, meaning number of transfers, vals are avg deltas  
#       val is False if there have been no such games yet 
def make_delta_dict(folder, max_hit):
    delta_dict = {}
    deltas = pd.read_csv(folder+'deltas.csv', index_col=0)
    for num_transfers in range(1, 3+max_hit//4):
        if deltas.shape[0] == 0: #before has been initialized
            delta_dict[num_transfers] = False 
            continue 

        these_deltas = deltas.loc[deltas['num_transfers']==num_transfers]
        if these_deltas.shape[0]==0:
            delta_dict[num_transfers] = False 
        else:
            delta_dict[num_transfers] = these_deltas['delta'].mean()
    return delta_dict

#@param: folder with training slash, choice report already comes with proper column labelling
# concatenates this weeks num_transfers and scores to the folder 
def update_delta_db(folder, choice_report):
    deltas = pd.read_csv(folder+'deltas.csv', index_col=0)
    all_deltas = pd.concat([deltas, choice_report],axis=0).reset_index(drop=True)
    all_deltas.to_csv(folder+'deltas.csv')


#@param, folder with trailing slash, dict with {chip: (%of top, std_deviations, choice_method)}
    #where choice method is either 'max', 'min', 'avg'
#@return: dict with keys= str(chipname) val=float, avg score
def make_chip_dict(folder, gw, chip_threshold_construction):
    chip_dict = {}
    chips = pd.read_csv(folder+'chips.csv', index_col=0)
    for chip in ('wildcard', 'freehit', 'bench_boost', 'triple_captain'):
        if chips.shape[0] == 0: #before has been initialized
            chip_dict[chip] = False 
            continue 

        these_chips = chips[chip]
        if these_chips.shape[0]==0:
            chip_dict[chip] = False 
        else:
            nominal_mult, stddev_thresh, choice_method = chip_threshold_construction[chip]
            nominalmethod = these_chips.max()*nominal_mult
            if chips.shape[0] > 1:
                zscoremethod = these_chips.mean() + stddev_thresh*these_chips.std() # will be 0 if N=1
            else: 
                zscoremethod = these_chips.mean() + stddev_thresh*0
            
            if choice_method == 'min':
                chip_dict[chip] = min(nominalmethod, zscoremethod)
            elif choice_method == 'max':
                chip_dict[chip] = max(nominalmethod, zscoremethod)
            elif choice_method == 'avg':
                chip_dict[chip] = sum([nominalmethod, zscoremethod])/2
            else:
                raise Exception("not a valid chip-threshold-selection choice method")
            if chip == 'wildcard' and gw > 33: 
                games_left = 39-gw
                chip_dict[chip] *= games_left/6

    return chip_dict

#@param: folder with slash,  floats that describe weekly scores
# writes to the csv where we store the info
def update_chip_db(folder, gw, wildcard_pts, freehit_pts, captain_pts, bench_pts):
    chips = pd.read_csv(folder+'chips.csv', index_col=0)
    week_info = pd.DataFrame([[gw, wildcard_pts, freehit_pts, bench_pts, captain_pts]],\
        columns=['gw','wildcard', 'freehit','bench_boost','triple_captain']) 
    all_chips = pd.concat([chips, week_info], axis=0).reset_index(drop=True)
    all_chips.to_csv(folder+'chips.csv')

# checks if overseer has completed this week
def already_made_moves(folder, gw):
    try:
        path = folder + 'made_moves.csv'
        print('right before this')
        df = pd.read_csv(path, index_col=0)
        print('right after')
        print(df)
        truthality = gw in df['gw'].to_list()
        print(truthality)
        return truthality
    except: #first time doing this
        pd.DataFrame().to_csv(path)
        return False

# logs that overseer has completed this week
def log_gameweek_completion(folder, gw, transfer_info):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date_info = [int(t[s:e]) for s,e in zip([0,5,8,11],[4,7,10,13])]
    path = folder + 'made_moves.csv'
    df = pd.read_csv(path, index_col=0)
    new_row = pd.DataFrame([[gw] + date_info + transfer_info], columns=['gw','year', 'month', 'day', 'hour','num_transfers', 'chip'])
    final_df = pd.concat([df, new_row], axis=0)
    final_df.to_csv(path)

#return true if the csv sees evidence of database and predictions constructed today
def check_if_explored_today():
    path = DROPBOX_PATH + 'dates_updated.csv'
    df = pd.read_csv(path, index_col=0)
    if df.shape[0] == 0: # first read
        return False

    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    y,m,d = [int(t[s:e]) for s,e in zip([0,5,8],[4,7,10])]

    did_occur = df.loc[(df['year']==y)&(df['month']==m)&(df['day']==d)].shape[0] 
    if did_occur == 1:
        return True
    elif did_occur ==0:
        return False
    else:
        raise Exception("recording exploration dates wrong = more than one occurance")

#updates the db with verifcation predictions are up to date
def update_explored_today():
    path = DROPBOX_PATH + 'dates_updated.csv'
    df = pd.read_csv(path, index_col=0)
    
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    y,m,d = [int(t[s:e]) for s,e in zip([0,5,8],[4,7,10])]

    if df.shape[0] == 0: #for now
        not_already_logged = True
    else:
        not_already_logged = df.loc[(df['year']==y)&(df['month']==m)&(df['day']==d)].shape[0] == 0
    if not_already_logged:
        new_row = pd.DataFrame([[y,m,d]], columns=['year', 'month', 'day'])
        df = pd.concat([df, new_row], axis=0)
    df.to_csv(path)
    
if __name__ == "__main__":
    FORM_LENGTHS = [1,2,3,6]
    PREDICTION_LENGTHS = [1,2,3,4,5,6]
    #finish_recording_season_gw38('2020-21')
    #get_season_data_and_save('2020-21', FORM_LENGTHS, PREDICTION_LENGTHS)