''' File Goals ''' 
# Create Season-Long Dataset
### 1. Individual Stats 
##### a. meta= gw, date, time?, team, opponent, player_id
####### *to get the team id might have to do a search gw1 to check for teams with same match id 
####### *bcz only match id is given
##### b. record relevant stats
'''-----------------new function------------------'''
### 2. Team Stats
###### a. gf, ga, pts avg(-1, 0, 1), s, sot, c, yc/rc, 
###### c. get team stats from api or datahub (s, sot, yc, rc, corner)
###### b. odds
'''-----------------new function------------------'''
### 3. Aggregation and Opponents
###### a. day_rest forms 1-6, season-avg (total, home/away), scores 1-6
###### b. figure out next 6 opponents 
###### c. opponents stuff for part a

''' Vaastav File Summary '''
## used gw folder and players_raw.csv
#%%
from Accountant_helpers import get_and_save_teamconverter
import importlib 
import Database_helpers 
importlib.reload(Database_helpers)
from general_helpers import safe_to_csv, get_opponents
from Database_helpers import * #helper functions

# Vaastav Data :: (22467 x 50) COMPLETE ~instantaneous
# season= str like '2017-18', gw is number, day0 is got from calling day0 on season, is tuple
# @return N-players (except on double/blank gw) x 48 columns, 35 of which are the core gw stats
def get_raw_player_gw(season, gw, day0):
    #round= gw, name= string, opponent_team= id number, value= whole number, was_home=boolean
    playerMETAstats = ['round', 'name', 'element', 'opponent_team', 'value', 'was_home'] #maybe get rid of was_home and opponent_team if can get somewhere else
    playerMATCHstats = ['assists', 'goals_scored', 'goals_conceded', 'bonus', 'bps', 'clean_sheets',\
        'influence','creativity', 'threat', 'ict_index','yellow_cards', 'red_cards', 'own_goals',\
        'penalties_missed','penalties_saved', 'saves','minutes','total_points'] 
    playerTRANSFERstats = ['selected','transfers_in','transfers_out','transfers_balance']
    
    filename = season + '/gws/gw' + str(gw) + '.csv'
    gw_df = pd.read_csv(filename, encoding='ISO-8859-1')

    ''' get non-statistics and non-processed  '''
    meta = gw_df[playerMETAstats]
    meta.columns = ['gw'] + meta.columns[1:].tolist() #rename round --> gw for reusing old functions
    minimeta = element_position_team(season)
    meta = meta.apply(lambda x: join_minimeta(x, minimeta), axis=1, result_type='expand') #adds team and element type (position)

    ''' change transfers to percentage of weekly '''
    # selected is not % but #selected/TOTAL_selections (tsb%/15)
    # id is used to drop duplicates for the calculation, 
    # this still won't be completely accurate when we take the average, this gameweek will be double weighted, 
    # but at least the weights are accurate
    transfers = gw_df[['element'] + playerTRANSFERstats]
    unique_games = transfers.drop_duplicates()
    total_transfer = unique_games.sum(axis=0)
    total_transfer['selected'] = total_transfer['selected'] / 15 #number of teams
    transfers['transfers_balance'] = transfers['transfers_balance'] / transfers['selected'] #delta trans %
    total_transfer['transfers_balance'] = 1 
    std_transfers =  (transfers / total_transfer).drop('element',axis=1)
    if gw == 1: # these would all be 0, but we get divide by 0 error aka nan
        for col in ['transfers_balance', 'transfers_in', 'transfers_out']:
            std_transfers[col].values[:] = 0
            
    ''' get match stats , make season_column,  make day and time columns  2019-08-10T11:30:00Z '''
    other_stats = gw_df.loc[:, playerMATCHstats]  
    season_col = season_column(season, meta.shape[0])
    day_time_df = pd.DataFrame(gw_df['kickoff_time']).apply(lambda x: format_kickoff(x, day0), axis=1, result_type='expand')
    day_time_df.columns = ['day', 'hour']

    boolean_stats = gw_df.apply(add_boolean_stats, axis=1, result_type='expand')
    gw_raw = pd.concat([season_col, meta,day_time_df,std_transfers,other_stats, boolean_stats],axis=1)###big_errors,other_stats],axis=1)
    return gw_raw

# gets the raw data for the whole season :: COMPLETE ~1 minute
def get_raw_player_season(season):
    print('\n In get_raw_player_season:\n')
    season_df = pd.DataFrame()
    day0 = get_day0(season) 
    for gw in range(1,LAST_GAMEWEEK+1):
        gw_df = get_raw_player_gw(season, gw, day0)
        season_df = pd.concat([season_df, gw_df], axis=0, ignore_index=True)
    return season_df

# (22461 x 205) :: COMPLETE ~28 minutes to run
# Returns some meta, and 5*38 main features 
def processed_player_season(raw_players, form_lengths, forward_pred_lengths):
    print('\n In processed_player_season:\n')
    print('number of players= ', raw_players['element'].unique().shape[0])

    processed_season_df = pd.DataFrame()
    for player in raw_players['element'].unique():
        player_df = raw_players.loc[raw_players['element']==player]
        # now we create custom columns for last 1, last3, last6, avg/match also output columns (pts in next1, next3, next6)
        match_stats = player_df[CORE_STATS + BOOLEAN_STATS + ['gw']] #just gw and integer stats
        extended_match_stats = player_df[CORE_STATS + BOOLEAN_STATS + ['gw', 'was_home']] #includes home

        '''adds _Ln for all n in form_lengths, 22xn additional columns'''
        stat_list = []
        for n in form_lengths:
            stat_list.append( last_games_stats(match_stats, n) )
        stats = pd.concat(stat_list, axis=1)
        stats = drop_duplicate_columns(stats) # 38x115 (gw + 3x38cols)
        metadata = make_player_metadata_df(player_df) # 38x9
        player_df = pd.merge(metadata, stats, how='left', left_on=['gw'], right_on = ['gw']) #38x126
        #print(player_df.shape)
        
        '''adds _SAH, _SAA 'season average home/away' for all stats, 22x2 columns'''
        home_avgs, away_avgs, total_avgs = season_averages(extended_match_stats)
        avgs_df = pd.concat([home_avgs, away_avgs, total_avgs],axis=1)
        avgs_df = drop_duplicate_columns(avgs_df)
        player_df = pd.merge(player_df, avgs_df, how='left', left_on=['gw'], right_on = ['gw']) #38 x 240 (gw + 11 + 6*38)
        #print(player_df.shape)

        '''turn home and away statistics into only 1, the <stat>SALOC'''
        home_away = locations_per_week(extended_match_stats)
        saloc_columns = location_specific_columns(player_df, home_away)
        player_df = pd.merge(player_df, saloc_columns, how='left', left_on=['gw'], right_on = ['gw']) #38 x 202 (gw + 11 + 5*38)
        player_df = drop_columns_containing(['SAH', 'SAA'], player_df) 

        '''specific additions :: ppmin'''
        engineered_columns = ppm_column(player_df)
        player_df = pd.merge(player_df, engineered_columns, how='left', left_on=['gw'], right_on = ['gw'])

        '''NEXT MATCHES :: pts on next 1,2,3,4,5,6 ---  '_N1', '_N3', '_N6' '''
        for n in forward_pred_lengths:
           player_df = pd.merge(player_df, prediction_stats(match_stats, n, 'total_points'), how='left', left_on=['gw'], right_on = ['gw'])
        player_df = pd.merge(player_df, prediction_stats(match_stats, 1, 'minutes'), how='left', left_on=['gw'], right_on = ['gw'])
        #print(player_df.shape) #38 x 208
        
        processed_season_df = pd.concat([processed_season_df, player_df], axis=0)

    '''change transfer info to be w.r.t. the top in the gw to reduce interweek/year variance'''
    processed_season_df = change_columns_to_weekly_comparisons(processed_season_df, ['transfers'])
    '''create multiple position representations as this feature is heavily selected for yet difficult to capture proper relationship'''
    processed_season_df = add_alternate_position_representations(processed_season_df)
    return processed_season_df #23660 x 205 (with 650 ish players)



# Datahub data, preprocessed :: COMPLETE  ~ < 15 sec
# @return: gf, ga, pts, s, sot, corner, fouls, opponent, 
def get_raw_team_season(season):
    df = pd.read_csv(DATAHUB)
    season = int(season[2:4]+season[5:])
    df = df.loc[df['season']==season]
    converter = get_team_metaInfo(season) #getting ids
    
    df['team'] = df.apply(lambda x: team_to_id(x, converter), axis=1)
    df['opponent'] = df.apply(lambda x: team_to_id(x,converter, opponent=True), axis=1)
    df.columns = ['round#' if x=='gw' else 'was_home' if x=='home' else x for x in df.columns] #change it to what it really is, fix home to match also

    relevant = ['season', 'round#', 'day', 'team', 'opponent', 'was_home', 'FTgf', 'FTga', 'pts', 'oddsW', 'oddsD', 'oddsL', 'Sf', 'Sa', 'STf', 'STa', 'Cf', 'Ca', 'Ff', 'Fa']
    team_stats = df[relevant]
    team_stats = correct_bad_odds_team_stats(team_stats)
    return team_stats

# Get's individual team stats :: COMPLETE ~40 seconds
# @params: form_lengths and forward_pred_lengths are lists of integers that dictate how far we look back&foward
# @return: df with requested forms, forward predictions/opponents, season averages
def processed_team_season(season, form_lengths, forward_pred_lengths, raw_players):
    print('\n In processed_team_season:\n')
    # use raw_player to get the proper 'gw' 
    raw_teams = get_raw_team_season(season)
    team_concessions = database_make_team_fantasy_point_concessions(LAST_GAMEWEEK+1, raw_players)

    year_df = pd.DataFrame()
    for team in raw_teams['team'].unique():
        team_df = raw_teams.loc[(raw_teams['team']==team)].reset_index(drop=True)
        gws = get_gameweeks(team, raw_players)
        team_df = team_df.iloc[:gws.shape[0],:] # dropping to proper level for match
        team_df['gw']=gws
        team_df = pd.merge(team_df, team_concessions, how='left', on=['gw','team','opponent']) ## new addition
        match_stats = team_df.drop(['round#', 'season', 'day', 'team', 'opponent', 'was_home'], axis=1) #just gw and integer stats
        extended_match_stats = team_df.drop(['round#', 'season', 'day', 'team', 'opponent'], axis=1) #includes home

        ## now we create custom columns for last 1, last3, last6, avg/match also output columns (pts in next1, next3, next6) ##
        '''adds _Ln for all n in form_lengths, 22xn additional columns'''
        stat_list = []
        for n in form_lengths:
            stat_list.append( last_games_stats(match_stats, n) )
        stats = pd.concat(stat_list, axis=1)
        stats = drop_duplicate_columns(stats) # 38x115 (gw + 3x38cols) #this got rid of the gw repetition
        metadata = make_team_metadata_df(team_df) # 38x6
        team_df = pd.merge(metadata, stats, how='left', left_on=['gw'], right_on = ['gw']) #38x121
        #print('after metadata', team_df.shape)
        
        '''adds _SAH, _SAA 'season average home/away' for all stats, 22x2 columns'''
        home_avgs, away_avgs, total_avgs = season_averages(extended_match_stats)
        avgs_df = pd.concat([home_avgs, away_avgs, total_avgs],axis=1)
        avgs_df = drop_duplicate_columns(avgs_df)
        team_df = pd.merge(team_df, avgs_df, how='left', left_on=['gw'], right_on = ['gw']) #38 x 90 (gw + 5 + 6*14)
        #print('after season avgs: ', team_df.shape)

        '''turn home and away statistics into only 1, the <stat>SALOC'''
        home_away = locations_per_week(extended_match_stats)
        saloc_columns = location_specific_columns(team_df, home_away)
        team_df = pd.merge(team_df, saloc_columns, how='left', left_on=['gw'], right_on = ['gw']) #38 x 199 (gw + 8 + 5*38)
        team_df = drop_columns_containing(['SAH', 'SAA'], team_df) # 38 x 76
        #print('after saloc: ', team_df.shape)

        '''CUSTOM TEAM ADDITIONS :: days_rest, ?= (future: league ranking)'''
        new_col = team_df.apply(lambda x: days_rest(x, team_df), axis=1)
        team_df.insert(5, 'days_rest', new_col) # 38 x 77

        year_df = pd.concat([year_df, team_df], axis=0)
    return year_df # (GW * 20) x 77 array

# Gets opponent stats :: COMPLETE ~70 sec
# goes from processed individual dataset to include opponent information
# Now Just Get OPPN for N in forward_pred for all of 14 stat + days,
# also get num_opponents
def opponent_datacollect(year_df, forward_pred_lengths):
    print('\n In opponent_datacollect:\n')
    '''get 14*5 stats, day, (team & gw for merging)'''
    all_teams = []
    for team in year_df['team'].unique():
        team_df = year_df.loc[year_df['team']==team]
        # dataframe with team and gw
        anchor_df = team_df[['team', 'gw']]
        # for all gw
        all_dfs = []
        for gw in range(1, LAST_GAMEWEEK + 1):
            week_dfs = []
            # for all OPPN 
            for n in forward_pred_lengths:
                # return num opp, and opp stats (71)
                opp_n = opponent_statistics(year_df, team, gw, n)
                week_dfs.append(opp_n)
            this_week = pd.concat(week_dfs, axis=0)
            this_week['FIX1_days_rest'] = opp_days_rest(year_df, team, gw)#opponent's rest time
            all_dfs.append(this_week) # should be 1 + 71 * n --> 1 x 427

        #concat, concat with team/gw
        full_team_stats = pd.concat(all_dfs, axis=1).T #38 x 1+71*n
        full_team_stats[['team', 'gw']] = anchor_df #38 x 3 + 71*n --> 38x429
        all_teams.append(full_team_stats)
    #concat
    opp_df = pd.concat(all_teams, axis=0, ignore_index=True) #760 x 3 + 71*n --> 760 x 429
    return opp_df

# (760 x 504) :: COMPLETE ~3 minutes
# Goes full stack and gets all team data + opponent 
def full_team_season(season, form_lengths, forward_pred_lengths, raw_players=None):
    print('\n In full_team_season:\n')
    if raw_players is None:
        raw_players = get_raw_player_season(season)
    team = processed_team_season(season, form_lengths, forward_pred_lengths, raw_players)
    opponent = opponent_datacollect(team, forward_pred_lengths)
    total = pd.merge(team, opponent, how='left', left_on=['team','gw'], right_on = ['team','gw'])
    return total
    # should be 760 x 507


# (22461 x 706) :: COMPLETE ~ 30-35 minutes
def combine_player_and_team(season, form_lengths, forward_pred_lengths, raw_players=None):
    print('season= ', season)
    if raw_players is None:
        raw_players = get_raw_player_season(season)
    print('raw_players= ', raw_players.shape)
    player_info = processed_player_season(raw_players, form_lengths, forward_pred_lengths)
    print('processed_players = ', player_info.shape)
    team_info = full_team_season(season, form_lengths, forward_pred_lengths, raw_players=raw_players)
    print('team info= ', team_info.shape)

    # now need to merge team into player on (team, gw)
    total = pd.merge(player_info, team_info, how='left', left_on=['season', 'team','gw'], right_on = ['season', 'team','gw'])
    total = resolve_concessions(total)
    return total
    # 22461 x 205 (left) with 760 x 504 (right)
    # should be 23679 x 706



###################################################
############# FUNCTIONS TO CALL DIRECTLY ##########
###################################################

# we can move the combined dataset to the folders we using for the future
def shift_dataset_to_folders(dataset_path, years):
    full_df = pd.read_csv(dataset_path, index_col=0)
    for year in years:
        folder = DROPBOX_PATH + 'Our_Datasets/20' + str(year)[:2] + '-' + str(year)[2:] + '/'
        year_path = folder + f'Processed_Dataset_{year}.csv'
        fix_path = folder + 'fix_df.csv'
        df = full_df.loc[full_df['season']==year]
        safe_to_csv(df, year_path)
        df = df[['gw', 'team', 'opponent', 'day', 'hour', 'was_home']].drop_duplicates(ignore_index=True)
        
        # we need to fix that opponents are represented by the *20 thing
        fix_list = []
        for _, row in df.iterrows():
            opps = get_opponents(row['opponent'])
            for opp in opps:
                row.loc['opponent'] = opp 
                fix_list.append(row.copy())

        fix_df = pd.concat(fix_list, axis=1).T
        safe_to_csv(fix_df, fix_path)
        
        # saving team converter
        get_and_save_teamconverter(year) #int year



import time
#wtihout 2019 but with team stats, much better cv score
def get_the_database(form_lengths, prediction_lengths, years = ['2018-19','2017-18','2016-17']):
    start = time.time()
    all_years = []
    for year in years:
        year_df = combine_player_and_team(year, form_lengths, prediction_lengths)
        all_years.append(year_df)
    total = pd.concat(all_years, axis=0, ignore_index=True)
    print('all together = ', total.shape) #67909 x 627
    total.to_csv(DROPBOX_PATH + "updated_training_dataset.csv")
    end = time.time() 
    print("Timing (min): ", end-start) # 90 min or like 45 ish minutes now?? did python get mad optimized or i do something


if __name__ == "__main__":
    form_lengths = [1,2,3,6]
    prediction_lengths = [1,2,3,4,5,6]
    get_the_database(form_lengths, prediction_lengths)
    
    shift_dataset_to_folders(DROPBOX_PATH + 'updated_training_dataset.csv', [1617, 1718, 1819])    

# %% 
