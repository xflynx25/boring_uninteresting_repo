from private_versions.constants import STRING_SEASON
import importlib 
import Accountant_2helpers 
importlib.reload(Accountant_2helpers)
from Accountant_2helpers import * #helper functions



#########################HELPERS####################################

# returns set of all integers > 0 < top not in listlike
def missing_numbers(listlike, top):
    missing = set()
    for n in range(1, top):
        if n not in listlike:
            missing.add(n)
    return missing

#returns dictionary that takes api_id as key and returns string form
# 1 REQUEST
def api_id_to_teamname_converter(league_id):
    return team_id_converter_api(league_id)

#returns dictionary that takes id as key and returns string form
def id_to_teamname_converter(season):
    url = VASTAAV_ROOT + str(season) + '/teams.csv'
    print(url)
    df = get_df_from_internet(url)[['id', 'name']]
    converter = {}
    for _, row in df.iterrows():
        converter[row['id']] = row['name']
    return converter

#reverse of the above
def teamname_to_id_converter(season):
    url = VASTAAV_ROOT + str(season) + '/teams.csv'
    print('url is ', url)
    df = get_df_from_internet(url)[['id', 'name']]
    converter = {}
    for _, row in df.iterrows():
        converter[row['name']] = row['id']
    return converter

# api: api id --> name
# db: db id --> name
# return: api id --> db
def api_to_db_id_converter(api_converter, db_converter):
    converter = {}
    for item in api_converter.items():
        api_id = item[0]
        fullname = item[1]
        api_codename = fullname[:3] + fullname[-3:]
        for item2 in db_converter.items():
            name = item2[1]
            #print("Match::: ", fullname, "  w.  ", name) # if we need to make sure working
            db_codename = name[:3] + name[-3:]
            if api_codename == db_codename:
                converter[api_id] = item2[0]
                break
            if fullname[:3] == 'Tot': #spurs needs to be edited
                api_codename_2 = 'Spuurs'
                if api_codename_2 == db_codename:
                    converter[api_id] = item2[0]
                    break
            if fullname[-6:] == 'United': #fixing up man united vs utd
                api_codename_2 = api_codename[:3] + 'Utd'
                if api_codename_2 == db_codename:
                    converter[api_id] = item2[0]
                    break
                
    if len(converter) != 20:
        raise Exception('Not full List of teams returned by db or api, or match failure')
    return converter

# make two rows with Sh, Sa, ST, F, C dict (8 items) + pts + gf, ga 13 total
#  oddsW, oddsD, oddsL, team, opponent
# generator for use in update team previous
def bake_previous_stats_from_raw(home_id, away_id, home_goals, away_goals, stats, raw_odds):
    def pts_from_score(gf, ga):
        diff = gf - ga
        if diff>0:
            return 3
        elif diff < 0:
            return 0
        elif diff == 0:
            return 1
        else:
            raise Exception("Decimal number of goals...lost in the float~~")

    home_items = [home_id, home_goals, stats['Sh'], stats['STh'], stats['Ch'], stats['Fh']]
    away_items = [away_id, away_goals, stats['Sa'], stats['STa'], stats['Ca'], stats['Fa']]
    for n in range(2):
        if n % 2 == 0:
            team = home_items
            opp = away_items
            odds = pd.Series([raw_odds['oddsH'].iloc[0], raw_odds['oddsD'].iloc[0], raw_odds['oddsA'].iloc[0]], index=['oddsW', 'oddsD', 'oddsL'])
        else:
            team = away_items
            opp = home_items
            odds = pd.Series([raw_odds['oddsA'].iloc[0], raw_odds['oddsD'].iloc[0], raw_odds['oddsH'].iloc[0]], index=['oddsW', 'oddsD', 'oddsL'])
        

        team = pd.Series(team, index=['team', 'FTgf', 'Sf', 'STf', 'Cf', 'Ff'])
        opp = pd.Series(opp, index=['opponent', 'FTga', 'Sa', 'STa', 'Ca', 'Fa'])
        combined = pd.concat([team, opp, odds], axis=0)
        combined['pts'] = pts_from_score(combined['FTgf'], combined['FTga'])
        yield combined

# make two rows with oddsW, oddsD, oddsL, team, opponent
# generator for use in update team previous
def bake_current_stats_from_raw(home_id, away_id, raw_odds):
    for n in range(2):
        if n % 2 == 0:
            stats = [home_id, away_id, raw_odds['oddsH'].iloc[0], raw_odds['oddsD'].iloc[0], raw_odds['oddsA'].iloc[0]]
        else:
            stats = [away_id, home_id, raw_odds['oddsA'].iloc[0], raw_odds['oddsD'].iloc[0], raw_odds['oddsH'].iloc[0]]
        yield pd.Series(stats, index=['team', 'opponent', 'oddsW', 'oddsD', 'oddsL'])

# for use in odds df, making up for bad api
# takes in premier league id, current state of odds_df, fixtures_df, current_gw
# returns concatenated df of the odds information of previously missing stuff
# len(need_fixtures) REQUESTS
def individual_game_odds(premier_league, final_odds, fixtures_df, current_gw, what_weeks='equal'):
    if what_weeks == 'equal':
        unique_dates_this_week = fixtures_df.loc[fixtures_df['gw']==current_gw]['kickoff_time'].apply(lambda x: x[:10]).unique()
    if what_weeks == 'upto':
        unique_dates_this_week = fixtures_df.loc[fixtures_df['gw']<=current_gw]['kickoff_time'].apply(lambda x: x[:10]).unique()
    week_fixtures = []
    for date in unique_dates_this_week:
        week_fixtures = week_fixtures + [x[0] for x in get_fixture_ids(premier_league, date)]
    need_fixtures = set(week_fixtures).difference(final_odds['fixture_id'])
    print('in individual game odds')
    print(need_fixtures)
    clutch_odds = pd.DataFrame(get_bet365_odds_by_fixtures(need_fixtures))
    
    available_odds = []
    for match in clutch_odds.items():
        group = [ match[0],match[1][0],match[1][1],match[1][2] ]
        available_odds.append(pd.Series(group, index=['fixture_id','oddsH', 'oddsD', 'oddsA']))
    if available_odds == []:
        return pd.DataFrame()
    return pd.concat(available_odds, axis=1, ignore_index=True).T



#########################END HELPERS####################################

######################### Main Functions for Building Processed#########

def get_raw_gw_df(season, gw):
    stitching_a_404 = False
    try: 
        filename = VASTAAV_ROOT + season + '/gws/gw' + str(gw) + '.csv'
        gw_df = get_df_from_internet(filename)
    except Custom404Exception:
        backup_datacollection_path = MANUAL_VASTAAV_ROOT + 'gw' + str(gw) + '.csv'
        previous_player_raw = pd.read_csv(DROPBOX_PATH + 'player_raw.csv', index_col=0)
        manually_replace_vastaav_this_week(gw, backup_datacollection_path, previous_player_raw)
        gw_df = safe_read_csv(backup_datacollection_path)
        print('backup collection seemed to work')
        stitching_a_404 = True 
        VASTAAV_NO_RESPONSE_WEEKS.append(gw) 
    return gw_df, stitching_a_404

def online_raw_player_gw(season, gw):
    #element= id, round= gw, name= string, opponent_team= id number, value= whole number, was_home=boolean
    playerMETAstats = ['round', 'element', 'value', 'was_home'] #maybe get rid of was_home and opponent_team if can get somewhere else
    # 38 main player statistics (after combine errors to big_errors)
    playerMATCHstats = ['assists', 'goals_scored', 'goals_conceded', 'bonus', 'bps', 'clean_sheets',\
        'influence','creativity', 'threat', 'ict_index','yellow_cards', 'red_cards', 'own_goals',\
        'penalties_missed','penalties_saved', 'saves','minutes','total_points'] 
    playerTRANSFERstats = ['selected','transfers_in','transfers_out','transfers_balance']
    #big_error_keys = ['errors_leading_to_goal', 'errors_leading_to_goal_attempt']
 
    print('in online_raw_player_gw... season= ', season, 'gw= ', gw)
    gw_df, stitching_a_404 = get_raw_gw_df(season, gw)

    # get non-statistics and non-processed data
    meta = gw_df[playerMETAstats]
    meta.columns = ['gw'] + meta.columns[1:].tolist() #rename round --> gw for reusing old functions
    if stitching_a_404:
        minimeta = online_element_position_team404(gw_df)
    else:
        minimeta = online_element_position_team(season)
        
    #'''protecting for 404, if vastaav doesn't get back to it we will have to find a way to get new player positions'''
    #meta = meta.loc[meta['element'].isin(minimeta['element'])] ## otherwise will throw an error on custom404 weeks
    meta = meta.apply(lambda x: join_minimeta(x, minimeta), axis=1, result_type='expand') #adds team and element type (position)
    # change transfers to percentage of weekly, selected is not % but #selected/TOTAL_selections (tsb%/15)
    # id is used to drop duplicates for the calculation, 
    # this still won't be completely accurate when we take the average, this gameweek will be double weighted, 
    # but at least the weights are accurate
    transfers = gw_df[['element'] + playerTRANSFERstats]
    unique_games = transfers.drop_duplicates()
    total_transfer = unique_games.sum(axis=0)
    total_transfer.loc['selected'] = total_transfer['selected'] / 15 #number of teams
    #transfers.loc['transfers_balance'] = transfers['transfers_balance'] / transfers['selected']#delta trans %
    trans_balance = transfers['transfers_balance'] / transfers['selected']#delta trans %
    trans_balance.name = 'transfers_balance'
    transfers = pd.concat([transfers.drop('transfers_balance', axis=1), trans_balance], axis=1).reset_index(drop=True)
    total_transfer.loc['transfers_balance'] = 1

    std_transfers =  (transfers / total_transfer).drop('element',axis=1)
    if gw == 1: # these would all be 0, but we get divide by 0 error aka nan
        for col in ['transfers_balance', 'transfers_in', 'transfers_out']:
            std_transfers[col].values[:] = 0
    #print('gw: ', gw, ' ', std_transfers['transfers_balance'].iloc[290:310])
    # get match stats 
    other_stats = gw_df.loc[:, playerMATCHstats]
    #create stats that are 1 or 0 for satisfying certain conditions on the match
    boolean_stats = gw_df.apply(add_boolean_stats, axis=1, result_type='expand') 
    # combine errors columns into one big errors section
    ###big_errors = pd.DataFrame(gw_df[big_error_keys].sum(axis=1), columns=['big_errors'])
    #print('gw: ', gw, 'other stats ', other_stats.iloc[290:310, 0:12])
    gw_raw = pd.concat([meta, std_transfers,other_stats, boolean_stats],axis=1)
    """
    printdf = gw_raw.loc[gw_raw['element'].isin(list(range(230, 240)))]
    print('gw: ', gw, 'gw raw ', printdf.iloc[:, 0:17])
    raise Exception("ending on purpose")
    """
    return gw_raw
    
# (~GW*num_players x 205) :: COMPLETE ~28 minutes to run
# Returns some meta, and 5*38 main features 
def online_processed_player_season(raw_players, form_lengths, forward_pred_lengths, getting_postseason=False):
    
    '''now we process player form stats'''
    processed_season_df = pd.DataFrame()
    nplayers = len(raw_players['element'].unique())
    print('number of players ', nplayers)
    for i, player in enumerate(raw_players['element'].unique()):
        print('Player# ', i)
        player_df = raw_players.loc[raw_players['element']==player]
        if player_df.shape[0] == 0: #sometimes very new players will not have any stats yet and will throw us errors. We don't have to consider them
            print('skipping them')
            continue
        
        #print(player_df['name'])
        # now we create custom columns for last 1, last3, last6, avg/match also output columns (pts in next1, next3, next6)
        match_stats = player_df[CORE_STATS + BOOLEAN_STATS + ['gw']]#just gw and integer stats
        extended_match_stats = player_df[CORE_STATS + BOOLEAN_STATS + ['gw', 'was_home']]#includes home
        
        '''adds _Ln for all n in form_lengths, 22xn additional columns'''
        stat_list = []
        for n in form_lengths:
            stat_list.append( last_games_stats(match_stats, n) )
        stats = pd.concat(stat_list, axis=1)
        stats = drop_duplicate_columns(stats) # 38x115 (gw + 3x38cols)
        metadata = online_make_player_metadata_df(player_df) # 38x9
        player_df = pd.merge(metadata, stats, how='left', left_on=['gw'], right_on = ['gw']) #38x121
        
        '''adds _SAH, _SAA 'season average home/away' for all stats, 22x2 columns'''
        home_avgs, away_avgs, total_avgs = season_averages(extended_match_stats)
        avgs_df = pd.concat([home_avgs, away_avgs, total_avgs],axis=1)
        avgs_df = drop_duplicate_columns(avgs_df)
        player_df = pd.merge(player_df, avgs_df, how='left', left_on=['gw'], right_on = ['gw']) #38 x 237 (gw + 8 + 6*38)

        '''turn home and away statistics into only 1, the <stat>SALOC'''
        home_away = locations_per_week(extended_match_stats)
        saloc_columns = location_specific_columns(player_df, home_away)
        player_df = pd.merge(player_df, saloc_columns, how='left', left_on=['gw'], right_on = ['gw']) #38 x 199 (gw + 8 + 5*38)
        player_df = drop_columns_containing(['SAH', 'SAA'], player_df) 

        '''specific additions :: ppmin'''
        engineered_columns = ppm_column(player_df)
        player_df = pd.merge(player_df, engineered_columns, how='left', left_on=['gw'], right_on = ['gw'])
        """
        if int(player) == 224:
            for _, row in player_df.iterrows():
                if row['gw'] > 33:
                    patterns = ['gw', 'minutes', 'goals_scores', 'total_points']
                    print(get_columns_containing(patterns, row))
            raise Exception("custom fo rchecking point")
        """
        '''NEXT MATCHES :: pts on next 1,2,3,4,5,6 ---  '_N1', '_N3', '_N6' '''
        if getting_postseason:
            for n in forward_pred_lengths:
                player_df = pd.merge(player_df, prediction_stats(match_stats, n, 'total_points'), how='left', left_on=['gw'], right_on = ['gw'])
            player_df = pd.merge(player_df, prediction_stats(match_stats, 1, 'minutes'), how='left', left_on=['gw'], right_on = ['gw'])

        processed_season_df = pd.concat([processed_season_df, player_df], axis=0)
        
    '''change transfer info to be w.r.t. the top in the gw to reduce interweek/year variance'''
    processed_season_df = online_change_columns_to_weekly_comparisons(processed_season_df, ['transfers'])
    return processed_season_df #22461 x 205 (with 650 ish players)


# Get's individual team stats :: COMPLETE ~40 seconds
# @params: form_lengths and forward_pred_lengths are lists of integers that dictate how far we look back&foward
# @return: df with requested forms, forward predictions/opponents, season averages
def online_processed_team_season(raw_teams, form_lengths, forward_pred_lengths):

    year_df = pd.DataFrame()
    for team in raw_teams['team'].unique():
        team_df = raw_teams.loc[raw_teams['team']==team]
        #gws = get_gameweeks(team, raw_players)
        #team_df['gw'] = gws
        # now we create custom columns for last 1, last3, last6, avg/match also output columns (pts in next1, next3, next6)
        match_stats = team_df.drop(['day', 'hour', 'team', 'opponent', 'was_home'], axis=1) #just gw and integer stats
        extended_match_stats = team_df.drop(['day','hour', 'team', 'opponent'], axis=1) #includes home

        '''adds _Ln for all n in form_lengths, 12xn additional columns'''
        stat_list = []
        for n in form_lengths:
            stat_list.append( last_games_stats(match_stats, n) )
        stats = pd.concat(stat_list, axis=1)
        stats = drop_duplicate_columns(stats) # 38x115 (gw + 3x38cols) #this got rid of the gw repetition
        metadata = online_make_team_metadata_df(team_df) # 38x6
        team_df = pd.merge(metadata, stats, how='left', left_on=['gw'], right_on = ['gw']) #38x121
        #print('after metadata', team_df.shape)
        
        '''adds _SAH, _SAA 'season average home/away' for all stats, 22x2 columns'''
        home_avgs, away_avgs, total_avgs = season_averages(extended_match_stats)
        avgs_df = pd.concat([home_avgs, away_avgs, total_avgs],axis=1)
        avgs_df = drop_duplicate_columns(avgs_df)
        team_df = pd.merge(team_df, avgs_df, how='left', left_on=['gw'], right_on = ['gw']) #38 x 90 (gw + 5 + 6*14)
        #print('after season avgs: ', team_df.shape)

        '''turn home and away statistics into only 1, the <stat>SALOC'''
        home_away = locations_per_week(extended_match_stats,team_version=True)
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
def online_opponent_datacollect(year_df, forward_pred_lengths, fixtures_df):
    '''get 14*5 stats, day, (team & gw for merging)'''
    all_teams = []
    for team in year_df['team'].unique():
        team_df = year_df.loc[year_df['team']==team]
        # dataframe with team and gw 
        anchor_df = team_df[['team', 'gw', 'was_home', 'day', 'days_rest', 'hour']]
        # for all gw
        all_dfs = []
        for gw in range(1, LAST_GAMEWEEK + 1):
            week_dfs = []
            # for all OPPN 
            for n in forward_pred_lengths:
                # return num opp, and opp stats (71)
                opp_n = online_opponent_statistics(year_df, team, gw, n, fixtures_df)
                week_dfs.append(opp_n)
            this_week = pd.concat(week_dfs, axis=0)
            this_week['FIX1_days_rest'] = online_opp_days_rest(year_df, team, gw, fixtures_df)#opponent's rest time
            all_dfs.append(this_week) # should be 1 + 71 * n --> 1 x 427

        #concat, concat with team/gw
        opp_team_stats = pd.concat(all_dfs, axis=1).T #38 x 1+71*n
        full_team_stats = pd.concat([team_df, opp_team_stats], axis=1)
        #original_team_stats =  get_columns_containing(['_L','SA'], team_df)
        #full_team_stats = pd.concat([original_team_stats, opp_team_stats], axis=1)
        #full_team_stats[['team', 'gw', 'was_home', 'day', 'days_rest', 'hour']] = anchor_df #38 x 3 + 71*n --> 38x429
        all_teams.append(full_team_stats)
    #concat
    return pd.concat(all_teams, axis=0, ignore_index=True) #760 x 3 + 71*n --> 760 x 429

# whenever we read anything in we just want to get rid of the past weeks so we don't create a messed up db
def safely_get_database(db, current_gw):
    players = safe_read_csv(db)
    if players.shape[0] > 0: # skip if just first gameweek so no error
        players = players.loc[players['gw']<current_gw] #don't want to fill more than once 
    return players


# takes in the df with the 3 transfer information columns, selected is in percentage 0 - 100
# returns df with the transfers scaled 
def transfer_regularizer_for_current(df, current_gw, total_players):
    '''editing the raw data'''
    trans_balance = df.apply(lambda x: x['transfers_in']-x['transfers_out'], axis=1)
    new_selected = df.apply(lambda x: float(x['selected']), axis=1) # no harm in transfering these to float, sometimes str
    df = df.drop('selected', axis=1)
    new_selected.name = 'selected'
    trans_balance.name = 'transfers_balance'
    df = pd.concat([df, new_selected, trans_balance], axis=1).reset_index(drop=True)
    
    '''turning into a relative value'''
    total_transfer = df.sum(axis=0)
    total_transfer.loc['selected'] = 100 # it is already in teams selected percentage 

    df = df.assign(transfers_balance = df['transfers_balance'] / (df['selected']/100 * total_players) )#delta trans % (in decimal form)
    total_transfer.loc['transfers_balance'] = 1
    std_transfers =  df / total_transfer
    if current_gw == 1: # these would all be 0, but we get divide by 0 error aka nan
        for col in ['transfers_balance', 'transfers_in', 'transfers_out']:
            std_transfers[col].values[:] = 0
    return std_transfers
 



# @param: list of api_namestrings, df with index=['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
# return: dict with - api namestring --> patch namestring
def api_to_patch_converter(api_names, patch_database):
    patch_names = pd.concat([patch_database['HomeTeam'],patch_database['AwayTeam']],axis=0).unique()

    converter = {}
    for team in api_names:
        team_key = team[:3]

        if team_key == 'Man': #man u, man city
            if team[-1] == 'y':
                converter[team] = 'Man City'
            elif team[-1] == 'd':
                converter[team] = 'Man United'
            else:
                raise Exception("Something went wrong with Manchester")

        elif team_key == 'Wes': #west ham, west brom
            if team[-2] == 'o':
                converter[team] = 'West Brom'
            elif team[-2] == 'a':
                converter[team] = 'West Ham'
            else:
                raise Exception("Something went wrong with West Brom/Ham")

        else: 
            for patch_team in patch_names:
                if patch_team[:3] == team_key:
                    converter[team] = patch_team
                    break

    if len(converter) != 20:
        raise Exception('Not full List of teams returned by db or api, or match failure')
    return converter

# @param: Database is df with index=['Date', 'HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A']
### fixtures df = pd df with index=['gw', 'day', 'hour', 'team', 'opponent', 'was_home', 'kickoff_time']
### current_gw is integer
# @return: final odds df (index=['fixture_id','oddsH', 'oddsD', 'oddsA'])
def patch_odds(odds_df, database, fixtures_df, current_gw):
    print('in patch odds')
    '''go from fixture id to hometeam, awayteam, date'''
    SEASON = INT_SEASON_START
    season = STRING_SEASON
    print(season, SEASON)
    premier_league = get_premier_league_id(SEASON) 
    vastaav_converter = id_to_teamname_converter(season)# db id --> namestring
    api_converter = api_id_to_teamname_converter(premier_league) #teamname_to_id_converter(season) # api id --> namestring
    super_converter = api_to_db_id_converter(api_converter, vastaav_converter) #api id --> db id
    patch_converter = api_to_patch_converter(list(api_converter.values()), database)


    '''for each gameday already played we get all ids'''
    new_rows = []

    kickoff_times = fixtures_df.loc[fixtures_df['gw']<current_gw]['kickoff_time']
    kickoff_dates = kickoff_times.apply(lambda x: x[:10]).unique()
    for kickoff_time in kickoff_dates:
        date = kickoff_time[:10]
        print('date is ', date)
        for fix in list(get_fixture_ids(premier_league, date)): #otherwise start getting async stuff
            fixture_id = fix[0]
            homeName = api_converter[fix[1]]
            awayName = api_converter[fix[2]]
            if fixture_id not in odds_df['fixture_id'].to_list(): #haven't got already
                print('bad fix id is ', fixture_id)

                # get game which matches the fixture id
                home = patch_converter[homeName]
                away = patch_converter[awayName]
                target_match = database.loc[(database['HomeTeam']==home) & (database['AwayTeam']==away)].reset_index(drop=True)
                print('home/away/match is ', home, away, target_match.shape)

                # if there is no matching value check if it is a rescheduled game using fixtures df
                # if not... raise exception because then there is a match we are missing
                # this section is seeing whether the api is passing us stale data about games on this date...
                    # or if we are actually missing dtaa still for a real game (real defined by fix_df)
                if target_match.shape[0] == 0: 
                    home = super_converter[fix[1]]
                    away = super_converter[fix[2]]
                    match = fixtures_df.loc[(fixtures_df['team']==home)&(fixtures_df['was_home']==1)&(fixtures_df['opponent']==away)]
                    earliest_possible = max(kickoff_dates)
                    if match['kickoff_time'].iloc[0] < earliest_possible: #otherwise its a rescheduled game
                        print("No backup data for this match: WILL BE AUTOCORRECTED HOPEFULLY EEEK" + str(fixture_id))
                        continue
                        #raise Exception("No backup data for this match: " + str(fixture_id))
                    else:
                        continue

                # add to list a list of (fixture id, and the matching 3 odds )
                oddsH = target_match['B365H'].iloc[0]
                oddsD = target_match['B365D'].iloc[0]
                oddsA = target_match['B365A'].iloc[0]
                new_rows.append( [fixture_id, oddsH, oddsD, oddsA] )
                
    patched_odds = pd.DataFrame(new_rows, columns=['fixture_id','oddsH', 'oddsD', 'oddsA'])
    print('patched odds for this week\n\n')
    print(patched_odds)
    final_odds = pd.concat([patched_odds, odds_df], axis=0, ignore_index=True, sort=True)
    return final_odds



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
def make_team_fantasy_point_concessions(current_gw, raw_players, fixtures_df):
    fix_df = fixtures_df.loc[fixtures_df['gw']<current_gw]
    all_rows = []
    for _, fixture in fix_df.iterrows():
        gw, team, opponent = fixture['gw'], fixture['team'], fixture['opponent']

        opp_point_list = get_opp_fantasy_points(gw, opponent, raw_players)
        row = [gw, team, opponent] + opp_point_list 
        all_rows.append(row)

    col_names = ['gw','team', 'opponent'] + NEW_STATS

    final = pd.DataFrame(all_rows, columns=col_names)
    return final

# turn the many concession stats into just the one representing their position
# we must keep in mind there will be rows where the concessions have been averaged (dgw) and when they are nan (blank)
def resolve_concessions(total):
    temp_og = get_columns_containing(['concession_pos_1'], total)
    og_column_names = get_columns_containing(['FIX'], temp_og).columns
    new_column_names = []
    for name in og_column_names:
        cutoff_index = name.index("concession") + 14 
        new_column_names.append( name[:cutoff_index] + name[cutoff_index+2:] )

    core = drop_columns_containing(['concession'], total)
    total = online_change_columns_to_weekly_comparisons(total, ['concession']) #big deal line here
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

#@if odds are 0 0 0 in this row, make the avg of all games they involved wit
def correct_odds_if_necessary(row, odds_df, teams):
    sum_odds = row[['oddsW', 'oddsD', 'oddsL']].sum()
    if sum_odds <= 0:
        if sum_odds < 0:
            raise Exception("sum of odds is less than 0")
        else:
            team = row['team']
            print('error checking', row['team'])

            wins, draws, losses = teams.loc[teams['team']==team][['oddsW', 'oddsD', 'oddsL']].mean(axis=0).to_numpy()
                
            row[['oddsW', 'oddsD', 'oddsL']] = [np.mean(x) for x in (wins, draws, losses)]
    return row 

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


# ignores gameweeks by removing all fixtures that week and moving others up
# ok because fixtures df is nonpersistant
def hide_fixtures(df, ignore_gwks):
    for gw in sorted(ignore_gwks, reverse=True): #so updates won't mess with the ignore_gwks
        df = df.loc[df['gw']!=gw]
        df.loc[df['gw'] > gw, 'gw'] -= 1
    return df 



#pull from bootstrap static and put into filepath a replacement 
#that will function the same as vaastav's would have. need to add...:
# {'element', 'value', 'was_home', 'transfers_balance', 'round', 'selected'}
def manually_replace_vastaav_this_week(gw, filepath, previous_player_raw):
    
    past = previous_player_raw.loc[previous_player_raw['gw']==gw]
    past = past.rename(columns={'gw':'round'})
    #stupid fix because of first time doing this
    #past.loc[:,'was_home'] = past.loc[:,'was_home'].map({1.0: True, 0.0:False})

    '''fixing up metastats'''
    rows_needed = ['round', 'was_home', 'element', 'value', 'transfers_in', 'transfers_out', 'transfers_balance', 'selected','team','position']
    past_processed_meta = past[rows_needed]

    # for if home or not 
    
    url = 'https://fantasy.premierleague.com/api/fixtures/'
    response = proper_request("GET", url, headers=None)
    df_raw = pd.DataFrame(response.json())
    home_teams = df_raw.loc[df_raw['event']==gw]['team_h'].to_list()
    #past_processed_stats = past.apply(lambda x: raw_data_fixup(x, gw, home_teams), axis=1) #adds team and element type (position)
    '''fixing the stupid nans'''
    homeness = past['team'].isin(home_teams).to_list()
    past.loc[:,'was_home'] = homeness

    """Now for the week performance"""
    url = 'https://fantasy.premierleague.com/api/event/'+str(gw)+'/live/'
    response = proper_request("GET", url, headers=None)
    items = response.json()['elements']
    elements = [[x['id']] for x in items] #so we can make a df
    stats = [x['stats'] for x in items]

    element_df = pd.DataFrame(elements, columns=['element'])
    stats_df = pd.DataFrame(stats)
    stats_df = drop_columns_containing(['dreamteam'],stats_df)

    performance_stats = pd.concat([element_df, stats_df], axis=1)

    """combining them"""
    df_final = pd.merge(past_processed_meta,performance_stats, on='element')
    df_final.to_csv(filepath)
    print('put everything into the filepath, here is the head of the df and the shape\n', df_final.shape, df_final.head(), df_final.columns)


#@param: season int
#@do: get the team converter and save it to the seasons folder for future reference df ['id','name']
def get_and_save_teamconverter(season):
    century = 20
    hypenated_season = f'{century}{str(season)[:2]}-{str(season)[2:]}'
    try:
        converter = id_to_teamname_converter(hypenated_season)
    except:
        driver = webdriver.Chrome() #webdriver.Firefox()
        driver.get(f"https://www.skysports.com/premier-league-table/{hypenated_season[:4]}")
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')        
        teams = []
        for link in soup.find_all('a'):
            if link.has_attr('class') and "standing-table__cell--name-link" in link['class']:
                teams.append(link.get_text())
        teams.sort()
        converter = {i+1:team for (i,team) in enumerate(teams)}
        driver.close()
    df = pd.DataFrame(list(converter.items()), columns=['id', 'name'])
    path = DROPBOX_PATH + f"Our_Datasets/{hypenated_season}/team_converter.csv"
    df.to_csv(path)