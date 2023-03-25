"""
To optimize this for nn use we should eliminate the saving and reading and rather store these dfs in memory
    ` Probably low time compared to bottleneck (wildcard/freehit computation)

"""


import constants as constants
import Oracle 
import Brain 
import pandas as pd

# produce the specific transfer market for the personality & team situation
#@param: suit_tms: list of season long dfs (to avoid reading in every week)
def overseer_simulator_get_transfer_market(when_transfer, gw, field_suite_tms_list, keeper_suite_tms_list, bad_players, nerf_info,\
    adjustment_information, name_df):
    tms_index = (0 if type(when_transfer) == str else [l.count(gw) for l in when_transfer].index(1))
    field_suite_tms = field_suite_tms_list[tms_index] # so now we have the specific transfer market based on the when transfer 
    keeper_suite_tms = keeper_suite_tms_list[tms_index]
    print(field_suite_tms, keeper_suite_tms)
    
    dffieldplayers = Oracle.avg_transfer_markets([df.loc[df['gw']==gw].drop('gw', axis=1) for df in field_suite_tms if (gw in df['gw'].unique()) ]) #end part например early and late only want one
    
    #correct for how I did not put a model in for predicting full for the single gw predict for is dgw
    if len(field_suite_tms) > 1 and min([df['expected_pts_full'].max() for df in field_suite_tms]) == 0: #number of dgw in the version where we precompute everything
        dffieldplayers.loc[:, 'expected_pts_full'] = dffieldplayers.apply(lambda x: x['expected_pts_full']*len(field_suite_tms) / (len(field_suite_tms)-1), axis=1)
    dfkeepers = Oracle.avg_transfer_markets([df.loc[df['gw']==gw].drop('gw', axis=1) for df in keeper_suite_tms if (gw in df['gw'].unique()) ])
    ### COMBINING THE TWO ###
    full_transfer_market = pd.concat([dfkeepers, dffieldplayers],axis=0)

    '''Nerfing and Eliminating Players'''
    nerf_elements, nerf_scales = [x[0] for x in nerf_info], [x[1] for x in nerf_info]
    full_transfer_market = Oracle.nerf_players(full_transfer_market, nerf_elements, name_df, nerf_scales, visualize=False)
    full_transfer_market = Oracle.eliminate_players(full_transfer_market, bad_players, name_df).reset_index(drop=True)

    '''adjusting player prices in transfer market, making pd team representation'''
    squad, player_injury_penalties, blank_players = adjustment_information 
    price_adjusted_transfer_market = Oracle.adjust_team_player_prices(full_transfer_market, squad)
    point_adjusted_transfer_market = Oracle.adjust_team_expected_pts_full(price_adjusted_transfer_market, player_injury_penalties)
    full_transfer_market = Oracle.adjust_blank_gameweek_pts_N1(point_adjusted_transfer_market, blank_players)

    return full_transfer_market


class FPL_AI():

    def __init__(self, season, login_credentials, folder, allowed_healths, max_hit, bench_factors, value_vector,\
        num_options, quality_factor, hesitancy_dict, min_delta_dict, earliest_chip_weeks, chip_threshold_construction, chip_threshold_tailoffs,\
        wildcard_method, wildcard_model_path, player_protection, field_model_suites, keeper_model_suites, bad_players, nerf_info,\
        force_remake, when_transfer):
        self.season = season
        self.email = login_credentials[0] #str
        self.password = login_credentials[1] #str
        self.team_id = login_credentials[2] #int
        self.folder = folder #str filepath, with / at the end
        self.allowed_healths = allowed_healths #list - i.e.['a','d']
        self.max_hit = max_hit #int, positive
        self.bench_factor = bench_factors[0] #float - 0-1
        self.freehit_bench_factor = bench_factors[1] #float - 0-1
        self.value_vector = value_vector #listlike - [worth, next_match_delta, full_match_delta], worth tends to have least influence for same value 
        self.num_options = num_options #int - how many to consider in transfer search single n
        self.quality_factor = quality_factor #float - multiplier for relative quality of rank-n top transfer
        self.hesitancy_dict = hesitancy_dict #dict - keys=ft values=dict with keys=num_transfers, vals = 0-1 float
        self.min_delta_dict = min_delta_dict #dict -keys = ft, values = integer that we require week beats
        self.earliest_chip_weeks = earliest_chip_weeks # dict with keys= str of chip name, vals are int-gw (wildcard is tuple)
        self.chip_threshold_construction = chip_threshold_construction #{chip: (%of top, std_deviations, choice_method)} #where choice method is either 'max', 'min', 'avg'
        self.chip_threshold_tailoffs = chip_threshold_tailoffs #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        self.wildcard_method = wildcard_method
        self.wildcard_model_path = wildcard_model_path
        self.player_protection = player_protection #int, you will not sell any player in the top __ predicted for these next few weeks.
        self.field_suite_tms = field_model_suites #list of df with the season long data for tms
        self.keeper_suite_tms = keeper_model_suites #list of df with the season long data for tms
        self.bad_players = bad_players
        self.nerf_info = nerf_info
        self.force_remake = force_remake #if not all your people using the same modelling, force_remake should be true 
        self.when_transfer = when_transfer
    


    ### SUMMARY THROUGH SUBSECTIONS ### 
    '''Getting miscellaneous data (Health, Blanks, Delta/Chip Histories) '''
    '''Getting the scoreboard: player predicted performances'''
    '''Normal Transfer Selection'''
    '''Act as if doing normal transfers and get triple_captain/bench boost information'''
    '''evaluating wildcard/free hit, the pts we want to record is the improvement over current week'''
    '''decide whether or not to play chips and execute chosen path'''
    '''verify transfers and pick_team went through'''
    '''update all the tables'''        
    # @param: weekly_point_returns is a dict of the scores each gameweek
    def make_moves(self, current_gw_stats, gw, squad, sell_value, free_transfers, chip_status, weekly_point_returns):
        IGNORE_GWKS = [] # func_to_strategically_ignore_weeks ex) for freehit & bench_boost combo or wildcard
        
        '''Getting miscellaneous data (Health, Blanks, Delta/Chip Histories, fixtures) '''
        constants.change_global_last_gameweek(gw)
        import Accountant #it is dependent on this global gw
        
        
        
        health_df = pd.DataFrame([[x, 'a'] for x in current_gw_stats['element'].unique()], columns=['element', 'status']) #GENERIC, CAN BE OPTIMIZED
        player_injury_penalties = {x:1 for x in current_gw_stats['element'].unique()} #GENERIC, CAN BE OPTIMIZED
        
        blank_players = current_gw_stats.loc[current_gw_stats['FIX1_num_opponents']==0]['element'] #blank players get 0
        
        season_avg_delta_dict = Accountant.make_delta_dict(self.folder, self.max_hit)
        chip_threshold_dict = Accountant.make_chip_dict(self.folder, gw, self.chip_threshold_construction, self.wildcard_method)
        fixtures_df = pd.read_csv(constants.DROPBOX_PATH + f"Our_Datasets/{self.season}/fix_df.csv", index_col=0)
        
        name_df = current_gw_stats[['element','name']] #GENERIC, CAN BE OPTIMIZED


        '''Getting the scoreboard: player predicted performances'''
        ### NEW MODELS ###
        adjustment_information = squad, player_injury_penalties, blank_players
        full_transfer_market = overseer_simulator_get_transfer_market(self.when_transfer, gw, self.field_suite_tms, self.keeper_suite_tms,\
            self.bad_players, self.nerf_info, adjustment_information, name_df) 
        team_players = Oracle.make_team_players(full_transfer_market, squad)


        ''' For season 2021 and gw2 we want to not transfer because a blank in gw threw everything off'''
        if gw == 2 and self.season == '2020-21':
            squad_selection = Brain.pick_team(team_players, health_df, with_keeper_bench=True)[0]
            starters, bench_ordered, captain, vcaptain = squad_selection
            return [set(), set()], 'normal', captain, vcaptain, bench_ordered
            # this still leaves us with man u / brighton & their opponents benched, but c'est la


        '''Normal Transfer Selection'''
        choice_factors = (self.value_vector, self.hesitancy_dict, self.quality_factor, season_avg_delta_dict,self.min_delta_dict, \
            self.num_options, self.bench_factor) 
        chosen_transfer, choice_report = Brain.weekly_transfer(full_transfer_market, team_players, sell_value,\
            free_transfers, self.max_hit, choice_factors, self.player_protection, self.allowed_healths)


        '''Act as if doing normal transfers and get triple_captain/bench boost information'''
        #print('chosen transfer:\n', chosen_transfer)
        new_team_players = Oracle.change_team_players(full_transfer_market, team_players, chosen_transfer)
        squad_selection, captain_pts, bench_pts = Brain.pick_team(new_team_players, health_df, with_keeper_bench=True)


        '''evaluating wildcard/free hit, the pts we want to record is the improvement over current week'''
        if self.wildcard_method == 'classical':
            wildcard_players, wildcard_pts = Brain.best_wildcard_team(full_transfer_market, sell_value, self.bench_factor,\
                free_hit=False, allowed_healths=self.allowed_healths)
            wildcard_pts -= Brain.get_points(team_players.drop('expected_pts_N1', axis=1), self.bench_factor)
        if not(chip_status['freehit']):
            freehit_players, freehit_pts = Brain.free_hit_team(full_transfer_market, sell_value, self.freehit_bench_factor,\
                allowed_healths=self.allowed_healths)
            freehit_pts -= Brain.get_points(team_players.drop('expected_pts_full', axis=1), self.freehit_bench_factor)
        else:
            freehit_pts = .1


        if self.wildcard_method == 'modern': # these points on a 0-1 range
            datapoint = Oracle.create_wildcard_datapoint(current_gw_stats, fixtures_df, squad_selection[1], weekly_point_returns,\
                full_transfer_market, constants.WILDCARD_2_GW_STARTS[self.season], self.bench_factor, chip_status,\
                new_team_players['element'].to_list(), free_transfers, gw)
            model, feature_names = Oracle.load_model(self.wildcard_model_path)
            wildcard_pts = model.predict([[datapoint[x] for x in feature_names]])[0]
        


        '''decide whether or not to play chips and execute chosen path'''
        brain_pick_team_help = lambda x:\
            Brain.pick_team(full_transfer_market.loc[full_transfer_market['element'].isin(x)], health_df, with_keeper_bench=True)
        
        earliest_chip_weeks = {name: (x if name != 'wildcard' else x[0] if gw < constants.WILDCARD_2_GW_STARTS[self.season] else x[1]) for (name,x) in self.earliest_chip_weeks.items() }
        this_week_chip = Brain.play_chips_or_no(gw, chip_status, chip_threshold_dict, wildcard_pts, freehit_pts, captain_pts,\
           bench_pts, earliest_chip_weeks, self.chip_threshold_tailoffs)
        if this_week_chip == 'wildcard':
            wildcard_players, _ = Brain.best_wildcard_team(full_transfer_market, sell_value, self.bench_factor,\
                free_hit=False, allowed_healths=self.allowed_healths)
            squad_selection = brain_pick_team_help(wildcard_players['element'].to_list())[0]
        elif this_week_chip == 'freehit':
            squad_selection = brain_pick_team_help(freehit_players['element'].to_list())[0]
            
        starters, bench_ordered, captain, vcaptain = squad_selection
        


        '''update all the tables''' 
        Accountant.update_delta_db(self.folder, choice_report)
        Accountant.update_chip_db(self.folder, gw, wildcard_pts, freehit_pts, captain_pts, bench_pts)

        ''' get the transfer ''' 
        before = {x[0] for x in squad}
        after = set(starters+bench_ordered)
        inb = after.difference(before)
        outb = before.difference(after)
        transfer = (inb, outb)
        
        return transfer, this_week_chip, captain, vcaptain, bench_ordered
        