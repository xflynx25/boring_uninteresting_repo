from calendar import week
from statistics import variance
from private_versions.Personalities import Athena_v10a, Athena_v10p
import private_versions.constants as constants
import Oracle 
import Brain 
import Agent 
import time
from Requests import get_df_from_internet
import asyncio
import pandas as pd
from datetime import datetime, timezone
from general_helpers import difference_in_days, which_time_comes_first, safe_read_csv,\
    safe_to_csv, safer_eval, get_current_day, get_deadline_difference, get_year_month_day_hour
from random import random as randomrandom
import numpy as np 
print('finished imports ')

''' #### GENERIC HELPER FUNCTIONS #### '''
# declaring transfer 
def print_transfer(name_df, chosen_transfer):
    inb = safer_eval(chosen_transfer['inbound'])
    outb = safer_eval(chosen_transfer['outbound'])
    players_in = []
    players_out = []
    print(inb, outb)
    if inb != 'set()':
        for player_in in inb:
            players_in.append(name_df.loc[name_df['element']==player_in]['name'].tolist()[0])
        for player_out in outb:
            players_out.append(name_df.loc[name_df['element']==player_out]['name'].tolist()[0])
        print('THE TRANSFER WE ARE DOING IS', '\n\n''players in= ', players_in, 'players out= ', players_out)



''' #### END GENERIC HELPER FUNCTIONS #### '''


class FPL_AI():

    def __init__(self, season, login_credentials, folder, allowed_healths, max_hit, bench_factors, value_vector,\
        num_options, quality_factor, hesitancy_dict, min_delta_dict, earliest_chip_weeks, chip_threshold_construction, chip_threshold_tailoffs,\
        wildcard_method, wildcard_model_path, player_protection, field_model_suites, keeper_model_suites, bad_players, nerf_info,\
        force_remake, when_transfer, early_transfer_aggressiveness, early_transfer_evolution_length):
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
        self.early_transfer_aggressiveness = early_transfer_aggressiveness#higher is greater
        self.early_transfer_evolution_length = early_transfer_evolution_length #number is gw when revert to normal
    
    #currently just set to make transfers randomly on the second or first night before the 
    #gameweek starts, but can be extended to whatever user would like
    def should_we_make_transfer_today(self, decision_args):
        
        '''Unpack and get decision data'''
        gw, already_transfered, has_already_pick_team_today, fix_df = decision_args
        if constants.VERBOSITY['misc']:
            print('in should we make transefer')
            print('gw is ', gw)
        
        '''decision'''
        # NEED TO ALSO MAKE SURE THAT IT IS GRABBING THE CORRECT NEXT GAMEWEEK.
        try:
            deadline_date, deadline_time = Agent.get_deadline(gw)
            days_left = get_deadline_difference(deadline_date, deadline_time)
        except:
            deadline_date, deadline_time = Agent.get_deadline(gw+1)
            days_left = get_deadline_difference(deadline_date, deadline_time)
            print('thinks its the wrong gw, only moving if next wk starts tmrw')
            return (True, True, gw+1) if days_left==1 else (False, False, gw+1)
        if constants.VERBOSITY['misc']:
            print('days left are: ', days_left)
            
        # IF ALREADY TRANSFERRED, JUST DECIDING WHETHER TO RECOMPUTE PICK_TEAM
        if already_transfered:
            print('already transferred')
            return (False, days_left <= 2 and not has_already_pick_team_today, gw) #we recompute the pick team right before gameweek, once a day only though to save compute

        # MAKE SURE ALL THE GAMES HAVE BEEN PLAYED ALREADY THIS WEEK
        if max(fix_df.loc[fix_df['gw']<gw]['day']) > get_current_day():
            print("All week games haven't finished")
            return (False, False, gw)

        # MAKE THERE IS AT LEAST 1 GAME THIS WEEK
        if fix_df.loc[fix_df['gw']==gw].shape[0] == 0:
            print("No games this week")
            # write in same amount of points as last week for the points per week info, and update ft
            human_inputs_meta = safe_read_csv(self.folder + 'human_inputs_meta.csv')
            human_inputs_meta.loc[0, f'points_gw_{gw}'] = human_inputs_meta.loc[0, f'points_gw_{gw-1}']
            human_inputs_meta.loc[0, 'ft'] = 2
            safe_to_csv(human_inputs_meta, self.folder + 'human_inputs_meta.csv')
            return (False, False, gw)

        # WEIGHTED CHOICE BASED ON WEEK TIMING AND YEAR TIMING OF GW
        EARLY_TRANSFER_AGGRESSIVENESS = self.early_transfer_aggressiveness 
        EARLY_TRANSFER_EVOLUTION_LENGTH = self.early_transfer_evolution_length
        season_phase_factor = EARLY_TRANSFER_AGGRESSIVENESS - gw // ((EARLY_TRANSFER_EVOLUTION_LENGTH / EARLY_TRANSFER_AGGRESSIVENESS))
        season_phase_factor *= (gw <= EARLY_TRANSFER_EVOLUTION_LENGTH)
        move_probability = (season_phase_factor + 1) / (season_phase_factor + max(1, days_left))
        print('gw is ', gw, '\n', 'days left is ', days_left)

        if days_left == -1 or days_left > 5: # want to wait out the international breaks somewhat
            move_probability = 0

        print('Move probability is ', move_probability)
        to_move = randomrandom() < move_probability 
        return to_move, to_move, gw


    def self_full_make_transfer_today(self, decision_args):
        make_transfer_today, do_pick_team_today, gw = self.should_we_make_transfer_today(decision_args)
        if constants.VERBOSITY['misc']:
            print('the decider results (before constants forcing): ', make_transfer_today, do_pick_team_today, gw)
        if constants.FORCE_MOVE_TODAY:
            make_transfer_today, do_pick_team_today = True, True
        if constants.PICK_TEAM_ONLY:
            make_transfer_today, do_pick_team_today = False, True
        return make_transfer_today, do_pick_team_today

    def get_weekly_point_returns(self, gw, human_inputs_meta, squad, get_raw_gw_df_wrapper):
        #   this is not done, must query the outputs file to get the proper lineup, but then can get it. 
        # get past scores, by querying the scores of the past week for current squad. 
        # and having the previous weeks recorded
        # this will be a problem during a free hit though. Exception case we deal with.
        def generic_nan_comparison(a): # ripped off online
            b = np.nan
            return (a == b) | ((a != a) & (b != b))
        weekly_point_returns = {}
        for x in range(1, gw):
            gw_points = human_inputs_meta.loc[0, f'points_gw_{x}']
            if constants.VERBOSITY['Previous_Points_Calculation_Info']:
                print(f'gw {x} is {gw_points}')
            if generic_nan_comparison(gw_points):
                if x == gw - 1:
                    if constants.VERBOSITY['Previous_Points_Calculation_Info']:
                        print('in the adjustment')
                    gw_df, stitching_a_404 = get_raw_gw_df_wrapper(constants.STRING_SEASON, x)
                    if stitching_a_404:
                        print(f'vastaav not uploaded gw{x} data yet')
                    if constants.VERBOSITY['Previous_Points_Calculation_Info']:
                        print('GW DF IS ', gw_df)
                    squadplayers = [x[0] for x in squad]
                    gw_points = gw_df.loc[gw_df['element'].isin(squadplayers)]['total_points'].sum()
                    if constants.VERBOSITY['Previous_Points_Calculation_Info']:
                        print(gw_df.loc[gw_df['element'].isin(squadplayers)][['element','total_points']])
                    # updating the inputs file
                    human_inputs_meta.loc[0, f'points_gw_{x}'] = gw_points
                    human_inputs_meta.to_csv(self.folder + 'human_inputs_meta.csv')
                else:
                    raise Exception("some earlier week does not have points gw reported")
            weekly_point_returns[x] = gw_points # we wlil assume captain score + bench subs ~ left bench score  
        # THIS IS STILL NOT ACCURATELY WORKING, SO YOU NEED TO DO THE POINTS YOURSELF. 
        # BECAUSE WE NEED TO GET CAPTAIN INFO, & BENCH SUBS, SO NEED MAYBE AN EVALUATOR 
        # FUNC ... BUT WE CAN STILL WORK BY MANUALLY INPUTTING FOR NOW . 
        return weekly_point_returns

    
    def wildcard_adjustment_if_necessary(self, gw, adjusted_chips):
        if gw <= constants.MANUAL_WILDCARD_DATES[0]:
            wc_date = constants.MANUAL_WILDCARD_DATES[0]  
        else:
            wc_date = constants.MANUAL_WILDCARD_DATES[1] 
            # if we still have recorded the playtime of the last wildcard, change (read/write)
            if adjusted_chips[0] <= constants.MANUAL_WILDCARD_DATES[0]:
                human_inputs_meta = safe_read_csv(self.folder + 'human_inputs_meta.csv')
                human_inputs_meta.loc[0, 'wc'] = constants.MANUAL_WILDCARD_DATES[1] 
                safe_to_csv(human_inputs_meta, self.folder + 'human_inputs_meta.csv')
        return wc_date

    # This is the file the user interacts with. It tells them what to input into their app, but also gives them the projected scores
    # gw, '' / pts_1 / pts_full , 15 players , c, vc, chip
    def update_human_outputs(self, necessary_meta, starters, bench_order, captain, vice_captain, this_week_chip, update_chip):
        #return name (str)
        # return None if gw not inputted yet
        def get_week_chip(df, gw):
            if df.loc[df['gw']==gw].shape[0] == 0:
                return None
            return df.loc[(df['gw']==gw)].reset_index(drop=True).loc[0, 'chip']
        def set_week_chip(df, gw, name):
            idx = min(df.loc[(df['gw']==gw)].index)
            df.loc[idx, 'chip'] = name
            return df
            
        # Get the three rows for the current week
        ## meta
        gw, name_df, new_team_players = necessary_meta
        cols = ['gw', 'title'] + [f'name_{x}' for x in range(1, 16)] + ['captain', 'vc', 'chip']
        ## first row = player names
        players = list(starters) + list(bench_order) # starters first
        players += list(set(new_team_players['element'].to_numpy()).difference(set(players))) # keeper last
        players += [captain, vice_captain] # add the captaincy
        names = [name_df.loc[name_df['element']==player]['name'].tolist()[0] for player in players] # turn into names
        biglist = [gw, ''] + names + [this_week_chip] 
        human_outputs = pd.DataFrame([biglist], columns = cols)
        print('NAMES = ', names[:-2])
        print('Captain & VC = ', names[-2:])

        ## second row = pts_1
        points_one = [np.round(new_team_players.loc[new_team_players['element']==player, 'expected_pts_N1'].to_numpy()[0], 2) for player in players[:-2]]
        rowone = [gw, 'pts_1'] + points_one + ['', '', '']
        print('rowone', rowone)
        human_outputs.loc[1, :] = rowone

        ## third row = pts_6
        #points_full = [np.round(new_team_players.loc[new_team_players['element']==player]['expected_pts_full'], 2) for player in players[:-2]]
        #owfull = [gw, 'pts_full'] + points_full + ['', '', '']
        points_full = [np.round(new_team_players.loc[new_team_players['element']==player,'expected_pts_full'].to_numpy()[0] / 6, 2) for player in players[:-2]]
        rowfull = [gw, 'ppg_6'] + points_full + ['', '', '']
        print('rowfull', rowfull)
        human_outputs.loc[2, :] = rowfull

        # Read existing and remove currently listed gw if already there
        df = safe_read_csv(self.folder + 'human_outputs.csv')
        try:
            ## if a pick team, we don't change the chip signal, read existing
            if not update_chip:
                print('not update chip')
                wk_chip = get_week_chip(df, gw)
                print('week chip', wk_chip)
                if wk_chip: #if has been set
                    print('week chip set')
                    set_week_chip(human_outputs, gw, wk_chip)
                    print('wk chip new set')

            # remove previous input for this gw, replace with current
            df = df.loc[df['gw']!=gw, :]
            df = pd.concat([df, human_outputs], axis=0).reset_index(drop=True)

        except: #fails if it is new and no 'gw'
            df = human_outputs
        print(df)

        # append rows to document
        safe_to_csv(df, self.folder + 'human_outputs.csv')



    ### SUMMARY THROUGH SUBSECTIONS ### 
    '''obtaining metadata'''
    '''decide if make transfer today''' 
    '''determine if has gone through the data already today '''
    '''grabbing the data for the current week, computing statistics'''
    '''Getting miscellaneous data (Health, Blanks, Delta/Chip Histories) '''
    '''Getting the scoreboard: player predicted performances'''
    '''Normal Transfer Selection'''
    '''Act as if doing normal transfers and get triple_captain/bench boost information'''
    '''evaluating wildcard/free hit, the pts we want to record is the improvement over current week'''
    '''decide whether or not to play chips and execute chosen path'''
    '''verify transfers and pick_team went through'''
    '''update all the tables'''        
    def make_moves(self):
        print('started ', self.email)

        try:
            IGNORE_GWKS = [] # func_to_strategically_ignore_weeks ex) for freehit & bench_boost combo or wildcard
           
            '''Step 1: Prioritize getting gw'''
            if constants.NO_CAPTCHA:
                gw, squad, sell_value, free_transfers, chip_status, weekly_point_returns = asyncio.run(Agent.current_team(self.email, self.password, self.team_id))
            else:
                gw = Agent.get_current_gw()
            
            '''Step 2: Accountant Import'''
            constants.change_global_last_gameweek(gw)
            import Accountant #it is dependent on this global gw
            name_df = Accountant.make_name_df()
            price_df = Accountant.make_and_save_price_df() # we want to save price info for future learning
            fix_df = Accountant.make_fixtures_df(self.season)[0]

            '''Step 3: Decide if make transfer today'''
            day = get_year_month_day_hour()[2]
            decision_args = [gw, Accountant.has_already_transfered(self.folder, gw), Accountant.has_already_pick_team_today(self.folder, day), fix_df]
            make_transfer_today, do_pick_team_today = self.self_full_make_transfer_today(decision_args)
            if do_pick_team_today == False:
                Accountant.log_gameweek_completion(self.folder, gw, [0, 'nothing_today'])
                return


            '''Step 4: Obtaining rest of metadata'''
            if constants.NO_CAPTCHA:
                pass
            else: # new method, use human_input.csv
                '''Getting the sell-value of all players'''
                human_inputs_players = safe_read_csv(self.folder + 'human_inputs_players.csv')
                human_inputs_players.loc[:, 'current_value'] = human_inputs_players.apply(lambda row: \
                    price_df.loc[price_df['element']==row['player']]['value'].to_numpy()[0], axis=1)
                human_inputs_players.loc[:, 'sell_value'] = human_inputs_players.apply(lambda row: \
                    (row['current_value'] if row['current_value'] < row['purchase_value'] else \
                    int(row[['current_value', 'purchase_value']].sum()) // 2), axis=1)
                squad = human_inputs_players[['player', 'sell_value']].to_numpy()
                sell_value = human_inputs_players['sell_value'].sum()
                
                '''Get meta information'''
                human_inputs_meta = safe_read_csv(self.folder + 'human_inputs_meta.csv')
                itb, free_transfers, wc, bb, tc, fh = [int(x) for x in human_inputs_meta.loc[0, ['itb', 'ft','wc','bb','tc','fh']]]
                sell_value += itb
                # ^ ft can be stored because doesn't matter ft given if transfer already made
                # chip info can easily be stored, we seed it with the end dates for each, or the played date
                adjusted_chips = []
                for chip_gw in wc, bb, tc, fh:
                    if chip_gw > gw:
                        chip_gw = 0
                    adjusted_chips.append(chip_gw)

                # dealing with fact that there are 2 wildcards
                wc_date = self.wildcard_adjustment_if_necessary(gw, adjusted_chips)
                chip_status = {
                    'freehit': adjusted_chips[3], 'bench_boost': adjusted_chips[1], 
                    'triple_captain': adjusted_chips[2], 'wildcard': (adjusted_chips[0], wc_date)
                }

                '''Past scores must be calculated'''
                func_wrapper = lambda x: Accountant.get_raw_gw_df(*x)
                weekly_point_returns = self.get_weekly_point_returns(gw, human_inputs_meta, squad, func_wrapper)
    
            if constants.VERBOSITY['squad']:
                print(gw, squad, sell_value, free_transfers, chip_status, weekly_point_returns)
            
            '''Step 5: Grabbing the data for the current week, computing statistics'''
            explored_already_today = Accountant.check_if_explored_today()
            if constants.VERBOSITY['misc']:
                print('Has explored already today? ', explored_already_today)
            current_gw_stats = safe_read_csv(constants.DROPBOX_PATH + "current_stats.csv") #speedup if multiple personalities
            if not(explored_already_today) or current_gw_stats.loc[current_gw_stats['gw']==gw].shape[0] == 0: # only run once per day
                current_gw_stats = Accountant.current_week_full_stats(self.season, {1,2,3,6}, {1,3,6}, ignore_gwks=IGNORE_GWKS)
                current_gw_stats.to_csv(constants.DROPBOX_PATH + "current_stats.csv") 
                pd.DataFrame().to_csv(constants.TRANSFER_MARKET_SAVED) #reset transfer market every time update gw_stats
            
            '''Step 6: Getting miscellaneous data (Health, Blanks, Delta/Chip Histories, fixtures_df) '''
            health_df = Accountant.make_and_save_health_df(gw)
            print('made health df')
            player_injury_penalties = Agent.injury_penalties(gw, health_df, [x[0] for x in squad])#just the elements
            blank_players = current_gw_stats.loc[current_gw_stats['FIX1_num_opponents']==0]['element'] #blank players get 0
            season_avg_delta_dict = Accountant.make_delta_dict(self.folder, self.max_hit)
            chip_threshold_dict = Accountant.make_chip_dict(self.folder, gw, self.chip_threshold_construction, self.wildcard_method)
            fixtures_df, _ = Accountant.make_fixtures_df(self.season, ignore_gwks=IGNORE_GWKS)


            '''Step 7: Getting the scoreboard = player predicted performances'''
            ### NEW MODELS ###
            force_remake = self.force_remake or not(explored_already_today)
            adjustment_information = squad, player_injury_penalties, blank_players
            tms_index = (0 if type(self.when_transfer) == str else [l.count(gw) for l in self.when_transfer].index(1)) # should be rewritten to just take into account time of week
            field_suite_tms = self.field_suite_tms[tms_index] # so now we have the specific transfer market based on the when transfer 
            keeper_suite_tms = self.keeper_suite_tms[tms_index]
            full_transfer_market = Oracle.full_transfer_creation(current_gw_stats, health_df, field_suite_tms, keeper_suite_tms, self.bad_players,\
                self.nerf_info, adjustment_information, name_df=name_df, visualize=True, force_remake=force_remake, save=False)#gw) #claim don't need to save because we can just recompute with evaluator at the end of the season 
            team_players = Oracle.make_team_players(full_transfer_market, squad)
            ### OLD MODELS ###
            """
            full_transfer_market = OldOracle.full_transfer_creation(current_gw_stats, health_df, name_df=name_df, visualize=True)
            """
            ### Update with verifcation that predictions are up to date // doing here saves time while constructing code ###
            Accountant.update_explored_today()
            ### VISUALIZING
            print('first sorted by next match')
            Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_N1', constants.NUM_PLAYERS_ON_SCOREBOARD, healthy=health_df, allowed_healths=['a','d']) 
            Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_full', constants.NUM_PLAYERS_ON_SCOREBOARD, healthy=health_df, allowed_healths=['a','d'])
            

            '''Step 8: If only doing PICK TEAM today early end'''
            if not make_transfer_today and do_pick_team_today:
                if constants.NO_CAPTCHA:
                    starters, bench_order, captain, vice_captain = Brain.pick_team(team_players, health_df)[0]
                    print('OUR INFO FOR VERIFYING !! \n\n\n starters = ', starters, 'bench order', bench_order, 'captain and vice ', captain, vice_captain, 'OUR INFO FOR VERIFYING !! \n\n\n')
                    start, on_bench = asyncio.run(Agent.get_bench_and_starters(self.email, self.password, self.team_id))
                    sub_in, sub_out = Brain.figure_out_substitution(start, on_bench, starters, set(bench_order))
                    asyncio.run(Agent.select_team(self.email, self.password, self.team_id, sub_in, sub_out, captain, vice_captain, bench_order))
                    Accountant.log_gameweek_completion(self.folder, gw, [0, 'pick_team_only']) 
                else:
                    starters, bench_order, captain, vice_captain = Brain.pick_team(team_players, health_df)[0]
                    print('OUR INFO FOR VERIFYING !! \n\n\n starters = ', starters, 'bench order', bench_order, 'captain and vice ', captain, vice_captain, 'OUR INFO FOR VERIFYING !! \n\n\n')   
                    necessary_meta = gw, name_df, team_players
                    # NEED TO MAKE SURE WE DONT OVERWRITE THE CHIP 
                    self.update_human_outputs(necessary_meta, starters, bench_order, captain, vice_captain, 'normal', False)
                    Accountant.log_gameweek_completion(self.folder, gw, [0, 'pick_team_only'])
                return 


            '''Step 9: Normal Transfer Selection'''
            choice_factors = (self.value_vector, self.hesitancy_dict, self.quality_factor, season_avg_delta_dict,self.min_delta_dict, \
                self.num_options, self.bench_factor) 
            chosen_transfer, choice_report = Brain.weekly_transfer(full_transfer_market, team_players, sell_value,\
                free_transfers, self.max_hit, choice_factors, self.player_protection, self.allowed_healths,\
                    visualize_names=True, name_df=name_df)


            '''Step 10: Act as if doing normal transfers and get triple_captain/bench boost information'''
            new_team_players = Oracle.change_team_players(full_transfer_market, team_players, chosen_transfer)
            squad_selection, captain_pts, bench_pts = Brain.pick_team(new_team_players, health_df)
            starters, bench_order, captain, vice_captain = squad_selection
            print(squad_selection, bench_order, captain_pts, bench_pts, chosen_transfer, choice_report, 'this was a ton of information from the transfer')
            print('OUR INFO FOR VERIFYING !! \n\n\n starters = ', starters, 'bench order', bench_order, 'captain and vice ', captain, vice_captain, 'OUR INFO FOR VERIFYING !! \n\n')
            print_transfer(name_df, chosen_transfer)


            '''Step 11.0: Evaluating Free Hit'''
            if not(chip_status['freehit']):
                freehit_players, freehit_pts = Brain.free_hit_team(full_transfer_market, sell_value, self.freehit_bench_factor,\
                    allowed_healths=self.allowed_healths)
                freehit_pts -= Brain.get_points(team_players.drop('expected_pts_full', axis=1), self.freehit_bench_factor)
            else:
                freehit_pts, freehit_players = .1, new_team_players

            '''Step 11.1: Evaluating Modern Wildcard'''
            if self.wildcard_method == 'modern': # these points on a 0-1 range
                datapoint = Oracle.create_wildcard_datapoint(current_gw_stats, fixtures_df, squad_selection[1], weekly_point_returns,\
                    full_transfer_market, constants.WILDCARD_2_GW_STARTS[self.season], self.bench_factor, chip_status,\
                    new_team_players['element'].to_list(), free_transfers, gw)
                model, feature_names = Oracle.load_model(self.wildcard_model_path)
                wildcard_pts = wildcard_prct = model.predict([[datapoint[x] for x in feature_names]])[0] #prct for recording
                wildcard_players = [] # just a storeholder if we don't even compute
                modern_wildcard_active = wildcard_pts >= self.chip_threshold_construction['wildcard'][0] and\
                    gw >= self.earliest_chip_weeks['wildcard'][0] and not(chip_status['wildcard'][0])

                if constants.FORCE_MODERN_WILDCARD:
                    modern_wildcard_active = True # FORCE TRUE FOR FREE WC WEEK
                print("Wildcard Prediction Probability: ", wildcard_pts)
                print("Playing wildcard? ", modern_wildcard_active)

            '''Step 11.2: Evaluating Classical Wildcard or Converting Active Wildcard'''
            if self.wildcard_method == 'classical' or (self.wildcard_method == 'modern' and modern_wildcard_active):
                print('made it into actual wildcard players')
                wildcard_players, wildcard_pts = Brain.best_wildcard_team(full_transfer_market, sell_value, self.bench_factor,\
                    free_hit=False, allowed_healths=self.allowed_healths)
                wildcard_pts -= Brain.get_points(team_players.drop('expected_pts_N1', axis=1), self.bench_factor)


            '''Step 12.0: Decide on Chips'''
            earliest_chip_weeks = {name: (x if name != 'wildcard' else x[0] if gw < constants.WILDCARD_2_GW_STARTS[self.season] else x[1]) for (name,x) in self.earliest_chip_weeks.items() }
            this_week_chip = Brain.play_chips_or_no(gw, chip_status, chip_threshold_dict, wildcard_pts, freehit_pts, captain_pts,\
            bench_pts, earliest_chip_weeks, self.chip_threshold_tailoffs)
            print('original chip choice was ', this_week_chip)
            
            '''Step 13: All info computed, (virtual) front-end section'''
            if constants.NO_CAPTCHA: # old, can't expect it to return to this.
                '''Online-Execution Helper Objects'''
                login_info = (self.email, self.password, self.team_id)
                brain_transfer_help = lambda x,y: Brain.match_positions(x, y, full_transfer_market)
                brain_substitution_help = lambda x,y,z,a: Brain.figure_out_substitution(x,y,z,a)
                brain_pick_team_help = lambda x:\
                    Brain.pick_team(full_transfer_market.loc[full_transfer_market['element'].isin(x)], health_df)
                potential_teams = new_team_players, wildcard_players, freehit_players

                '''manually execute the transfer'''
                verify_info = Agent.execute_chip(this_week_chip, chosen_transfer, squad_selection, potential_teams,\
                        login_info, brain_transfer_help, brain_substitution_help, brain_pick_team_help)
                '''verify transfers and pick_team went through'''
                all_success = asyncio.run(Agent.verify_front_end_successful(login_info, verify_info + [squad]))
                if all_success:
                    transfer_info = [verify_info[-1],this_week_chip]
                    Accountant.log_gameweek_completion(self.folder, gw, transfer_info)
                    if this_week_chip != 'normal': # this method no longer working either because of gmail 2FA
                        no_input_gmail = constants.NOTIFICATION_SENDER_GMAIL == "" or constants.NOTIFICATION_SENDER_PASSWORD == "" #still need to catch more problems
                        send_email, send_password = [(self.email, self.password) if no_input_gmail else (constants.NOTIFICATION_SENDER_GMAIL,constants.NOTIFICATION_SENDER_PASSWORD)][0]
                        Agent.notify_human(self.email, send_email, send_password, constants.NOTIFICATION_RECEIVER_EMAIL, gw, this_week_chip)
            else:
                '''Logging'''
                # (either way we organize this and the rest of the logging, there could be problems if we crash in between - this way is less catastrophic, we will just not move this week if the case'''
                action = len(safer_eval(chosen_transfer['inbound'])) #num transfers for nochip
                transfer_info = [action, this_week_chip]
                Accountant.log_gameweek_completion(self.folder, gw, transfer_info)    

                '''update human_outputs.csv'''
                # first adjust if chip
                if this_week_chip == 'wildcard':
                    free_transfers = 0 # reset ft at 1
                    new_team_players = wildcard_players
                    starters, bench_order, captain, vice_captain = Brain.pick_team(new_team_players, health_df)[0]
                    print('WILDCARD !!!! BEING PLAYED !!!')
                elif this_week_chip == 'freehit':
                    free_transfers = 0 # reset ft at 1
                    new_team_players = freehit_players
                    starters, bench_order, captain, vice_captain = Brain.pick_team(new_team_players, health_df)[0]
                    print('FREEHIT !!!! BEING PLAYED !!!')
                necessary_meta = gw, name_df, new_team_players 
                self.update_human_outputs(necessary_meta, starters, bench_order, captain, vice_captain, this_week_chip, True)
            
                '''update the human_inputs_players.csv, and human_inputs_meta.csv, accordingly ''' 
                # deal with the chips
                if this_week_chip == 'wildcard':
                    human_inputs_meta.loc[0, 'wc'] = gw
                elif this_week_chip == 'bench_boost':
                    human_inputs_meta.loc[0, 'bb'] = gw
                elif this_week_chip == 'triple_captain':
                    human_inputs_meta.loc[0, 'tc'] = gw
                elif this_week_chip == 'freehit':
                    human_inputs_meta.loc[0, 'fh'] = gw

                if this_week_chip == 'freehit': # only transfers changes for freehit  
                    free_transfers = 1  
                    human_inputs_meta.loc[0, 'ft'] = free_transfers      
                    safe_to_csv(human_inputs_meta, self.folder + 'human_inputs_meta.csv')       
                else:
                    # update new players, and their proper prices
                    # old guys keep their purchase price, new ones are set to the current sell price
                    remaining = human_inputs_players.loc[human_inputs_players['player'].isin(new_team_players['element'])][['player', 'purchase_value']]
                    newelements = set(new_team_players['element'].to_numpy()).difference(set(human_inputs_players['player'].to_numpy()))
                    newguys = new_team_players.loc[new_team_players['element'].isin(newelements)][['element', 'value']]
                    newguys.columns = ['player', 'purchase_value']
                    new_players_and_prices = pd.concat([remaining, newguys], axis=0).reset_index(drop=True)
                    print(new_players_and_prices)
                    safe_to_csv(new_players_and_prices, self.folder + 'human_inputs_players.csv')

                    # update new ft, and itb, if chip used adjust
                    free_transfers = min(2, max(1, free_transfers - action + 1)) #action is num transfers
                    itb = sell_value - new_team_players['value'].sum()
                    human_inputs_meta.loc[0, 'itb'] = itb
                    human_inputs_meta.loc[0, 'ft'] = free_transfers
                    print(human_inputs_meta)
                    safe_to_csv(human_inputs_meta, self.folder + 'human_inputs_meta.csv')
                    
                    

            '''Step 14: Update Personal Tables'''        
            Accountant.update_delta_db(self.folder, choice_report)
            if self.wildcard_method == 'modern':
                wildcard_pts = wildcard_prct
            Accountant.update_chip_db(self.folder, gw, wildcard_pts, freehit_pts, captain_pts, bench_pts)
        
        # recording the exceptions to help me with asynchronous debugging
        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stdout)
            #no_input_gmail = constants.NOTIFICATION_SENDER_GMAIL == "" or constants.NOTIFICATION_SENDER_PASSWORD == "" #still need to catch more problems
            #send_email, send_password = [(self.email, self.password) if no_input_gmail else (constants.NOTIFICATION_SENDER_GMAIL,constants.NOTIFICATION_SENDER_PASSWORD)][0]
            try:
                gw += 0
            except: # make sure we send errors even when we die before the error starts
                gw = -1
            # notify human turned off because it does not work right now. 
            #Agent.notify_human(self.email, send_email, send_password, constants.NOTIFICATION_RECEIVER_EMAIL, gw, 'THERE WAS AN EXCEPTION')
            date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            logging_file = constants.DROPBOX_PATH + 'log_exceptions.txt'
            with open(logging_file, "a+") as f:
                
                f.write(f"Date: {date} -- " + str(e) + '\n')

 


if __name__ == '__main__':
    from private_versions.Personalities import personalities_to_run
    for pers in personalities_to_run:
        ai = FPL_AI(**pers)
        ai.make_moves()