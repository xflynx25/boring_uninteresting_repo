import constants
#import Accountant
import Oracle 
import Brain 
import Agent 
import time
import asyncio
import pandas as pd
from datetime import datetime
from general_helpers import difference_in_days, which_time_comes_first
from random import random as randomrandom
#import importlib 
#importlib.reload(Agent)

''' #### GENERIC HELPER FUNCTIONS #### '''
# declaring transfer 
def print_transfer(name_df, chosen_transfer):
    inb = chosen_transfer['inbound'][0]
    outb = chosen_transfer['outbound'][0]
    players_in = []
    players_out = []
    print(inb, outb)
    if inb != [{}]:
        for player_in in inb:
            players_in.append(name_df.loc[name_df['element']==player_in]['name'].tolist()[0])
        for player_out in outb:
            players_out.append(name_df.loc[name_df['element']==player_out]['name'].tolist()[0])
        print('THE TRANSFER WE ARE DOING IS', '\n\n''players in= ', players_in, 'players out= ', players_out)

''' #### END GENERIC HELPER FUNCTIONS #### '''


class FPL_AI():

    def __init__(self, season, login_credentials, folder, allowed_healths, max_hit, bench_factors, value_vector,\
        num_options, quality_factor, hesitancy_dict, min_delta_dict, earliest_chip_weeks, chip_threshold_construction, chip_threshold_tailoff,\
        player_protection, field_model_suites, keeper_model_suites, bad_players, nerf_info, force_remake):
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
        self.chip_threshold_tailoff = chip_threshold_tailoff #float - 0-1, .2 tails off around gw34, .1 around gw30, .05 around gw20, slow tailoff
        self.player_protection = player_protection #int, you will not sell any player in the top __ predicted for these next few weeks.
        self.field_model_suites = field_model_suites #list of str referencing folders 
        self.keeper_model_suites = keeper_model_suites #list of str referencing folders 
        self.bad_players = bad_players
        self.nerf_info = nerf_info
        self.force_remake = force_remake #if not all your people using the same modelling, force_remake should be true 
    
    #currently just set to make transfers randomly on the second or first night before the 
    #gameweek starts, but can be extended to whatever user would like
    def should_we_make_transfer_today(self, decision_args):
        def get_deadline_difference(gw):
            deadline_date, deadline_time = Agent.get_deadline(gw)
            current_date = [int(x) for x in datetime.utcnow().strftime('%Y-%m-%d').split('-')]
            day_diff =  difference_in_days(current_date, deadline_date)
            if day_diff > 0:
                return day_diff
            else: #check time diff 
                current_time =  [int(x) for x in datetime.utcnow().strftime('%H:%M:%S').split(':')]
                if which_time_comes_first(current_time, deadline_time) == 0: 
                    return 0
                else:
                    return -1

        gw,_ = decision_args

        #decision
        # NEED TO ALSO MAKE SURE THAT IT IS GRABBING THE CORRECT NEXT GAMEWEEK.
        try:
            days_left = get_deadline_difference(gw)
        except:
            days_left = get_deadline_difference(gw+1)
            print('thinks its the wrong gw, only moving if next wk starts tmrw')
            return (True, gw+1) if days_left==1 else (False, gw+1)

        move_probability = 1/max(1, days_left)
        if days_left > 2 or days_left == -1: 
            move_probability = 0
        return randomrandom() < move_probability, gw



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
        IGNORE_GWKS = [] # func_to_strategically_ignore_weeks ex) for freehit & bench_boost combo or wildcard

        '''obtaining metadata'''
        start = time.time()
        gw, squad, sell_value, free_transfers, chip_status = asyncio.run(Agent.current_team(self.email, self.password, self.team_id))
        constants.change_global_last_gameweek(gw)
        import Accountant #it is dependent on this global gw
        name_df = Accountant.make_name_df()


        '''decide if make transfer today''' 
        if Accountant.already_made_moves(self.folder, gw):
            return 
        if asyncio.run(Agent.has_made_transfers_already(self.email, self.password, self.team_id)):
            raise Exception("Have already done transfers for this week but not recorded")
        decision_args = [gw, None]
        make_transfer_today, gw = self.should_we_make_transfer_today(decision_args)
        if not make_transfer_today:
            return 
        print('we are making a transfer')


        ''' determine if has gone through the data already today '''
        explored_already_today = Accountant.check_if_explored_today()

        '''grabbing the data for the current week, computing statistics'''
        current_gw_stats = pd.read_csv(constants.DROPBOX_PATH + "current_stats.csv", index_col=0) #speedup if multiple personalities
        if current_gw_stats.loc[current_gw_stats['gw']==gw].shape[0] == 0 or not(explored_already_today): # only run once per day
            current_gw_stats = Accountant.current_week_full_stats(self.season, {1,2,3,6}, {1,3,6}, ignore_gwks=IGNORE_GWKS)
            current_gw_stats.to_csv(constants.DROPBOX_PATH + "current_stats.csv") 
            pd.DataFrame().to_csv(constants.TRANSFER_MARKET_SAVED) #reset transfer market every time update gw_stats
        
        
        '''Getting miscellaneous data (Health, Blanks, Delta/Chip Histories) '''
        health_df = Accountant.make_health_df()
        #health_df.to_csv(r"C:\Users\JFlyn\Dropbox (MIT)\FPL_Datasets\health_df.csv") #just for working off wifi
        player_injury_penalties = asyncio.run(Agent.injury_penalties(gw, health_df, [x[0] for x in squad]))#just the elements
        blank_players = current_gw_stats.loc[current_gw_stats['FIX1_num_opponents']==0]['element'] #blank players get 0
        season_avg_delta_dict = Accountant.make_delta_dict(self.folder, self.max_hit)
        chip_threshold_dict = Accountant.make_chip_dict(self.folder, gw, self.chip_threshold_construction)


        '''Getting the scoreboard: player predicted performances'''
        ### NEW MODELS ###
        force_remake = self.force_remake or not(explored_already_today)
        adjustment_information = squad, player_injury_penalties, blank_players
        full_transfer_market = Oracle.full_transfer_creation(current_gw_stats, health_df, self.field_model_suites, self.keeper_model_suites, self.bad_players,\
            self.nerf_info, adjustment_information, name_df=name_df, visualize=False, force_remake=force_remake, save=gw)
        team_players = Oracle.make_team_players(full_transfer_market, squad)
        ### OLD MODELS ###
        """
        full_transfer_market = OldOracle.full_transfer_creation(current_gw_stats, health_df, name_df=name_df, visualize=True)
        """
        ### Update with verifcation that predictions are up to date // doing here saves time while constructing code ###
        Accountant.update_explored_today()
        ### VISUALIZING
        print('first sorted by next match')
        Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_N1', 35, healthy=health_df, allowed_healths=['a','d']) 
        Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_full', 35, healthy=health_df, allowed_healths=['a','d'])
        

        '''Normal Transfer Selection'''
        choice_factors = (self.value_vector, self.hesitancy_dict, self.quality_factor, season_avg_delta_dict,self.min_delta_dict, \
            self.num_options, self.bench_factor) 
        chosen_transfer, choice_report = Brain.weekly_transfer(full_transfer_market, team_players, sell_value,\
            free_transfers, self.max_hit, choice_factors, self.player_protection, self.allowed_healths,\
                visualize_names=True, name_df=name_df)


        '''Act as if doing normal transfers and get triple_captain/bench boost information'''
        new_team_players = Oracle.change_team_players(full_transfer_market, team_players, chosen_transfer)
        squad_selection, captain_pts, bench_pts = Brain.pick_team(new_team_players, health_df)
        starters, bench_order, captain, vice_captain = squad_selection
        print(squad_selection, bench_order, captain_pts, bench_pts, chosen_transfer, choice_report, 'this was a ton of information from the transfer')
        print('OUR INFO FOR VERIFYING !! \n\n\n starters = ', starters, 'bench order', bench_order, 'captain and vice ', captain, vice_captain, 'OUR INFO FOR VERIFYING !! \n\n\n')
        print_transfer(name_df, chosen_transfer)              


        '''evaluating wildcard/free hit, the pts we want to record is the improvement over current week'''
        wildcard_players, wildcard_pts = Brain.best_wildcard_team(full_transfer_market, sell_value, self.bench_factor,\
             free_hit=False, allowed_healths=self.allowed_healths, visualize_names=True, name_df=name_df)
        freehit_players, freehit_pts = Brain.free_hit_team(full_transfer_market, sell_value, self.freehit_bench_factor,\
            allowed_healths=self.allowed_healths)
        wildcard_pts -= Brain.get_points(team_players.drop('expected_pts_N1', axis=1), self.bench_factor)
        freehit_pts -= Brain.get_points(team_players.drop('expected_pts_full', axis=1), self.freehit_bench_factor)


        '''decide whether or not to play chips and execute chosen path'''
        this_week_chip = Brain.play_chips_or_no(gw, chip_status, chip_threshold_dict, wildcard_pts, freehit_pts, captain_pts,\
           bench_pts, self.earliest_chip_weeks, self.chip_threshold_tailoff)
        print('original chip choice was ', this_week_chip)

        login_info = (self.email, self.password, self.team_id)
        brain_transfer_help = lambda x,y: Brain.match_positions(x, y, full_transfer_market)
        brain_substitution_help = lambda x,y,z,a: Brain.figure_out_substitution(x,y,z,a)
        brain_pick_team_help = lambda x:\
            Brain.pick_team(full_transfer_market.loc[full_transfer_market['element'].isin(x)], health_df)
        potential_teams = new_team_players, wildcard_players, freehit_players 
        verify_info = Agent.execute_chip(this_week_chip, chosen_transfer, squad_selection, potential_teams,\
                login_info, brain_transfer_help, brain_substitution_help, brain_pick_team_help)


        '''verify transfers and pick_team went through'''
        all_success = asyncio.run(Agent.verify_front_end_successful(login_info, verify_info + [squad]))


        '''update all the tables'''        
        Accountant.update_delta_db(self.folder, choice_report)
        Accountant.update_chip_db(self.folder, gw, wildcard_pts, freehit_pts, captain_pts, bench_pts)
        if all_success:
            transfer_info = [verify_info[-1],this_week_chip]
            Accountant.log_gameweek_completion(self.folder, gw, transfer_info)
            if this_week_chip != 'normal':
                no_input_gmail = constants.NOTIFICATION_SENDER_GMAIL == "" or constants.NOTIFICATION_SENDER_PASSWORD == "" #still need to catch more problems
                send_email, send_password = [(self.email, self.password) if no_input_gmail else (constants.NOTIFICATION_SENDER_GMAIL,constants.NOTIFICATION_SENDER_PASSWORD)][0]
                Agent.notify_human(self.email, send_email, send_password, constants.NOTIFICATION_RECEIVER_EMAIL, gw, this_week_chip)


 
        
if __name__ == '__main__':
    from Personalities import personalities_to_run
    for pers in personalities_to_run:
        ai = FPL_AI(**pers)
        ai.make_moves()