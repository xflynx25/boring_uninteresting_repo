from Personalities import Athena_v10a, Athena_v10p
from Accountant import already_made_moves
import constants
#import Accountant
import Oracle 
import Brain 
import Agent 
import time
from Requests import get_df_from_internet
import asyncio
import pandas as pd
from datetime import datetime
from general_helpers import difference_in_days, which_time_comes_first, safe_read_csv, safer_eval, get_current_day
from random import random as randomrandom

# to get around this urllib error certificate_verify_failed
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#import importlib 
#importlib.reload(Agent)
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

        gw, already_transfered, fix_df = decision_args
        
        '''decision'''
        # NEED TO ALSO MAKE SURE THAT IT IS GRABBING THE CORRECT NEXT GAMEWEEK.
        try:
            days_left = get_deadline_difference(gw)
        except:
            days_left = get_deadline_difference(gw+1)
            print('thinks its the wrong gw, only moving if next wk starts tmrw')
            return (True, True, gw+1) if days_left==1 else (False, False, gw+1)
            
        # IF ALREADY TRANSFERRED, JUST DECIDING WHETHER TO RECOMPUTE PICK_TEAM
        if already_transfered:
            return (False, days_left <= 2, gw) #we recompute in 2 days preceding deadline in case miss one


        # MAKE SURE ALL THE GAMES HAVE BEEN PLAYED ALREADY THIS WEEK
        if max(fix_df.loc[fix_df['gw']==gw-1]['day']) > get_current_day():
            print("All week games haven't finished")
            return (False, False, gw)

        """
        # ALSO MAKE SURE VAASTAV HAS DATA FOR THIS WEEK, (probably don't need this as we have manual vastaav replace but will keep like this for now)
        try: 
            wk_stats = get_df_from_internet(constants.VASTAAV_ROOT + self.season + '/gws/gw' + str(gw-1) + '.csv')
        except:
            print("Vastaav hasn't uploaded the stats")
            #raise Exception("Vastaav hasn't uploaded the stats")
            return (False, False, gw) """


        # WEIGHTED CHOICE BASED ON WEEK TIMING AND YEAR TIMING OF GW
        EARLY_TRANSFER_AGGRESSIVENESS = self.early_transfer_aggressiveness 
        EARLY_TRANSFER_EVOLUTION_LENGTH = self.early_transfer_evolution_length
        season_phase_factor = EARLY_TRANSFER_AGGRESSIVENESS - gw // ((EARLY_TRANSFER_EVOLUTION_LENGTH / EARLY_TRANSFER_AGGRESSIVENESS))
        season_phase_factor *= (gw <= EARLY_TRANSFER_EVOLUTION_LENGTH)
        move_probability = (season_phase_factor + 1) / (season_phase_factor + max(1, days_left))

        if days_left == -1 or days_left > 5: # want to wait out the international breaks somewhat
            move_probability = 0
        print('Move probability is ', move_probability)
        to_move = randomrandom() < move_probability
        return to_move, to_move, gw



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
        try:
            IGNORE_GWKS = [] # func_to_strategically_ignore_weeks ex) for freehit & bench_boost combo or wildcard
           
            '''obtaining metadata'''
            print('started ', self.email)
            if constants.NO_CAPTCHA:
                gw, squad, sell_value, free_transfers, chip_status, weekly_point_returns = asyncio.run(Agent.current_team(self.email, self.password, self.team_id))
            else: # new method, use human_input.csv
                pass
                '''get the same values as above, might need 2 files'''
                '''1) human_inputs_players.csv = the squad, the buy and current prices of them'''
                    # sell value computed from this
                '''2) meta'''
                    # gw, ft, status_wc, status_fh, status_bb, status_tc, points_gw_{x} for x in range(1, 39)
                    #read and convert to the api dict that we need per before values.
            print(chip_status)
            print(squad)
            
            constants.change_global_last_gameweek(gw)
            import Accountant #it is dependent on this global gw
            name_df = Accountant.make_name_df()
            Accountant.make_and_save_price_df() # we want to save price info for future learning
        

            '''decide if make transfer today'''
            already_transfered = False
            fix_df = Accountant.make_fixtures_df(self.season)[0]
            if Accountant.already_made_moves(self.folder, gw):
                already_transfered = True
            elif asyncio.run(Agent.has_made_transfers_already(self.email, self.password, self.team_id)):
                raise Exception("Have already done transfers for this week but not recorded")

            decision_args = [gw, already_transfered, fix_df]
            make_transfer_today, do_pick_team_today, gw = self.should_we_make_transfer_today(decision_args)
            print('the decider results: ', make_transfer_today, do_pick_team_today, gw)
            if do_pick_team_today == False:
                return


            ''' determine if has gone through the data already today '''
            explored_already_today = Accountant.check_if_explored_today()
            print('Has explored already today? ', explored_already_today)

            '''grabbing the data for the current week, computing statistics'''
            current_gw_stats = safe_read_csv(constants.DROPBOX_PATH + "current_stats.csv") #speedup if multiple personalities
            if not(explored_already_today) or current_gw_stats.loc[current_gw_stats['gw']==gw].shape[0] == 0: # only run once per day
                current_gw_stats = Accountant.current_week_full_stats(self.season, {1,2,3,6}, {1,3,6}, ignore_gwks=IGNORE_GWKS)
                current_gw_stats.to_csv(constants.DROPBOX_PATH + "current_stats.csv") 
                pd.DataFrame().to_csv(constants.TRANSFER_MARKET_SAVED) #reset transfer market every time update gw_stats
            
            '''Getting miscellaneous data (Health, Blanks, Delta/Chip Histories, fixtures_df) '''
            health_df = Accountant.make_and_save_health_df(gw)
            print('made health df')
            player_injury_penalties = asyncio.run(Agent.injury_penalties(gw, health_df, [x[0] for x in squad]))#just the elements
            blank_players = current_gw_stats.loc[current_gw_stats['FIX1_num_opponents']==0]['element'] #blank players get 0
            season_avg_delta_dict = Accountant.make_delta_dict(self.folder, self.max_hit)
            chip_threshold_dict = Accountant.make_chip_dict(self.folder, gw, self.chip_threshold_construction, self.wildcard_method)
            fixtures_df, _ = Accountant.make_fixtures_df(self.season, ignore_gwks=IGNORE_GWKS)


            '''Getting the scoreboard: player predicted performances'''
            ### NEW MODELS ###
            force_remake = self.force_remake or not(explored_already_today)
            adjustment_information = squad, player_injury_penalties, blank_players
            tms_index = (0 if type(self.when_transfer) == str else [l.count(gw) for l in self.when_transfer].index(1))
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
            Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_N1', 35, healthy=health_df, allowed_healths=['a','d']) 
            Oracle.visualize_top_transfer_market(full_transfer_market, name_df, 'expected_pts_full', 35, healthy=health_df, allowed_healths=['a','d'])
            

            '''If only doing pick team today early end'''

            if not make_transfer_today and do_pick_team_today:
                starters, bench_order, captain, vice_captain = Brain.pick_team(team_players, health_df)[0]
                print('OUR INFO FOR VERIFYING !! \n\n\n starters = ', starters, 'bench order', bench_order, 'captain and vice ', captain, vice_captain, 'OUR INFO FOR VERIFYING !! \n\n\n')
                start, on_bench = asyncio.run(Agent.get_bench_and_starters(self.email, self.password, self.team_id))
                sub_in, sub_out = Brain.figure_out_substitution(start, on_bench, starters, set(bench_order))
                asyncio.run(Agent.select_team(self.email, self.password, self.team_id, sub_in, sub_out, captain, vice_captain, bench_order))
                Accountant.log_gameweek_completion(self.folder, gw, [0, 'pick_team_only'])
                return 


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
            if self.wildcard_method == 'modern': # these points on a 0-1 range
                datapoint = Oracle.create_wildcard_datapoint(current_gw_stats, fixtures_df, squad_selection[1], weekly_point_returns,\
                    full_transfer_market, constants.WILDCARD_2_GW_STARTS[self.season], self.bench_factor, chip_status,\
                    new_team_players['element'].to_list(), free_transfers, gw)
                model, feature_names = Oracle.load_model(self.wildcard_model_path)
                wildcard_pts = model.predict([[datapoint[x] for x in feature_names]])[0]
                wildcard_players = [] # just a storeholder if we don't even compute
                modern_wildcard_active = wildcard_pts >= self.chip_threshold_construction['wildcard'][0] and\
                    gw >= self.earliest_chip_weeks['wildcard'][0] and not(chip_status['wildcard'][0])
                print("Wildcard Prediction Probability: ", wildcard_pts)
                print("Playing wildcard? ", modern_wildcard_active)


            if self.wildcard_method == 'classical' or (self.wildcard_method == 'modern' and modern_wildcard_active):
                print('made it into actual wildcard players')
                wildcard_players, wildcard_pts = Brain.best_wildcard_team(full_transfer_market, sell_value, self.bench_factor,\
                    free_hit=False, allowed_healths=self.allowed_healths)
                wildcard_pts -= Brain.get_points(team_players.drop('expected_pts_N1', axis=1), self.bench_factor)
            if not(chip_status['freehit']):
                freehit_players, freehit_pts = Brain.free_hit_team(full_transfer_market, sell_value, self.freehit_bench_factor,\
                    allowed_healths=self.allowed_healths)
                freehit_pts -= Brain.get_points(team_players.drop('expected_pts_full', axis=1), self.freehit_bench_factor)
            else:
                freehit_pts, freehit_players = .1, new_team_players


            '''decide whether or not to play chips and execute chosen path'''
            earliest_chip_weeks = {name: (x if name != 'wildcard' else x[0] if gw < constants.WILDCARD_2_GW_STARTS[self.season] else x[1]) for (name,x) in self.earliest_chip_weeks.items() }
            this_week_chip = Brain.play_chips_or_no(gw, chip_status, chip_threshold_dict, wildcard_pts, freehit_pts, captain_pts,\
            bench_pts, earliest_chip_weeks, self.chip_threshold_tailoffs)
            print('original chip choice was ', this_week_chip)

            login_info = (self.email, self.password, self.team_id)
            brain_transfer_help = lambda x,y: Brain.match_positions(x, y, full_transfer_market)
            brain_substitution_help = lambda x,y,z,a: Brain.figure_out_substitution(x,y,z,a)
            brain_pick_team_help = lambda x:\
                Brain.pick_team(full_transfer_market.loc[full_transfer_market['element'].isin(x)], health_df)
            potential_teams = new_team_players, wildcard_players, freehit_players 
            
            '''NOW ALL INFO HAS BEEN COMPUTED, IT IS NOW A FRONT END PROBLEM'''
            if constants.NO_CAPTCHA: # old, can't expect it to return to this.
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
            else: # new method, record information in human_output.csv
                pass
                '''read df and make df if not exists, if we don't already have a func for this make it'''
                '''dataframe with gw, chip, players in, players out, starters, bench order, captain, vc'''
                '''update the human_inputs_players.csv, accordingly ''' 
                # Accountant.log_gameweek_completion(self.folder, gw, transfer_info)


            '''update all the tables'''        
            Accountant.update_delta_db(self.folder, choice_report)
            Accountant.update_chip_db(self.folder, gw, wildcard_pts, freehit_pts, captain_pts, bench_pts)
        
        # recording the exceptions to help me with asynchronous debugging
        except Exception as e:
            import traceback, sys
            traceback.print_exc(file=sys.stdout)
            no_input_gmail = constants.NOTIFICATION_SENDER_GMAIL == "" or constants.NOTIFICATION_SENDER_PASSWORD == "" #still need to catch more problems
            send_email, send_password = [(self.email, self.password) if no_input_gmail else (constants.NOTIFICATION_SENDER_GMAIL,constants.NOTIFICATION_SENDER_PASSWORD)][0]
            try:
                gw += 0
            except: # make sure we send errors even when we die before the error starts
                gw = -1
            Agent.notify_human(self.email, send_email, send_password, constants.NOTIFICATION_RECEIVER_EMAIL, gw, 'THERE WAS AN EXCEPTION')
            date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            logging_file = constants.DROPBOX_PATH + 'log_exceptions.txt'
            with open(logging_file, "a+") as f:
                
                f.write(f"Date: {date} -- " + str(e) + '\n')

 


        
import sys 
if __name__ == '__main__':
    from Personalities import personalities_to_run
    for pers in personalities_to_run:
        ai = FPL_AI(**pers)
        ai.make_moves()