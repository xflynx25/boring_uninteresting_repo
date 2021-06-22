################# SUMMARY #################
# Summary: This file interfaces with the website / email human
#
# 1) Helper Functions
# 2) Pulling information from the website
# 3) Pushing team information up to the website / email human
###########################################
from Requests import proper_request
import pandas as pd
import aiohttp
import asyncio
from FPL_Remote.fpl import FPL
import smtplib
from email.mime.text import MIMEText


''' ########################### '''
''' #### $$$$ HELPERS $$$$ #### '''

def bootstrap_data():
    url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    response = proper_request("GET", url, headers=None)
    return response.json()


# take a string of news and look for return date
def parse_news_for_datestring(news):
    if len(news) < 6:
        return None
    day, month = news[-6:-4], news[-3:]
    month_dict = {
        'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12
    }
    if month not in month_dict:
        return None
    return int(day), month_dict[month]

# both in (int, month) form
def weeks_to_return(current_gw, gw_start_times, datestring):
    target_day, target_month = datestring
    closest_gw = 44
    day_difference = 32

    for i in range(12):
        check_month = [target_month - i if i < target_month else 12 + target_month - i][0]
        for gw, time in gw_start_times.items():
            gw_day, gw_month = time 
            if gw_month != check_month:
                continue 
            if gw_day <= target_day:
                diff = target_day - gw_day
                if diff < day_difference:
                    closest_gw = gw 
                    day_difference = diff
        if closest_gw < 38: #early termination if found one that works
            break
    return min(closest_gw - current_gw, 6)


''' ####################################### '''
''' #### $$$$ PULLING INFORMATION $$$$ #### '''


def get_current_gw():
    url = 'https://fantasy.premierleague.com/api/fixtures/'
    response = proper_request("GET", url, headers=None)
    df_raw = pd.DataFrame(response.json())
    try:
        return int( min( df_raw.loc[df_raw['started']==False]['event'] ) )
    except:
        return 38 #postseason debugging

# returns date and time lists
def get_deadline(gw):
    boot_data = bootstrap_data()
    deadline = [wk['deadline_time'] for wk in boot_data['events'] if wk['id']==int(gw)][0]
    #[yyyy, mm, dd]
    return [int(x) for x in deadline[:10].split('-')], [int(x) for x in deadline[11:19].split(':')]

# @return 
#   gw: int of the current gw
#   squad: list of (element, sell_value) 
#   team_value: the selling_value of the team
#   free_transfers: int
#   chips: dict of wildcard, freehit, bench_boost, triple_captain.
#              wildcard also contains expiration gameweek
async def current_team(email, password, user_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)

        my_team = await user.get_team()
        transfer_status = await user.get_transfers_status()
        chips = await user.get_chips()

    gw = get_current_gw()
    free_transfers = transfer_status['limit']
    itb = transfer_status['bank']

    '''player information'''
    team_value = itb
    squad = []
    for player in my_team:
        element = player['element']
        sell_value = player['selling_price']
        player_info = (element, sell_value)
        squad.append(player_info)

        team_value += sell_value
        
    '''chip information'''
    chip_status = {}
    for chip in chips:
        status = chip['status_for_entry'] == 'available'
        if chip['name'] == 'wildcard':
            chip_status['wildcard'] = (status, chip['stop_event'])
        if chip['name'] == 'freehit':
            chip_status['freehit'] = status 
        if chip['name'] == 'bboost':
            chip_status['bench_boost'] = status 
        if chip['name'] == '3xc':
            chip_status['triple_captain'] = status 

    return gw, squad, team_value, free_transfers, chip_status

# input: squad = list of elements in the players squad to check for injuries
# output: dict {element: multiplier (0-1)}
async def injury_penalties(gw, health_df, squad):
    bootstrap = bootstrap_data()
    gw_start_times = {x['id']:(int(x['deadline_time'][8:10]),int(x['deadline_time'][5:7]))\
        for x in bootstrap['events']}
    elements = {x['id']:(x['chance_of_playing_next_round'], x['news'])\
        for x in bootstrap['elements'] if x['id'] in squad}
    # that was gw -> start time && element -> percentage
    multipliers = {}
    for player in squad:
        percent_chance = elements[player][0]
        news = elements[player][1]
        status = health_df.loc[health_df['element']==player]['status'].to_list()[0]
        flagged = (status != 'a')

        if not flagged: #fully healthy
            multipliers[player] = 1
        elif percent_chance is None: #weird no info, use health
            if status == 'd':
                multipliers[player] == .75
            else: 
                multipliers[player] = 0
        elif percent_chance >= 75: # prob illness or knock 
            multipliers[player] = 23/24
        else: #less than 75% chance of playing
            datestring = parse_news_for_datestring(news)
            if datestring is None: #no news, decide off health
                if percent_chance == 50:
                    multipliers[player] = 5/6
                if percent_chance == 25:
                    multipliers[player] = 2/3
                if percent_chance == 0:
                    multipliers[player] = 0
            else: 
                wks_gone = weeks_to_return(gw, gw_start_times, datestring) 
                multipliers[player] = (6 - wks_gone) / 6

    return multipliers

#@return: whether or not transfers have been made this week
# so you are allowed to try again if you went for free transfer
async def has_made_transfers_already(email, password, user_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        print(user)
        transfers = await user.get_latest_transfers()
        print("transfers made this week= ", transfers)
        if len(transfers) == 0:
            return False
        else:
            print('returning true')
            return True


async def get_bench_and_starters(email, password, user_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        my_team = await user.get_team()
        
    bench = []
    start = []
    for player in my_team:
        if player['position'] > 11:
            bench.append(player['element'])
        else:
            start.append(player['element'])
    return start, bench

async def get_captain_and_vice_captain(email, password, user_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        my_team = await user.get_team()

    captain, vice_captain = 0,0
    for player in my_team:
        if player['is_captain']:
            captain = player['element']
        if player['is_vice_captain']:
            vice_captain = player['element']
    return captain, vice_captain

# Check to make sure there aren't any front-end issues
async def verify_front_end_successful(login_info, verify_info):
    def print_failure_item(bools):
        bad_spots = [i for i,x in enumerate(bools) if x]
        print('The failure positions are: ', bad_spots)
    
    starters, bench, captain, vc, action, squad = verify_info
    everything_went_through = True
    
    new_starters, new_bench = await get_bench_and_starters(*login_info)
    new_captain, new_vc = await get_captain_and_vice_captain(*login_info)

    starters_unequal = set(new_starters) != set(starters)
    bench_unequal = set(new_bench) != set(bench)
    captain_unequal = new_captain != captain 
    vc_unequal = new_vc != vc
    #nothing_changed = action != 0.0 and set(new_starters+new_bench) == set([x[0] for x in squad])
    print(action, set(new_starters+new_bench), set([x[0] for x in squad]))
    
    if starters_unequal or bench_unequal or captain_unequal or vc_unequal:# or nothing_changed:
        everything_went_through = False
        print_failure_item([starters_unequal , bench_unequal , captain_unequal , vc_unequal , nothing_changed])
        #print('item four might just be a bug yet to be fixed')
    
    print_msg = 'everything worked successfully' if everything_went_through else 'SOMETHING WENT WRONG, CHECK FPL SITE'
    print('\n\n',print_msg,'\n\n')
    return everything_went_through


''' ####################################### '''
''' #### $$$$ PUSHING INFORMATION $$$$ #### '''


#@param: inb and outb list are lists of players to go in or out
#@return: None, makes transfer
async def make_transfers(email, password, user_id, inbound_list, outbound_list, wildcard=False, freehit=False):
    
    try:
        if len(inbound_list) != len(outbound_list):
            raise Exception('transfer is inconsistant: inbound and outbound sizes do not match')
        if inbound_list == []:
            return

        async with aiohttp.ClientSession() as session:
            fpl = FPL(session)
            await fpl.login(email=email, password=password)
            user = await fpl.get_user(user_id)
            print('user is ', user)
            await user.transfer(outbound_list, inbound_list, wildcard=wildcard, free_hit=freehit)

    except Exception as e:
            print('exception was', e)
            print('we hit an exception on transfering, but we go on!')

#@param: captain/vice_captain are int, starters is list of players to start
#       my team is listlike of all elements on team (ints)
#async def select_team(email, password, user_id, starters, bench, captain, vice_captain):
async def select_team(email, password, user_id, sub_in, sub_out, captain, vice_captain, desired_bench_order):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        """
        # substitute in all the starters, subbing out the bench players each time, eventually will have good team
        for start, stop in ((0, 4), (4, 8), (7, 11)):
            #print((starters[start:stop],bench))
            await user.substitute(starters[start:stop],bench)
        """
        my_team = await user.get_team() 
        trials = 0
        for inb in sub_in:
            for outb in sub_out:
                await user.substitute([inb], [outb])
        await user.captain(captain)
        await user.vice_captain(vice_captain)
        
        # Now we fix the bench
        my_team = await user.get_team()
        current_bench_order = [x['element'] for x in my_team[-3:]]

        c1, c2, c3 = current_bench_order
        d1, d2, d3 = desired_bench_order
        if c3 != d3:
            if c2 != d2:
                await user.substitute([c3], [c2])
                if c1 != d1:
                    await user.substitute([c3], [c1])
            else:
                await user.substitute([c3], [c1])
        elif c2 != d2:
            await user.substitute([c2], [c1])

    
async def play_bench_boost(email, password, user_id):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        await user.substitute([], [], chip='benchboost')

async def play_triple_captain(email, password, user_id, captain):
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        await fpl.login(email=email, password=password)
        user = await fpl.get_user(user_id)
        await user.substitute([], [], chip='triplecaptain')


# play the chosen transfer/chips & pick team but currently execution through amosbastian does not work
def execute_chip(this_week_chip, chosen_transfer, squad_selection, potential_teams, login_info,\
    brain_transfer_help, brain_substitution_help, brain_pick_team_help):
    new_team_players, wildcard_players, freehit_players = potential_teams

    '''transfer business''' #reverse this logic
    if this_week_chip == 'wildcard':
        my_team = wildcard_players['element'].to_list()
        asyncio.run(play_chip(*login_info, wildcard_players, brain_transfer_help, 'wildcard'))
    elif this_week_chip == 'freehit':
        my_team = freehit_players['element'].to_list()
        asyncio.run(play_chip(*login_info, freehit_players, brain_transfer_help, 'freehit'))
    else: #normal transfer
        inbound_list = list(chosen_transfer['inbound'][0])
        outbound_list = list(chosen_transfer['outbound'][0])
        if inbound_list != [] and inbound_list != [{}]: # avoid on blank gw, TEMP FIX, PLZ MAKE BETTER
            inbound_list, outbound_list = brain_transfer_help(inbound_list, outbound_list)
            asyncio.run(make_transfers(*login_info, inbound_list, outbound_list))
        my_team = new_team_players['element'].to_list()
        print(my_team)
        
    '''picking team'''
    squad_selection = brain_pick_team_help(my_team)[0]
    starters, bench_order, captain, vice_captain = squad_selection
        
    bench = list( set(my_team).symmetric_difference(set(starters)) )
    start, on_bench = asyncio.run(get_bench_and_starters(*login_info))
    sub_in, sub_out = brain_substitution_help(start, on_bench, starters, bench)
    print('sub_in, sub_out= ', sub_in, sub_out)

    asyncio.run(select_team(*login_info, sub_in, sub_out, captain, vice_captain, bench_order)) 
    if this_week_chip == 'triple_captain':
        asyncio.run(play_triple_captain(*login_info, captain))
    if this_week_chip == 'bench_boost':
        print('playing bench_boost')
        asyncio.run(play_bench_boost(*login_info))

    action = len(inbound_list) if this_week_chip not in ('freehit', 'wildcard') else this_week_chip
    if type(action) == int and isinstance(inbound_list[0], set): #short before addressing inb list which doesn't exist always
        action = 0
    return [starters, bench, captain, vice_captain, action] #for verifying


# if we are trying to play a chip, I believe some do not work properly
# so we will send an email to ourselves saying that we are trying to play such a chip 
# possibly the human will see it and manually check to make sure it went through
def notify_human(teamname, comp_email, comp_password, destination_email, gw, chip):
    def send_email(sender_email, sender_password, receiver_email, content):
        #Sends an email to the receiver with the given content
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email,sender_password) #CHANGE ME!
        server.sendmail(sender_email, receiver_email, content)
        server.close()
    
    # metadate specifics
    sender_email = comp_email  # Enter your address
    sender_name = teamname
    receiver_email = destination_email  # Enter receiver address
    receiver_name = ""
    sender_password = comp_password
    
    #constructing email
    from_sec = "From: " + sender_name + " <" + sender_email + ">\n"
    to_sec = "To: " + receiver_name + " <"+receiver_email+">\n"
    sub_sec = "Subject: FPL CHIP ALERT\n"
    body = MIMEText(
        """
        <html>
          <body>
              
            <div style="text-align: left; font-family: Arial; font-size: small">
              <p>
                Reporting for team: <b>{0}</b>
              </p>
              <p>
                This is your FPL_AI alerting you that we have 
                decided to play the <b>{1}</b> chip this week.
              </p>

              <div>
                <p>
                  -your friendly ai
                </p>
              </div>
            </div>
          </body>
        </html>

        """.format(teamname,chip), # any formatted variables 
        'html'
    ).as_string()
    
    message = from_sec+to_sec+sub_sec+body
    send_email(sender_email, sender_password, receiver_email, message)
