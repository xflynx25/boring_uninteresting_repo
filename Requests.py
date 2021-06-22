import requests
import re  
import io
import time
import pandas as pd
from math import log2
from constants import RAPID_API_HOST, RAPID_API_KEY

ENCODING1 = 'UTF-8'
ENCODING2 = 'ISO-8859-1'
HEADERS = {
        'x-rapidapi-host': RAPID_API_HOST,
        'x-rapidapi-key': RAPID_API_KEY
        }

class Custom404Exception(Exception):
    """Exception raised when you request url that isn't there"""
    pass

# making api calls, exponential backoff for 429
def proper_request(req_type, url, headers=None, params=None, max_requests=10, max_429_wait_time = 15*60, checking_for_404=False):
    def make_requests():
        count = 0
        response = ''
        while response == '' and count < max_requests:
            try:
                response = requests.request(req_type, url, params=params, headers=headers)
            except:
                time.sleep(5)
                count += 1
        if count == max_requests:
            raise Exception("Could not get any response at all!")
        return response

    response = make_requests()

    errors = 0
    max_requests = log2(max_429_wait_time / 15) // 1
    while response.status_code == 429: #recover from 429 errors
        print('429 response: ', response)
        time.sleep(15*2**errors)
        response = make_requests()
        errors += 1
        if errors > max_requests:
            raise Exception("Too many 429 can't break down barrier")

    if response.status_code != 200:
        if response.status_code == 404:
            raise Custom404Exception("got a 404")
        raise Exception("Response was code " + str(response.status_code))
    return response


# used for drawing csv from github
def get_df_from_internet(url):
    r = proper_request("GET", url).content    
    try:
        csv = io.StringIO(r.decode(ENCODING1))
    except UnicodeDecodeError:
        print('UTF-8 not working')
        csv = io.StringIO(r.decode(ENCODING2)) 
    return pd.read_csv(csv)


# param: start season year integer yyyy, returns: league id
def get_premier_league_id(year):
    url = "https://api-football-v1.p.rapidapi.com/v2/leagues/country/england/" + str(year)

    response = proper_request("GET", url, headers=HEADERS, max_requests=10)
    leagues = response.json()['api']['leagues']
    index = 0
    for n in range(len(leagues)):
        if leagues[n]['name']=='Premier League':
            index = n
            break
    return leagues[index]['league_id']

# league_id : 2790 for prem2020
# date_string : yyyy-mm-dd
# return: list of (id, h, a, home_g, away_g, status) for all fixtures in the league on the date 
def get_fixture_ids(league_id, date_string):
    import requests

    url = "https://api-football-v1.p.rapidapi.com/v2/fixtures/league/" + str(league_id) + '/' + date_string

    response = proper_request("GET", url, headers=HEADERS, max_requests=10)
    fixtures = response.json()['api']['fixtures']
    all_fix = []
    for fix in fixtures:
        id = fix['fixture_id']
        home = fix['homeTeam']['team_id']
        away = fix['awayTeam']['team_id']
        homeGoals = fix['goalsHomeTeam']
        awayGoals = fix['goalsAwayTeam']
        status = fix['status'] #'Match Finished'
        all_fix.append((id, home, away, homeGoals, awayGoals, status))

    return all_fix

# return: dictionary id --> name ***the manchesters are full name, so we will match some other way
def team_id_converter_api(league_id):
    url = "https://api-football-v1.p.rapidapi.com/v2/teams/league/" + str(league_id)

    response = proper_request("GET", url, headers=HEADERS, max_requests=10)
    teams = response.json()['api']['teams']
    all_teams = {}
    for team in teams:
        id = team['team_id']
        name = team['name']
        all_teams[id] = name

    return all_teams

# returns dictionary of the 8 stats, Sh, Sa, ST, F, C dict (8 items)
def get_match_stats(fixture_id):
    url = "https://api-football-v1.p.rapidapi.com/v2/statistics/fixture/" + str(fixture_id) + '/'

    response = proper_request("GET", url, headers=HEADERS, max_requests=10)
    stats = response.json()['api']['statistics']
    info = {}
    info['STh'] = stats["Shots on Goal"]['home']
    info['Sh'] = stats["Total Shots"]['home']
    info['Fh'] = stats["Fouls"]['home']
    info['Ch'] = stats["Corner Kicks"]['home']
    info['STa'] = stats["Shots on Goal"]['away']
    info['Sa'] = stats["Total Shots"]['away']
    info['Fa'] = stats["Fouls"]['away']
    info['Ca'] = stats["Corner Kicks"]['away']
    return info

# returns dictionary with keys as fixture id and tuple of oddsH, oddsD, oddsA as val
def get_bet365_odds(league_id):
    url = "https://api-football-v1.p.rapidapi.com/v2/odds/league/" + str(league_id)

    querystring = {"page":"2"}

    response = proper_request("GET", url, headers=HEADERS, params=querystring, max_requests=10)
    fixtures = response.json()['api']['odds']
    all_fix = {}
    for fix in fixtures:
        try:
            bookies = fix['bookmakers']
            index = 0
            for n in range(len(bookies)):
                if bookies[n]['bookmaker_name']=='Bet365':
                    index=n
                    break
            bet365= bookies[index]['bets'][0]['values'] #match winner
            odds = [0,0,0]
            for odd in bet365:
                if odd['value'] == 'Home':
                    odds[0] = odd['odd']
                elif odd['value'] == 'Draw':
                    odds[1] = odd['odd']
                elif odd['value'] == 'Away':
                    odds[2] = odd['odd']
            fix_id = fix['fixture']['fixture_id']
            all_fix[fix_id] = tuple(odds)
        except:
            print('Tried to get bet365 data for singular fixture but got error')
            continue
    return all_fix


# takes in list of ids (int), 
# returns dictionary with keys as fixture id and tuple of oddsH, oddsD, oddsA as val
def get_bet365_odds_by_fixtures(fixture_id_list):

    all_fix = {}
    for fixture_id in fixture_id_list:
        url = "https://api-football-v1.p.rapidapi.com/v2/odds/fixture/" + str(fixture_id)

        response = proper_request("GET", url, headers=HEADERS)
        fix_all = response.json()['api']['odds']
        try:
            fix = fix_all[0]
        except:
            print(fixture_id, ': ', response.json()['api'])
            continue

        bookies = fix['bookmakers']
        index = 0
        for n in range(len(bookies)):
            if bookies[n]['bookmaker_name']=='Bet365':
                index=n
                break
        bet365= bookies[index]['bets'][0]['values'] #match winner
        odds = [0,0,0]
        for odd in bet365:
            if odd['value'] == 'Home':
                odds[0] = odd['odd']
            elif odd['value'] == 'Draw':
                odds[1] = odd['odd']
            elif odd['value'] == 'Away':
                odds[2] = odd['odd']
        fix_id = fix['fixture']['fixture_id']
        all_fix[fix_id] = tuple(odds)

    return all_fix

#returns list of the two teams playing
def teams_in_match(fixture_id):
    url = "https://api-football-v1.p.rapidapi.com/v2/lineups/" + str(fixture_id)

    response = proper_request("GET", url, headers=HEADERS)
    return list(response.json()['api']['lineUps'].keys())

