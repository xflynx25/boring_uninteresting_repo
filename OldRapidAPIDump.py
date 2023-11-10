"""VERSION 2 SYNTAX"""

# param: start season year integer yyyy, returns: league id
def get_premier_league_id(year):
    url = "https://api-football-v1.p.rapidapi.com/v2/leagues/country/england/" + str(year)
    print(url)
    response = proper_request("GET", url, headers=HEADERS, max_requests=10)
    leagues = response.json()['api']['leagues']
    print(leagues)
    index = 0
    for n in range(len(leagues)):
        if leagues[n]['name']=='Premier League':
            print(index)
            index = n
            break
    print(leagues[index]['league_id'])
    return leagues[index]['league_id']


# league_id : 2790 for prem2020
# date_string : yyyy-mm-dd
# return: list of (id, h, a, home_g, away_g, status) for all fixtures in the league on the date 
def get_fixture_ids(league_id, date_string):
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
# api calls 1-10 REQUESTS
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
# api calls 1-10 REQUESTS
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
# api calls 1-10 REQUESTS
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
# api calls 1-10 REQUESTS
def teams_in_match(fixture_id):
    url = "https://api-football-v1.p.rapidapi.com/v2/lineups/" + str(fixture_id)

    response = proper_request("GET", url, headers=HEADERS)
    return list(response.json()['api']['lineUps'].keys())




