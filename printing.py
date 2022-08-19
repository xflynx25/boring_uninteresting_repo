import pandas as pd


def center_string(maybe_strng, width):
    strng = str(maybe_strng)
    space = width - len(strng) 
    if space <= 0:
        return strng[:width]
    else:
        left, right = [space // 2] * 2
        if space % 2 == 1:
            left += 1
        return ' ' * left + strng + ' ' * right 



# height = 9
def get_box_print_lines(width, name, opp, score, armband):
    middle_space = width - 2
    line_1 = '-' * width

    person_names = name.split()
    this_name = ''
    for i in range(len(person_names) -1, -1, -1):
        if len(person_names[i]) + 1 <= middle_space - len(this_name):
            this_name = person_names[i] + ' ' * (i != len(person_names) -1) + this_name
    if this_name == '':
        this_name = name

    line_2 = '|' + center_string(this_name[-middle_space:], middle_space) + '|'
    line_3 = '|' + ' ' * (middle_space) + '|'
    line_4 = '|' + center_string('vs '+opp, middle_space) + '|'
    #line_5 = '|' + ' ' * (middle_space) + '|'
    line_6 = '|' + ' ' * (middle_space) + '|'
    line_7 = '|' + center_string(score, middle_space) + '|'
    
    if armband == 'captain':
        line_7 = line_7[:-5] + '|C|' + line_7[-2:]
        line_6 = line_6[:-4] + '_' + line_6[-3:]
    elif armband == 'vcaptain':
        line_7 =  line_7[:-6] + '|vC|' + line_7[-2:]
        line_6 = line_6[:-5] + '__' + line_6[-3:]
    #line_8 = '|' + ' ' * (middle_space) + '|'
    line_9 = '-' * width

    linestrings = []
    for i in range(1, 10):
        if i not in (5, 8):
            linestrings.append(eval('line_' + str(i)))
    return linestrings



# total width will be player_width * 5 + 4 * spacing
# @params: first 4 meta, next 3 specifics
def print_row(field, bench, captain, vcaptain, position, player_width, spacing):
    if position == 'bench':
        players = bench
    else:
        players = field.loc[field['position']==position]

    total_width = player_width * 5 + 4 * spacing
    n_players = players.shape[0]
    center_takeup = (player_width * n_players + spacing * (n_players-1) )
    initial_offset = (total_width -  center_takeup) // 2
    final_offset = total_width - center_takeup - initial_offset #unnecessary

    boxes = []
    for i, player in players.iterrows():
        armband = ('captain' if captain == player['name'] else 'vcaptain' if vcaptain == player['name'] else None)
        lines = get_box_print_lines(player_width, player['name'], player['opponent'], player['points'], armband)
        boxes.append(lines)

    height = len(boxes[0])
    for line in range(height):
        printstring = ''
        printstring += ' ' * initial_offset
        for box in range(n_players):
            printstring += boxes[box][line]
            printstring += ' ' * spacing
        printstring = printstring[:-spacing] #get rid of last spacing addition
        printstring += ' ' * final_offset
        print(printstring)


# @params: first 4 meta, next 3 specifics
def print_all_players(field, bench, captain, vcaptain, player_width, spacing_width, spacing_height):
    for position in range(1,5):
        print_row(field, bench, captain, vcaptain, position, player_width, spacing_width)
        if spacing_height > 0:  
            print('\n' * spacing_height)

    print('\/' *( (5 * player_width + 4 * spacing_width) // 2))
    print_row(field, bench, captain, vcaptain, 'bench', player_width, spacing_width)


def get_top_left(width, wk_transfer, total_points, benchmark_points):
    line_1 = center_string('In',( width - 2) // 2) + '|' + center_string('Out',( width - 2) // 2) + '|'
    line_2 = '-' * width
    printlines = [line_1, line_2]

    section_width = (width-2) // 2 - 5
    for inb, outb in zip(wk_transfer[0], wk_transfer[1]):
        line = center_string(inb[0], section_width) + '| ' + center_string(str(inb[1]), 2) + ' |'\
            + center_string(outb[0], section_width) + '| ' + center_string(str(outb[1]), 2) + ' |' 
        printlines.append(line)

    ''' score compare section (actually goes on top)'''
    midline = center_string(f'Computer :: {total_points}  _vs_  {benchmark_points} :: Benchmark', width)
    score_lines = [center_string('________Benchmark Tracker________', width), midline, ' ' * width]
    return score_lines + printlines


# '|' on sides with the 5 by 3 number
def weekly_points_lines(width, num, hit):
    #digit_draw = {
    #    0: ['---', '| |','| |','---'], 1: ['---', '| |','| |','---'],
    #    2: ['---', '| |','| |','---'], 3: ['---', '| |','| |','---'],
    #    4: ['---', '| |','| |','---'], 5: ['---', '| |','| |','---'],
    #    6: ['---', '| |','| |','---'], 7: ['---', '| |','| |','---'],
    #    8: ['---', '| |','| |','---'], 9: ['---', '| |','| |','---'],
    #}
    
    line_1 = '|' + ' ' * (width - 2) + '|'
    line_2 = '|' +  center_string(f'WK PTS ==  {num}', width - 2) + '|' 
    line_4 = '-' * width
    if hit:
        line_3 = '|' +  center_string(f'HIT is -{hit}', width - 2) + '|' 
    else:
        line_3 = line_1
    return [line_1, line_2, line_3, line_4]

# this has the gw, the points, and the chip played
def get_top_right(width, gw, chip, wk_pts, hit):
    line_1 = '-' * width
    line_2 = '|' + center_string(f'GW {gw}', width - 2) + '|' 
    line_3 = '|' + ' ' * (width - 2) + '|'
    lines = [line_1, line_2, line_3]

    lines += weekly_points_lines(width, wk_pts, hit)

    if chip not in  ('none', 'normal'):
        line_5 = '|' + center_string(f'Chip = {chip}', width - 2) + '|' 
        line_6 = '-' * width
        lines += [line_5, line_6]

    return lines



#@PARAM: field/bench: df with ('name', 'opponent(s)', 'position', 'points')  JUST THE RAW TEAM NO AUTO SUBS OR CAPTAINCY POINTS
    # benchmark_path: read in csv which has the precomputed point scores of some opponent to compare self to
    # wk_transfer is (inb, outb) but instead of elements we have  (name, points) tuples
    # all points should be in integers already
def pretty_print_gw(gw, wk_transfer, field, bench, captain, vcaptain, chip, wk_pts, total_points, hit, benchmark_path=None):
    total_points += hit

    ''' DEFINE CONSTANTS '''
    # FOR BOTTOM SEC
    PLAYER_WIDTH, SPACING_WIDTH, SPACING_HEIGHT = 20, 3, 0

    # FOR TOP SEC 
    SECTION_SPACING_HEIGHT = 0
    SECTION_SPACING_WIDTH = 20
    RIGHT_BOX_WIDTH= 9 + 2 * 8


    ''' TOP SECTION '''
    left_side_width = 12 + 2 * PLAYER_WIDTH # must be even\
    left_side_offset = ((5*PLAYER_WIDTH + 4*SPACING_WIDTH) - left_side_width - RIGHT_BOX_WIDTH - SECTION_SPACING_WIDTH) // 2
    section_spacing_width = SECTION_SPACING_WIDTH#(5*PLAYER_WIDTH + 4*SPACING_WIDTH) - left_side_width - RIGHT_BOX_WIDTH
        

    #    ''' TOP LEFT SECTION - transfers'''
    benchmark_points = ('n/a' if benchmark_path is None else sum(pd.read_csv(benchmark_path, index_col=0)['points'].to_list()[:gw]))
    printleft = get_top_left(left_side_width, wk_transfer, total_points, benchmark_points)
    #    ''' TOP RIGHT SECTION - scores/chips  2 x 4 big numbers'''
    printright = get_top_right(RIGHT_BOX_WIDTH, gw, chip, wk_pts, hit)

    printleft += [' ' * left_side_width for _ in range(max(0, len(printright) - len(printleft)))]
    printright += [' ' * RIGHT_BOX_WIDTH for _ in range(max(0, len(printleft)-len(printright)))]


    for i in range(len(printleft)):
        print(' ' * left_side_offset + printleft[i] + ' ' * section_spacing_width + printright[i])
    print('\n' * SECTION_SPACING_HEIGHT)


    ''' BOTTOM SECTION ''' 
    print_all_players(field, bench, captain, vcaptain, PLAYER_WIDTH, SPACING_WIDTH, SPACING_HEIGHT)



# prints a scatter plot line graph of different players as they progress through the season
def print_season_comparison_graph(players):
    pass
