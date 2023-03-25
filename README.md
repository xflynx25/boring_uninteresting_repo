# Fantasy Premier League AI

**Version 0.0.0**

## In a Picture

![File Relations Picture](https://github.com/xflynx25/boring_uninteresting_repo/blob/main/Project_Diagram.png?raw=true "File Relations Visualizer")


---

## Information

1) ***What this project is*** </br>
    This is a future world champion FPL AI. The first lines of code were written on 8/20/20. Onlookers can't fathom the AI's method; they are frozen admiring the success. The wizardry. at least ... eventually
    
    <br/>The first iteration (Athena) is an engineered bot modelled after human reasoning. ML is used to predict player scores 
    over varying time periods. Features are derived from player stats, team stats, next opponent stats, 
    positional stats, and attempts to measure the qualitative aspects (transfers, odds, twitter sentiment, timing in year).
    The predicted points are run through a decision making framework to produce an action for the week (transfers, chips, captain, bench).  

    The second iteration (Brutus) will focus on optimizing the models. Neural Networks should help to find more robust patterns. In Athena, Random Forest is used, which is very similar to how humans make decisions - observing a few stats, and comparing to a previous player they've seen and how well they performed. I expect NN to generalize better.

    The third iteration (Cronus? Crassus?) will use MuZero or similar to learn end-to-end: to map directly from data to action. In athena, we map data to point projections, and use my own logic and heuristics to go to actions. It would be better to avoid my involvement altogether.

2) ***How to Configure*** (COMING SOON - This will not work rn)<br/>
    -- Clone amosbastian fpl repository (see contributers section), name it FPL_Remote, and place it in this directory
    -- set up models (or use pretrained)<br/>
    -- set up personality, constants, config files<br/>
    -- run setup.py to install folders<br/>
    -- rerun setup.py when anything has changed <br/>
    -- show how to automate, or can just run manually <br/>

    -- demo video <br/>
    -- docs folder<br/>
    
    -- server/website mode ::: newuser.py

3) ***Other Notes*** </br>

    -- This will be a decent load on the computer for ~30-60 minutes once per week. Days when it does
    not make a transfer will take just a couple minutes. <br/>
    -- Currently it is not 100% automated. Playing chips through amosbastian FPL_Remote no longer works due to captcha, so humans will need to check csv logging files daily to see whether they need to make a move, and if so, to check the move logging files. 


4) ***History*** </br>
    -- 2020-21: 718,084 / 8,240,321 (**8.714%**)</br>
    -- 2021-22: bugs</br>
    -- 2022-23 (ongoing): 1,791,779 / 11,312,228 (**15.839%**) [2nd chance league, bugs 1st half of season. ]


5) ***Status*** </br>

    -- Working to get things more seemlessly integrated for distribution <br/>
    -- Next season to introduce Brutus <br/>
    -- Along with to see how well a full season of this bot can do (never ran a full season live, starting late is huge disadvantage due to the market)<br/>

---

## Contributers
These persons provide open source repositories which were helpful in the creation of this project:<br/>
**Datasets:** Vaastav Anand  @<https://github.com/vaastav/Fantasy-Premier-League> <br/>
**Front-end FPL Remote:** Amos Bastian  @<https://github.com/amosbastian/fpl> <br/><br/>
These are other sources that were essential:<br/>
**Public Dataset** Datahub.io  @<https://datahub.io/sports-data/english-premier-league> <br/>
**Statistics** RapidAPI

---

## License and copyright

***Non-Licensing Statement:*** This repository is not intended for public use at this time. I expect to make this available in time, but as of now, users are not permitted to use or modify this code. 
