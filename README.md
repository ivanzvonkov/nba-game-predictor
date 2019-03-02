# NBA Playoff Predictor
Predicts NBA Championship team based on team statistics from that season.

## How to run?
1. Unzip nba-games-stats-from-2014-to-2018.zip in the directory.
2. Set `season` variable in classifier.py. Let's use `season = '1718'` as an example. 
3. Run classifier.py. This will do the followng:
    - Create features by averaging each team's stats for the 17/18 season and parsing the stats into more feature friendly numbers
    - Create targets based on team winning or losing games
    - Train the classifier on the features and targets
    - Save a model into the directory `saved_model/season_1718`
3. In client.py, set `self.season` to same season as classifier and ```model_dir``` to directory of recently created model. Set `current_round_west` and `current_round_east` to the teams that made playoffs in each conference in ascending order. For example `current_round_west = ['HOU', 'GSW', 'POR', 'OKC', 'UTA', 'NOP', 'SAS', 'MIN']`
4. Run client.py. This will do the following: 
    - Construct a playoff scenario and "play" the teams against each other by feeding the model with features containing two teams. 
    - Eliminate teams if they lose until NBA Champion is predicted.
    - Output results to `playoffs_1718.txt`

## Analysis
The classifier was fine tuned on the 14/15 season and then models were made for each season. Therefore the results are most accurate for the 14/15 season and least accurate for 17/18 season. 

While the model did not predict every match up correctly, it generally had good understanding of its prediction. For example in the 2014/2015 season when SAS faced off against LAC. LAC won in real life (4-3 series) but the model predicted SAS would win but with only a 2.24% confidence. On the other hand when CLE faced off against BOS the model predicted CLE would win with 63.85% confidence and CLE swept BOS. 

## To Do
Fine tune and test on 18/19 season prior to playoffs.




