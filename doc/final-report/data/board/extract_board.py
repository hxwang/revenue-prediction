#!/usr/bin/env python

import csv

best_scores = {}
with open('restaurant-revenue-prediction_public_leaderboard.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # skip the header
    next(reader, None)
    for row in reader:
        teamId = row[0]
        score = float(row[3])
        if teamId in best_scores:
            best_scores[teamId] = min(best_scores, score)
        else:
            best_scores[teamId] = score

for best_score in best_scores.values():
    print best_score
