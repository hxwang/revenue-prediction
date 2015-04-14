#!/usr/bin/env python

import csv
import sys
import numpy as np

# read a csv file into rows
def read_csv(filename, skip_headers = False):    
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        if skip_headers:
            next(reader, None)
        for row in reader:
            rows.append(row)
    return rows


def main():
    data = read_csv('restaurant-revenue-prediction_public_leaderboard.csv', True)
    scores = [float(row[3]) for row in data]
    for score in scores:
        print score


if __name__ == '__main__':
    main()


