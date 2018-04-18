import os.path

import numpy as np
import pandas as pd

def classify_picks(winners, t1_pick, t2_pick):
    win_pick = t1_pick
    lose_pick = t2_pick
    for i, r in enumerate(winners):
        if r < 0:
            win_pick[i], lose_pick[i] = lose_pick[i], win_pick[i]

    return win_pick, lose_pick

def compute_correlation(correlation, heroes):
    for row in heroes:
        for i in range(0, 5):
            for j in range(i + 1, 5):
                correlation[row[i]][row[j]] += 1

def compute_cooccurrence(t1_pick, t2_pick, num_heroes):
    cooccurrence = np.zeros(shape=(num_heroes, num_heroes), dtype=int)

    compute_correlation(cooccurrence, t1_pick)
    compute_correlation(cooccurrence, t2_pick)

    return cooccurrence

def compute_co_win_rate(win_pick, cooccurrence, num_heroes):
    co_win = np.zeros(shape=(num_heroes, num_heroes), dtype=int)

    compute_correlation(co_win, win_pick)

    co_win_rate = np.zeros(shape=(num_heroes, num_heroes))
    for i in range(0, num_heroes):
        for j in range(i + 1, num_heroes):
            match_count = cooccurrence[i][j]
            if match_count > 100:
                co_win_rate[i][j] = co_win[i][j] / match_count
            else:
                co_win_rate[i][j] = 0.5

    return co_win_rate

def compute_against_count(against, t1_pick, t2_pick):
    for t1_row, t2_row in zip(t1_pick, t2_pick):
        for i in range(0, 5):
            for j in range(0, 5):
                against[t1_row[i]][t2_row[j]] += 1

def compute_against_win_rate(win_pick, lose_pick, num_heroes):
    against_win = np.zeros(shape=(num_heroes, num_heroes), dtype=int)
    against_lose = np.zeros(shape=(num_heroes, num_heroes), dtype=int)

    compute_against_count(against_win, win_pick, lose_pick)
    compute_against_count(against_lose, lose_pick, win_pick)

    against_win_rate = np.zeros(shape=(num_heroes, num_heroes))
    for i in range(0, num_heroes):
        for j in range(0, num_heroes):
            match_count = against_win[i][j] + against_lose[i][j]
            if match_count > 100:
                against_win_rate[i][j] = against_win[i][j] / match_count
            else:
                against_win_rate[i][j] = 0.5

    return against_win_rate

def compute_match_count(heroes, num_heroes):
    match_count = np.zeros(num_heroes)
    for row in heroes:
        for i in range(0, 5):
            match_count[row[i]] += 1

    return match_count

def compute_win_rate_and_popularity(win_pick, lose_pick, num_heroes):
    win_count = compute_match_count(win_pick, num_heroes)
    lose_count = compute_match_count(lose_pick, num_heroes)

    global_win_rate = np.zeros(num_heroes)
    global_popularity = np.zeros(num_heroes)
    #print('Total match count: ' + str(len(win_pick)))
    for i in range(0, num_heroes):
        match_count = win_count[i] + lose_count[i]
        #print('Match count of hero ' + str(i) + ": " + str(match_count))
        if match_count > 100:
            global_win_rate[i] = win_count[i] / match_count
        else:
            global_win_rate[i] = 0.5

        global_popularity[i] = (win_count[i] + lose_count[i]) / len(win_pick)
        #print('Popularity of hero ' + str(i) + ": " + str(global_popularity[i]))
        #print('Win rate of hero ' + str(i) + ": " + str(global_win_rate[i]))

    return global_win_rate, global_popularity

def get_cooccurrence_file_path(data_path):
    return data_path + r'\cooccurrence.csv'

def get_co_win_rate_file_path(data_path):
    return data_path + r'\co_win_rate.csv'

def get_against_win_rate_file_path(data_path):
    return data_path + r'\against_win_rate.csv'

def get_global_win_rate_file_path(data_path):
    return data_path + r'\global_win_rate.csv'

def get_global_popularity_file_path(data_path):
    return data_path + r'\global_popularity.csv'

def load_knowledge_if_existed(num_heroes):
    data_path = os.path.dirname(os.path.abspath(__file__)) + r'\data'
    headers = ['hero' + str(i) for i in range(num_heroes)]

    cooc_df = pd.read_csv(get_cooccurrence_file_path(data_path), names=headers, delimiter=',', skipinitialspace=True)
    co_wr_df = pd.read_csv(get_co_win_rate_file_path(data_path), names=headers, delimiter=',', skipinitialspace=True)
    against_wr_df = pd.read_csv(get_against_win_rate_file_path(data_path), names=headers, delimiter=',',
                                skipinitialspace=True)
    global_wr_df = pd.read_csv(get_global_win_rate_file_path(data_path), names=['unused'], delimiter=',',
                               skipinitialspace=True)
    global_pop_df = pd.read_csv(get_global_popularity_file_path(data_path), names=['unused'], delimiter=',',
                                skipinitialspace=True)

    print('Archived knowledge loaded.')
    return cooc_df.as_matrix(), co_wr_df.as_matrix(), against_wr_df.as_matrix(),\
           global_wr_df.as_matrix().flatten(), global_pop_df.as_matrix().flatten()

def archive_knowledge(cooccurrence, co_win_rate, against_win_rate, global_win_rate, global_popularity):
    data_path = os.path.dirname(os.path.abspath(__file__)) + r'\data'

    np.savetxt(get_cooccurrence_file_path(data_path), cooccurrence, fmt='%i', delimiter=',')
    np.savetxt(get_co_win_rate_file_path(data_path), co_win_rate, fmt='%10.5f', delimiter=',')
    np.savetxt(get_against_win_rate_file_path(data_path), against_win_rate, fmt='%10.5f', delimiter=',')
    np.savetxt(get_global_win_rate_file_path(data_path), global_win_rate, fmt='%10.5f', delimiter=',')
    np.savetxt(get_global_popularity_file_path(data_path), global_popularity, fmt='%10.5f', delimiter=',')

def build_knowledge(t1_pick, t2_pick, winners, num_heroes):
    try:
        return load_knowledge_if_existed(num_heroes)
    except FileNotFoundError:
        pass

    print('No archives found. Start building knowledge..')

    win_pick, lose_pick = classify_picks(winners, t1_pick, t2_pick)
    cooccurrence = compute_cooccurrence(t1_pick, t2_pick, num_heroes)
    co_win_rate = compute_co_win_rate(win_pick, cooccurrence, num_heroes)
    against_win_rate = compute_against_win_rate(win_pick, lose_pick, num_heroes)
    global_win_rate, global_popularity = compute_win_rate_and_popularity(win_pick, lose_pick, num_heroes)

    archive_knowledge(cooccurrence, co_win_rate, against_win_rate, global_win_rate, global_popularity)
    return cooccurrence, co_win_rate, against_win_rate, global_win_rate, global_popularity