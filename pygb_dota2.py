import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import data_expansion as de

from sklearn.ensemble import GradientBoostingClassifier as gdc
from sklearn.ensemble import RandomForestClassifier as rfc

def extract_heroes_picks(raw_matches, num_heroes):
    num_rows = raw_matches.shape[0]
    t1_pick = np.zeros(shape=(num_rows, 5), dtype=int)
    t2_pick = np.zeros(shape=(num_rows, 5), dtype=int)

    row_index = 0 # DataFrame.iterrows() may return some indices larger than the number of rows.
    for _, row in raw_matches.iterrows():
        item_index = 0

        t1_iter = t1_pick[row_index]
        t2_iter = t2_pick[row_index]
        for _, item in row.iteritems():
            if item > 0:
                t1_iter[0] = item_index
                t1_iter = t1_iter[1:]
            elif item < 0:
                t2_iter[0] = item_index
                t2_iter = t2_iter[1:]

            item_index += 1

        row_index += 1

    return t1_pick, t2_pick

def compute_coop_features_impl(cooperation, heroes):
    avg_cooc = np.zeros(len(heroes))
    max_cooc = np.zeros(len(heroes))

    for index, row in enumerate(heroes):
        sum = 0
        max_value = 0
        for i in range(0, 5):
            for j in range(i + 1, 5):
                coop = cooperation[row[i]][row[j]]
                sum += coop
                max_value = max(coop, max_value)

        avg_cooc[index] = sum / 10
        max_cooc[index] = max_value

    return avg_cooc, max_cooc

def compute_cooc_features(t1_pick, t2_pick, cooccurrence):
    t1_avg_cooc, t1_max_cooc = compute_coop_features_impl(cooccurrence, t1_pick)
    t2_avg_cooc, t2_max_cooc = compute_coop_features_impl(cooccurrence, t2_pick)

    return t1_avg_cooc, t1_max_cooc, t2_avg_cooc, t2_max_cooc

def compute_co_win_rate_features(t1_pick, t2_pick, co_win):
    t1_avg_co_wr, t1_max_co_wr = compute_coop_features_impl(co_win, t1_pick)
    t2_avg_co_wr, t2_max_co_wr = compute_coop_features_impl(co_win, t2_pick)

    avg_co_wr_diff = t1_avg_co_wr - t2_avg_co_wr
    max_co_wr_diff = t1_max_co_wr - t2_max_co_wr

    return t1_avg_co_wr, t1_max_co_wr, t2_avg_co_wr, t2_max_co_wr, avg_co_wr_diff, max_co_wr_diff

def compute_against_win_rate_features(against, t1_pick, t2_pick):
    avg_against_wr = np.zeros(len(t1_pick))
    max_against_wr = np.zeros(len(t1_pick))

    for index, t1_row in enumerate(t1_pick):
        sum = 0
        max_value = 0
        t2_row = t2_pick[index]
        for i in range(0, 5):
            for j in range(0, 5):
                against_wr = against[t1_row[i]][t2_row[j]]
                sum += against_wr
                max_value = max(against_wr, max_value)

        avg_against_wr[index] = sum / 25
        max_against_wr[index] = max_value

    return avg_against_wr, max_against_wr

def compute_abs_win_rate_features_impl(win_rate, heroes):
    avg_abs_wr = np.zeros(len(heroes))
    max_abs_wr = np.zeros(len(heroes))

    for index, row in enumerate(heroes):
        sum = 0
        max_value = 0
        for i in range(0, 5):
            r = win_rate[row[i]]
            sum += r
            max_value = max(r, max_value)

        avg_abs_wr[index] = sum / 5
        max_abs_wr[index] = max_value

    return avg_abs_wr, max_abs_wr

def compute_abs_win_rate_features(t1_pick, t2_pick, global_win_rate):
    t1_avg_abs_wr, t1_max_abs_wr = compute_abs_win_rate_features_impl(global_win_rate, t1_pick)
    t2_avg_abs_wr, t2_max_abs_wr = compute_abs_win_rate_features_impl(global_win_rate, t2_pick)

    return t1_avg_abs_wr, t1_max_abs_wr, t2_avg_abs_wr, t2_max_abs_wr

def compute_popularity_features_impl(popularity, heroes):
    avg_popularity = np.zeros(len(heroes))
    max_popularity = np.zeros(len(heroes))

    for index, row in enumerate(heroes):
        sum = 0
        max_value = 0
        for i in range(0, 5):
            r = popularity[row[i]]
            sum += r
            max_value = max(r, max_value)

        avg_popularity[index] = sum / 5
        max_popularity[index] = max_value

    return avg_popularity, max_popularity

def compute_popularity_features(t1_pick, t2_pick, global_popularity):
    t1_avg_popularity, t1_max_popularity = compute_popularity_features_impl(global_popularity, t1_pick)
    t2_avg_popularity, t2_max_popularity = compute_popularity_features_impl(global_popularity, t2_pick)

    return t1_avg_popularity, t1_max_popularity, t2_avg_popularity, t2_max_popularity

def create_enhanced_features(data_frame, t1_pick, t2_pick, cooccurrence, co_win_rate, against_win_rate,
                             global_win_rate, global_popularity):
    t1_avg_cooc, t1_max_cooc, t2_avg_cooc, t2_max_cooc = compute_cooc_features(t1_pick, t2_pick, cooccurrence)

    t1_avg_co_wr, t1_max_co_wr, t2_avg_co_wr, t2_max_co_wr, avg_co_wr_diff, max_co_wr_diff =\
        compute_co_win_rate_features(t1_pick, t2_pick, co_win_rate)
    t1_avg_abs_wr, t1_max_abs_wr, t2_avg_abs_wr, t2_max_abs_wr = \
        compute_abs_win_rate_features(t1_pick, t2_pick, global_win_rate)
    avg_against_wr, max_against_wr = compute_against_win_rate_features(against_win_rate, t1_pick, t2_pick)
    t1_avg_popularity, t1_max_popularity, t2_avg_popularity, t2_max_popularity = \
        compute_popularity_features(t1_pick, t2_pick, global_popularity)

    #np.savetxt('C:/Users/tanjianwen/Desktop/win_rate_inspect.csv', avg_against_wr, fmt='%10.5f', delimiter=',')

    feature_tuples = [('t1_avg_cooc', t1_avg_cooc),
                      ('t1_max_cooc', t1_max_cooc),
                      ('t2_avg_cooc', t2_avg_cooc),
                      ('t2_max_cooc', t2_max_cooc),
                      ('t1_avg_co_wr', t1_avg_co_wr),
                      ('t1_max_co_wr', t1_max_co_wr),
                      ('t2_avg_co_wr', t2_avg_co_wr),
                      ('t2_max_co_wr', t2_max_co_wr),
                      ('avg_co_wr_diff', avg_co_wr_diff),
                      ('max_co_wr_diff', max_co_wr_diff),
                      ('avg_against_wr', avg_against_wr),
                      ('max_against_wr', max_against_wr),
                      ('t1_avg_abs_wr', t1_avg_abs_wr),
                      ('t1_max_abs_wr', t1_max_abs_wr),
                      ('t2_avg_abs_wr', t2_avg_abs_wr),
                      ('t2_max_abs_wr', t2_max_abs_wr),
                      ('t1_avg_popularity', t1_avg_popularity),
                      ('t1_max_popularity', t1_max_popularity),
                      ('t2_avg_popularity', t2_avg_popularity),
                      ('t2_max_popularity', t2_max_popularity)]

    enhanced_features = []
    for x in feature_tuples:
        data_frame[x[0]] = x[1]
        enhanced_features.append(x[0])

    return enhanced_features

def prepare_data(train_csv_path, test_csv_path, headers):
    df = pd.read_csv(train_csv_path, names=headers, nrows=10000)

    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .9
    test = df.copy()
    train, test = df[df['is_train']==True], test[test['is_train']==False]

    return train, test

def prepare_data2(train_csv_path, test_csv_path, headers):
    train = pd.read_csv(train_csv_path, names=headers)
    test = pd.read_csv(test_csv_path, names=headers)

    return train, test

def main():
    train_csv_path = 'C:/Users/tanjianwen/Desktop/dota2data/dota2Train.csv'
    test_csv_path = 'C:/Users/tanjianwen/Desktop/dota2data/dota2Test.csv'
    num_heroes = 113
    headers = ['score', 'cluster_id', 'game_mode', 'game_type'] + ['hero' + str(i) for i in range(0, num_heroes)]

    train, test = prepare_data2(train_csv_path, test_csv_path, headers)
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    hero_columns = headers[4:]
    #train = test # For test.
    y = train['score']
    train_t1_pick, train_t2_pick = extract_heroes_picks(train[hero_columns], num_heroes)
    cooccurrence, co_win_rate, against_win_rate, global_win_rate, global_popularity =\
        de.build_enhanced_info(train_t1_pick, train_t2_pick, y, num_heroes)

    #train_t1_pick, train_t2_pick = extract_heroes_picks(train[hero_columns], num_heroes)
    enhanced_features = create_enhanced_features(train, train_t1_pick, train_t2_pick,
                                                 cooccurrence, co_win_rate, against_win_rate,
                                                 global_win_rate, global_popularity)

    # Train.
    combined_features = enhanced_features + headers[1:4]
    clf = gdc(random_state=0, n_estimators=50, max_depth=4, subsample=0.2, learning_rate=0.1)
    #clf = rfc(random_state=0) # For test.
    clf.fit(train[combined_features], y)

    # Predict.
    test_t1_pick, test_t2_pick = extract_heroes_picks(test[hero_columns], num_heroes)
    create_enhanced_features(test, test_t1_pick, test_t2_pick, cooccurrence, co_win_rate,
                             against_win_rate, global_win_rate, global_popularity)

    print('Score:', clf.score(test[combined_features], test['score']))
    print('Feature Importance:', list(zip(combined_features, clf.feature_importances_)))

main()