import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def compute_against_count(against, t1_pick, t2_pick):
    for index, t1_row in enumerate(t1_pick):
        t2_row = t2_pick[index]
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
        if match_count > 200:
            global_win_rate[i] = win_count[i] / match_count
        else:
            global_win_rate[i] = 0.5

        global_popularity[i] = (win_count[i] + lose_count[i]) / len(win_pick)
        #print('Popularity of hero ' + str(i) + ": " + str(global_popularity[i]))

    return global_win_rate, global_popularity

def compute_cooc_features_impl(cooccurrence, heroes):
    avg_cooc = np.zeros(len(heroes))
    max_cooc = np.zeros(len(heroes))

    for index, row in enumerate(heroes):
        sum = 0
        max_value = 0
        for i in range(0, 5):
            for j in range(i + 1, 5):
                cooc = cooccurrence[row[i]][row[j]]
                sum += cooc
                max_value = max(cooc, max_value)

        avg_cooc[index] = sum / 10
        max_cooc[index] = max_value

    return avg_cooc, max_cooc

def compute_cooc_features(win_pick, lose_pick, cooccurrence):
    win_avg_cooc, win_max_cooc = compute_cooc_features_impl(cooccurrence, win_pick)
    lose_avg_cooc, lose_max_cooc = compute_cooc_features_impl(cooccurrence, lose_pick)

    return win_avg_cooc, win_max_cooc, lose_avg_cooc, lose_max_cooc

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

def build_enhanced_info(t1_pick, t2_pick, winners, num_heroes):
    win_pick, lose_pick = classify_picks(winners, t1_pick, t2_pick)
    cooccurrence = compute_cooccurrence(t1_pick, t2_pick, num_heroes)
    #np.savetxt('C:/Users/tanjianwen/Desktop/cooccurrence.csv', cooc, fmt='%i', delimiter=',')

    against_win_rate = compute_against_win_rate(win_pick, lose_pick, num_heroes)
    #np.savetxt('C:/Users/tanjianwen/Desktop/against_win_rate.csv', against_win_rate, fmt='%10.5f', delimiter=',')

    global_win_rate, global_popularity = compute_win_rate_and_popularity(win_pick, lose_pick, num_heroes)
    #np.savetxt('C:/Users/tanjianwen/Desktop/global_win_rate.csv', global_win_rate, fmt='%10.5f', delimiter=',')
    #np.savetxt('C:/Users/tanjianwen/Desktop/global_popularity.csv', global_popularity, fmt='%10.5f', delimiter=',')

    return cooccurrence, against_win_rate, global_win_rate, global_popularity

def create_enhanced_features(data_frame, t1_pick, t2_pick, cooccurrence, against_win_rate, global_win_rate,
                             global_popularity):
    t1_avg_cooc, t1_max_cooc, t2_avg_cooc, t2_max_cooc = compute_cooc_features(t1_pick, t2_pick, cooccurrence)

    t1_avg_popularity, t1_max_popularity, t2_avg_popularity, t2_max_popularity = \
        compute_popularity_features(t1_pick, t2_pick, global_popularity)
    t1_avg_abs_wr, t1_max_abs_wr, t2_avg_abs_wr, t2_max_abs_wr = \
        compute_abs_win_rate_features(t1_pick, t2_pick, global_win_rate)
    avg_against_wr, max_against_wr = compute_against_win_rate_features(against_win_rate, t1_pick, t2_pick)

    #np.savetxt('C:/Users/tanjianwen/Desktop/win_rate_inspect.csv', avg_against_wr, fmt='%10.5f', delimiter=',')

    data_frame['t1_avg_cooc'] = t1_avg_cooc
    data_frame['t1_max_cooc'] = t1_max_cooc
    data_frame['t2_avg_cooc'] = t2_avg_cooc
    data_frame['t2_max_cooc'] = t2_max_cooc
    data_frame['t1_avg_abs_wr'] = t1_avg_abs_wr
    data_frame['t1_max_abs_wr'] = t1_max_abs_wr
    data_frame['t2_avg_abs_wr'] = t2_avg_abs_wr
    data_frame['t2_max_abs_wr'] = t2_max_abs_wr
    data_frame['avg_against_wr'] = avg_against_wr
    data_frame['max_against_wr'] = max_against_wr
    data_frame['t1_avg_popularity'] = t1_avg_popularity
    data_frame['t1_max_popularity'] = t1_max_popularity
    data_frame['t2_avg_popularity'] = t2_avg_popularity
    data_frame['t2_max_popularity'] = t2_max_popularity

    enhanced_features = []
    enhanced_features.append('t1_avg_cooc')
    enhanced_features.append('t1_max_cooc')
    enhanced_features.append('t2_avg_cooc')
    enhanced_features.append('t2_max_cooc')
    enhanced_features.append('t1_avg_popularity')
    enhanced_features.append('t1_max_popularity')
    enhanced_features.append('t2_max_popularity')
    enhanced_features.append('t2_avg_popularity')
    enhanced_features.append('t1_avg_abs_wr')
    enhanced_features.append('t1_max_abs_wr')
    enhanced_features.append('t2_avg_abs_wr')
    enhanced_features.append('t2_max_abs_wr')
    enhanced_features.append('avg_against_wr')
    enhanced_features.append('max_against_wr')

    return enhanced_features

def prepare_data(train_csv_path, test_csv_path, headers):
    df = pd.read_csv(train_csv_path, names=headers, nrows=10000)

    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .9
    test = df.copy()
    train, test = df[df['is_train']==True], test[test['is_train']==False]

    return train, test

def prepare_data2(train_csv_path, test_csv_path, headers):
    train = pd.read_csv(train_csv_path, names=headers, nrows=90000)
    test = pd.read_csv(test_csv_path, names=headers, nrows=10000)

    return train, test

def main():
    train_csv_path = 'C:/Users/tanjianwen/Desktop/dota2data/dota2Train.csv'
    test_csv_path = 'C:/Users/tanjianwen/Desktop/dota2data/dota2Test.csv'
    headers = ['score', 'cluster_id', 'game_mode', 'game_type']
    num_heroes = 113
    for i in range(0, num_heroes):
        headers.append('hero' + str(i))

    train, test = prepare_data(train_csv_path, test_csv_path, headers)
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    hero_columns = headers[4:]
    #train = test # For test.
    y = train['score']
    train_t1_pick, train_t2_pick = extract_heroes_picks(train[hero_columns], num_heroes)
    cooccurrence, against_win_rate, global_win_rate, global_popularity =\
        build_enhanced_info(train_t1_pick, train_t2_pick, y, num_heroes)

    #train_t1_pick, train_t2_pick = extract_heroes_picks(train[hero_columns], num_heroes)
    enhanced_features = create_enhanced_features(train, train_t1_pick, train_t2_pick,
                                                 cooccurrence, against_win_rate,
                                                 global_win_rate, global_popularity)

    # Train.
    combined_features = enhanced_features + headers[1:]
    clf = gdc(random_state=0, n_estimators=100, max_depth=4, subsample=0.5, learning_rate=0.1)
    #clf = rfc(random_state=0) # For test.
    clf.fit(train[combined_features], y)

    # Predict.
    test_t1_pick, test_t2_pick = extract_heroes_picks(test[hero_columns], num_heroes)
    create_enhanced_features(test, test_t1_pick, test_t2_pick, cooccurrence,
                             against_win_rate, global_win_rate, global_popularity)

    print('Score:', clf.score(test[combined_features], test['score']))
    print('Feature Importance:', list(zip(combined_features, clf.feature_importances_)))

main()