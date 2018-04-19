import os.path
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import knowledge_builder as kb

from sklearn.ensemble import GradientBoostingClassifier as gdc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.model_selection import StratifiedKFold as skf

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
    min_cooc = np.zeros(len(heroes))

    for index, row in enumerate(heroes):
        coop = [cooperation[row[i]][row[j]] for i in range(5) for j in range(i + 1, 5)]
        avg_cooc[index] = statistics.mean(coop)
        max_cooc[index] = max(coop)
        min_cooc[index] = min(coop)

    return avg_cooc, max_cooc, min_cooc

def compute_cooc_features(t1_pick, t2_pick, cooccurrence):
    t1_avg_cooc, t1_max_cooc, t1_min_cooc = compute_coop_features_impl(cooccurrence, t1_pick)
    t2_avg_cooc, t2_max_cooc, t2_min_cooc = compute_coop_features_impl(cooccurrence, t2_pick)

    avg_cooc_diff = t1_avg_cooc - t2_avg_cooc
    max_cooc_diff = t1_max_cooc - t2_max_cooc
    min_cooc_diff = t1_min_cooc - t2_min_cooc

    return t1_avg_cooc, t1_max_cooc, t1_min_cooc, t2_avg_cooc, t2_max_cooc, t2_min_cooc,\
           avg_cooc_diff, max_cooc_diff, min_cooc_diff

def compute_co_win_rate_features(t1_pick, t2_pick, co_win):
    t1_avg_co_wr, t1_max_co_wr, t1_min_co_wr = compute_coop_features_impl(co_win, t1_pick)
    t2_avg_co_wr, t2_max_co_wr, t2_min_co_wr = compute_coop_features_impl(co_win, t2_pick)

    avg_co_wr_diff = t1_avg_co_wr - t2_avg_co_wr
    max_co_wr_diff = t1_max_co_wr - t2_max_co_wr
    min_co_wr_diff = t1_min_co_wr - t2_min_co_wr

    return t1_avg_co_wr, t1_max_co_wr, t1_min_co_wr, t2_avg_co_wr, t2_max_co_wr, t2_min_co_wr,\
           avg_co_wr_diff, max_co_wr_diff, min_co_wr_diff

def compute_against_win_rate_features(against, t1_pick, t2_pick):
    avg_against_wr = np.zeros(len(t1_pick))
    med_against_wr = np.zeros(len(t1_pick))
    max_against_wr = np.zeros(len(t1_pick))
    min_against_wr = np.zeros(len(t1_pick))

    for index, t1_row, t2_row in zip(range(len(t1_pick)), t1_pick, t2_pick):
        against_wr = [against[t1_row[i]][t2_row[j]] for i in range(5) for j in range(5)]
        avg_against_wr[index] = statistics.mean(against_wr)
        med_against_wr[index] = statistics.median(against_wr)
        max_against_wr[index] = max(against_wr)
        min_against_wr[index] = min(against_wr)

    return avg_against_wr, med_against_wr, max_against_wr, min_against_wr

def compute_abs_win_rate_features_impl(win_rate, heroes):
    avg_abs_wr = np.zeros(len(heroes))
    max_abs_wr = np.zeros(len(heroes))
    min_abs_wr = np.zeros(len(heroes))

    for index, row in enumerate(heroes):
        wr = [win_rate[row[i]] for i in range(5)]
        avg_abs_wr[index] = statistics.mean(wr)
        max_abs_wr[index] = max(wr)
        min_abs_wr[index] = min(wr)

    return avg_abs_wr, max_abs_wr, min_abs_wr

def compute_abs_win_rate_features(t1_pick, t2_pick, global_win_rate):
    t1_avg_abs_wr, t1_max_abs_wr, t1_min_abs_wr = compute_abs_win_rate_features_impl(global_win_rate, t1_pick)
    t2_avg_abs_wr, t2_max_abs_wr, t2_min_abs_wr = compute_abs_win_rate_features_impl(global_win_rate, t2_pick)

    return t1_avg_abs_wr, t1_max_abs_wr, t1_min_abs_wr, t2_avg_abs_wr, t2_max_abs_wr, t2_min_abs_wr

# Is the hero having a higher win rate the one that having a higher against win rate?
def compute_union_win_rate_features(t1_pick, t2_pick, global_win_rate, against_win_rate):
    t1_avg_union_wr = np.zeros(len(t1_pick))
    t1_max_union_wr = np.zeros(len(t1_pick))
    t1_min_union_wr = np.zeros(len(t1_pick))
    t2_avg_union_wr = np.zeros(len(t2_pick))
    t2_max_union_wr = np.zeros(len(t2_pick))
    t2_min_union_wr = np.zeros(len(t2_pick))

    for index, t1_row, t2_row in zip(range(len(t1_pick)), t1_pick, t2_pick):
        t1_union_wr = [global_win_rate[t1_row[i]] * against_win_rate[t1_row[i]][t2_row[j]]
                       for i in range(5) for j in range(5)]
        t2_union_wr = [global_win_rate[t2_row[j]] * against_win_rate[t2_row[j]][t1_row[i]]
                       for i in range(5) for j in range(5)]

        t1_avg_union_wr[index] = statistics.mean(t1_union_wr)
        t1_max_union_wr[index] = max(t1_union_wr)
        t1_min_union_wr[index] = min(t1_union_wr)
        t2_avg_union_wr[index] = statistics.mean(t2_union_wr)
        t2_max_union_wr[index] = max(t2_union_wr)
        t2_min_union_wr[index] = min(t2_union_wr)

    avg_union_wr_diff = t1_avg_union_wr - t2_avg_union_wr
    max_union_wr_diff = t1_max_union_wr - t2_max_union_wr
    min_union_wr_diff = t1_min_union_wr - t2_min_union_wr

    return t1_avg_union_wr, t1_max_union_wr, t1_min_union_wr, t2_avg_union_wr, t2_max_union_wr, t2_min_union_wr,\
           avg_union_wr_diff, max_union_wr_diff, min_union_wr_diff

def compute_popularity_features_impl(popularity, heroes):
    avg_popularity = np.zeros(len(heroes))
    max_popularity = np.zeros(len(heroes))
    min_popularity = np.zeros(len(heroes))

    for index, row in enumerate(heroes):
        p = [popularity[row[i]] for i in range(5)]
        avg_popularity[index] = statistics.mean(p)
        max_popularity[index] = max(p)
        min_popularity[index] = min(p)

    return avg_popularity, max_popularity, min_popularity

def compute_popularity_features(t1_pick, t2_pick, global_popularity):
    t1_avg_popularity, t1_max_popularity, t1_min_popularity =\
        compute_popularity_features_impl(global_popularity, t1_pick)
    t2_avg_popularity, t2_max_popularity, t2_min_popularity =\
        compute_popularity_features_impl(global_popularity, t2_pick)

    return t1_avg_popularity, t1_max_popularity, t1_min_popularity,\
           t2_avg_popularity, t2_max_popularity, t2_min_popularity

def create_enhanced_features(data_frame, t1_pick, t2_pick, cooccurrence, co_win_rate, against_win_rate,
                             global_win_rate, global_popularity):
    t1_avg_cooc, t1_max_cooc, t1_min_cooc, t2_avg_cooc, t2_max_cooc, t2_min_cooc,\
    avg_cooc_diff, max_cooc_diff, min_cooc_diff =\
        compute_cooc_features(t1_pick, t2_pick, cooccurrence)
    t1_avg_co_wr, t1_max_co_wr, t1_min_co_wr, t2_avg_co_wr, t2_max_co_wr, t2_min_co_wr,\
    avg_co_wr_diff, max_co_wr_diff, min_co_wr_diff =\
        compute_co_win_rate_features(t1_pick, t2_pick, co_win_rate)
    t1_avg_abs_wr, t1_max_abs_wr, t1_min_abs_wr, t2_avg_abs_wr, t2_max_abs_wr, t2_min_abs_wr =\
        compute_abs_win_rate_features(t1_pick, t2_pick, global_win_rate)
    avg_against_wr, med_against_wr, max_against_wr, min_against_wr =\
        compute_against_win_rate_features(against_win_rate, t1_pick, t2_pick)
    t1_avg_union_wr, t1_max_union_wr, t1_min_union_wr, t2_avg_union_wr, t2_max_union_wr, t2_min_union_wr,\
    avg_union_wr_diff, max_union_wr_diff, min_union_wr_diff =\
        compute_union_win_rate_features(t1_pick, t2_pick, global_win_rate, against_win_rate)
    t1_avg_popularity, t1_max_popularity, t1_min_popularity, t2_avg_popularity, t2_max_popularity, t2_min_popularity = \
        compute_popularity_features(t1_pick, t2_pick, global_popularity)

    #np.savetxt('C:/Users/tanjianwen/Desktop/win_rate_inspect.csv', avg_against_wr, fmt='%10.5f', delimiter=',')

    feature_tuples = [('t1_avg_cooc', t1_avg_cooc),
                      ('t1_max_cooc', t1_max_cooc),
                      ('t1_min_cooc', t1_min_cooc),
                      ('t2_avg_cooc', t2_avg_cooc),
                      ('t2_max_cooc', t2_max_cooc),
                      ('t2_min_cooc', t2_min_cooc),
                      ('avg_cooc_diff', avg_cooc_diff),
                      ('max_cooc_diff', max_cooc_diff),
                      ('min_cooc_diff', min_cooc_diff),
                      ('t1_avg_co_wr', t1_avg_co_wr),
                      ('t1_max_co_wr', t1_max_co_wr),
                      ('t1_min_co_wr', t1_min_co_wr),
                      ('t2_avg_co_wr', t2_avg_co_wr),
                      ('t2_max_co_wr', t2_max_co_wr),
                      ('t2_min_co_wr', t2_min_co_wr),
                      ('avg_co_wr_diff', avg_co_wr_diff),
                      ('max_co_wr_diff', max_co_wr_diff),
                      ('min_co_wr_diff', min_co_wr_diff),
                      ('avg_against_wr', avg_against_wr),
                      ('med_against_wr', med_against_wr),
                      ('max_against_wr', max_against_wr),
                      ('min_against_wr', min_against_wr),
                      ('t1_avg_union_wr', t1_avg_union_wr),
                      ('t1_max_union_wr', t1_max_union_wr),
                      ('t1_min_union_wr', t1_min_union_wr),
                      ('t2_avg_union_wr', t2_avg_union_wr),
                      ('t2_max_union_wr', t2_max_union_wr),
                      ('t2_min_union_wr', t2_min_union_wr),
                      ('avg_union_wr_diff', avg_union_wr_diff),
                      ('max_union_wr_diff', max_union_wr_diff),
                      ('min_union_wr_diff', min_union_wr_diff),
                      ('t1_avg_abs_wr', t1_avg_abs_wr),
                      ('t1_max_abs_wr', t1_max_abs_wr),
                      ('t1_min_abs_wr', t1_min_abs_wr),
                      ('t2_avg_abs_wr', t2_avg_abs_wr),
                      ('t2_max_abs_wr', t2_max_abs_wr),
                      ('t2_min_abs_wr', t2_min_abs_wr),
                      ('t1_avg_popularity', t1_avg_popularity),
                      ('t1_max_popularity', t1_max_popularity),
                      ('t1_min_popularity', t1_min_popularity),
                      ('t2_avg_popularity', t2_avg_popularity),
                      ('t2_max_popularity', t2_max_popularity),
                      ('t2_min_popularity', t2_min_popularity)]

    enhanced_features = []
    for x in feature_tuples:
        data_frame[x[0]] = x[1]
        enhanced_features.append(x[0])

    return enhanced_features

def prepare_data(train_csv_path, test_csv_path, headers):
    df = pd.read_csv(train_csv_path, names=headers, nrows=90000)

    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .9
    test = df.copy()
    train, test = df[df['is_train']==True], test[test['is_train']==False]

    return train, test

def prepare_data2(train_csv_path, test_csv_path, headers):
    train = pd.read_csv(train_csv_path, names=headers)
    test = pd.read_csv(test_csv_path, names=headers)

    return train, test

def prepare_data3(train_csv_path, test_csv_path, headers):
    train = pd.read_csv(train_csv_path, names=headers)
    test = pd.read_csv(test_csv_path, names=headers)

    test['chosen'] = np.random.uniform(0, 1, len(test)) >= .9
    test = test[test['chosen']==True]

    return train, test

def archive_train_set(train_dataset, score):
    if 'is_train' in train_dataset.columns:
        data_path = os.path.dirname(os.path.abspath(__file__)) + '/data'
        id = '%.4f_' % score + str(len(train_dataset)) + time.strftime("_%y%h%d-%H%M%S")
        np.savetxt(data_path + '/train_rows_chosen_' + id + '.csv', train_dataset['is_train'], fmt='%i')

def show_feature_importance(features, importances):
    feature_importances = sorted(zip(features, importances), key=lambda t: t[1])

    fig, ax = plt.subplots()

    y_pos = np.arange(len(feature_importances))

    ax.barh(y_pos, [x for (_, x) in feature_importances], color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([x for (x, _) in feature_importances])
    ax.set_title('Feature Importance')

    plt.tight_layout()
    plt.show()

def enhance_features(headers, train, test, num_heroes):
    hero_columns = headers[4:]
    train_t1_pick, train_t2_pick = extract_heroes_picks(train[hero_columns], num_heroes)
    cooccurrence, co_win_rate, against_win_rate, global_win_rate, global_popularity = \
        kb.build_knowledge(train_t1_pick, train_t2_pick, train['score'], num_heroes)

    enhanced_features = create_enhanced_features(train, train_t1_pick, train_t2_pick,
                                                 cooccurrence, co_win_rate, against_win_rate,
                                                 global_win_rate, global_popularity)

    if not test is None:
        test_t1_pick, test_t2_pick = extract_heroes_picks(test[hero_columns], num_heroes)
        create_enhanced_features(test, test_t1_pick, test_t2_pick, cooccurrence, co_win_rate,
                                 against_win_rate, global_win_rate, global_popularity)

    return enhanced_features

def use_splitted_data_set(train, combined_features):
    splitter = skf(5, shuffle=True, random_state=0)
    sp_iter = splitter.split(train[combined_features], train['score'])
    chosen_index, _ = next(sp_iter)
    return train.iloc[chosen_index]

def train_and_evaluate(train_csv_path, test_csv_path, headers, num_heroes):
    train, test = prepare_data(train_csv_path, test_csv_path, headers)
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    #train = test # For test.
    enhanced_features = enhance_features(headers, train, test, num_heroes)
    combined_features = enhanced_features + headers[1:4]

    # Use the set from grid search.
    #train = use_splitted_data_set(train, combined_features)

    clf = gdc(random_state=0, n_estimators=30, max_depth=8, subsample=0.5, learning_rate=0.05)
    #clf = rfc(random_state=0) # For test.
    clf.fit(train[combined_features], train['score'])

    score = clf.score(test[combined_features], test['score'])
    print('Score:' + '\x1b[1;33;40m', score, '\x1b[0m')
    print('Score for train set:', clf.score(train[combined_features], train['score']))
    archive_train_set(train, score)
    show_feature_importance(combined_features, clf.feature_importances_)

def perform_grid_search(train_csv_path, headers, num_heroes):
    df = pd.read_csv(train_csv_path, names=headers, nrows=10000)
    print('Number of observations in the training data:', len(df))

    enhanced_features = enhance_features(headers, df, None, num_heroes)
    combined_features = enhanced_features + headers[1:4]

    tuned_parameters = {'n_estimators': [50, 100],
                        'max_depth': [6, 8],
                        'subsample': [0.5],
                        'learning_rate': [0.01, 0.05]}
    splitter = skf(5, shuffle=True, random_state=0)
    clf = gscv(gdc(), tuned_parameters, cv=splitter, n_jobs=-1)
    clf.fit(df[combined_features], df['score'])

    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in sorted(zip(means, stds, clf.cv_results_['params'])):
        print("%0.4f (+/-%0.04f) for %r" % (mean, std * 2, params))

    print()
    print('Best score: ' + '\x1b[1;33;40m', clf.best_score_, '\x1b[0m')
    print('Best parameters set found on development set:')
    print()
    print(clf.best_params_)

def main():
    data_path = os.path.dirname(os.path.abspath(__file__)) + r'\data'
    train_csv_path = data_path + '/dota2Train.csv'
    test_csv_path = data_path + '/dota2Test.csv'
    num_heroes = 113
    headers = ['score', 'cluster_id', 'game_mode', 'game_type'] + ['hero' + str(i) for i in range(0, num_heroes)]

    #train_and_evaluate(train_csv_path, test_csv_path, headers, num_heroes)
    perform_grid_search(train_csv_path, headers, num_heroes)

if __name__ == '__main__':
    main()