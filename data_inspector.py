import numpy as np
import pandas as pd

def compare_popularity():
    data_path1 = 'C:/Users/tanjianwen/Desktop/dota2data/hero_popularity_tr.txt'
    data_path2 = 'C:/Users/tanjianwen/Desktop/dota2data/hero_popularity_te.txt'
    headers = ['unused', 'popularity']

    df1 = pd.read_table(data_path1, names=headers, delimiter=':', skipinitialspace=True)
    df2 = pd.read_table(data_path2, names=headers, delimiter=':', skipinitialspace=True)
    df1['diff'] = df1['popularity'] - df2['popularity']
    #np.savetxt('C:/Users/tanjianwen/Desktop/popularity_diff.csv', df1['diff'], fmt='%10.5f', delimiter=',')

def compare_global_win_rate():
    data_path1 = 'C:/Users/tanjianwen/Desktop/dota2data/hero_win_rate_tr.txt'
    data_path2 = 'C:/Users/tanjianwen/Desktop/dota2data/hero_win_rate_te.txt'
    headers = ['unused', 'win_rate']

    df1 = pd.read_table(data_path1, names=headers, delimiter=':', skipinitialspace=True)
    df2 = pd.read_table(data_path2, names=headers, delimiter=':', skipinitialspace=True)
    df1['diff'] = df1['win_rate'] - df2['win_rate']
    #np.savetxt('C:/Users/tanjianwen/Desktop/win_rate_diff.csv', df1['diff'], fmt='%10.5f', delimiter=',')

def compare_against_win_rate():
    data_path1 = 'C:/Users/tanjianwen/Desktop/dota2data/against_win_rate_tr.csv'
    data_path2 = 'C:/Users/tanjianwen/Desktop/dota2data/against_win_rate_te.csv'
    headers = ['hero' + str(i) for i in range(113)]

    df1 = pd.read_csv(data_path1, names=headers, delimiter=',', skipinitialspace=True)
    df2 = pd.read_csv(data_path2, names=headers, delimiter=',', skipinitialspace=True)

    df1 -= df2
    #np.savetxt('C:/Users/tanjianwen/Desktop/against_win_rate_diff.csv', df1, fmt='%10.5f', delimiter=',')

    for name, col in df1.iteritems():
        for i, v in enumerate(col):
            if abs(v) > 0.15:
                print('(' + str(i) + ", " + name + "): " + str(v))

def main():
    compare_popularity()
    compare_global_win_rate()
    compare_against_win_rate()

main()