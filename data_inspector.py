import pandas as pd
import numpy as np

def compare_popularity():
    data_path1 = 'C:/Users/tanjianwen/Desktop/dota2data/hero_popularity_tr.txt'
    data_path2 = 'C:/Users/tanjianwen/Desktop/dota2data/hero_popularity_te.txt'
    headers = ['unused', 'popularity']

    df1 = pd.read_table(data_path1, names=headers, delimiter=':', skipinitialspace=True)
    df2 = pd.read_table(data_path2, names=headers, delimiter=':', skipinitialspace=True)
    df1['diff'] = df1['popularity'] - df2['popularity']
    np.savetxt('C:/Users/tanjianwen/Desktop/popularity_diff.csv', df1['diff'], fmt='%10.5f', delimiter=',')

def main():
    compare_popularity()

main()