import tqdm
import os
import random
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
from collections import Counter, defaultdict

dataset_location = '../implicit-bias/data/LFM-1b/LFM-1b_LEs.txt'
random.seed(42)


def get_awearnes_engagement(counts):
    artist_fan = []
    for a in counts.keys():
        covered_catalog = len(counts[a]['s'])
        total_tracks_listen = counts[a]['t']
        times_counter = 0
        days_counter = 1
        engagement = 0
        s_times = sorted(counts[a]['times'])
        curr_sec = s_times[0]
        for i in s_times:
            if i< (curr_sec + (24*3600)):
                times_counter += 1
            else:
                engagement += (times_counter*np.log1p(days_counter))
                curr_sec = i
                days_counter += 1
                times_counter = 1

        if times_counter >0:
            engagement += (times_counter*np.log1p(days_counter))
        artist_fan.append((a, engagement, covered_catalog, total_tracks_listen, covered_catalog, days_counter))
    return artist_fan

def split(test_size):
    artists_catalog = {}
    artists_users = {}
    last_user = None
    fan_data_awe = []
    fan_data_eng = []
    fan_data_play = []
    fan_data_days= []
    fan_row_train = []
    fan_col_train = []
    fan_test_data = []
    fan_user_ids = []
    fan_item_ids = []
    fan_items_dict = {}
    fan_users_dict = {}
    counts_dict = {}
    user_test_split = {}
    user_pos = {}
    count = 0
    first_time = 9999999999
    last_time = 0
    numb_user_p_artist = 3
    numb_artists_p_user = 10

    # user session item time artist
    for line in tqdm.tqdm(open(dataset_location)):
        hists = line.strip().split('\t')
        user_pos[hists[0]] = count
        if hists[1] not in artists_catalog:
            artists_catalog[hists[1]] = set()
        artists_catalog[hists[1]].add(hists[3])
        if hists[1] not in artists_users:
            artists_users[hists[1]] = set()
        artists_users[hists[1]].add(hists[0])
        curr_time = int(hists[4])
        if curr_time > last_time:
            last_time = curr_time
        if curr_time < first_time:
            first_time = curr_time
        count += 1

    #split_time = ((last_time - first_time)*0.8) + first_time
    print ('last_time', last_time)
    print ('first_time', first_time)
    split_time = 1401584400
    print ('last_time', last_time)
    print ('first_time', first_time)
    print ('split_time', split_time)
 
    count = 0
    for line in tqdm.tqdm(open(dataset_location)):
        hists = line.strip().split('\t')
        if hists[0] not in counts_dict:
            counts_dict[hists[0]] = {}
        curr_time = int(hists[4])
        if curr_time < split_time:
            if hists[1] not in counts_dict[hists[0]]:
                counts_dict[hists[0]][hists[1]] = {'s':set(), 't': 0, 'times':[]}
            counts_dict[hists[0]][hists[1]]['s'].add(hists[3])
            counts_dict[hists[0]][hists[1]]['t'] += 1
            counts_dict[hists[0]][hists[1]]['times'].append(curr_time)
        else:
            if hists[0] not in user_test_split:
                user_test_split[hists[0]] = {}
            if hists[1] not in user_test_split[hists[0]]:
                user_test_split[hists[0]][hists[1]] = {'s':set(), 't': 0, 'times':[]}
            user_test_split[hists[0]][hists[1]]['s'].add(hists[3])
            user_test_split[hists[0]][hists[1]]['t'] += 1
            user_test_split[hists[0]][hists[1]]['times'].append(curr_time)
            #if hists[0] not in user_test_split:
            #    user_test_split[hists[0]] = []
            #user_test_split[hists[0]].append(hists[1])

        last_user = hists[0]
        if user_pos[last_user] == count:
            counts = {a:v for a,v in counts_dict[last_user].items() if len(artists_users[a]) > numb_user_p_artist}
            artist_fan = get_awearnes_engagement(counts)
            if last_user in user_test_split:
                counts = {a:v for a,v in user_test_split[last_user].items() if len(artists_users[a]) > numb_user_p_artist}
                test_artist_fan = get_awearnes_engagement(counts)

            del counts_dict[last_user]
            if len(artist_fan) <= numb_artists_p_user:
                count +=1
                continue

            artist_fan_dict = {a[0]:1 for a in artist_fan}
            fan_users_dict[last_user] = len(fan_user_ids)
            fan_user_ids.append(last_user)
            for item, engagement, awearnes, play, different_songs, days in artist_fan:
                if item not in fan_items_dict:
                    fan_items_dict[item] = len(fan_item_ids)
                    fan_item_ids.append(item)
                fan_col_train.append(fan_items_dict[item])
                fan_row_train.append(fan_users_dict[last_user])
                fan_data_awe.append(awearnes)
                fan_data_eng.append(engagement)
                fan_data_play.append(play)
                fan_data_days.append(days)
            curr_user_test = []
            if last_user in user_test_split:
                for item, engagement, awearnes, play, different_songs, days in test_artist_fan:
                    if item not in artist_fan_dict:
                        if item not in fan_items_dict:
                            fan_items_dict[item] = len(fan_item_ids)
                            fan_item_ids.append(item)
                        curr_user_test.append([fan_items_dict[item], engagement, awearnes, play, different_songs, days])
            fan_test_data.append(curr_user_test)
        count += 1
    print ("ADDED USERS", len(fan_user_ids))
    print ("TEST USERS", len(fan_test_data))
    return fan_data_awe, fan_data_eng, fan_data_play, fan_data_days, fan_row_train, fan_col_train, fan_test_data, fan_items_dict, fan_users_dict

if __name__== "__main__":
    fan_data_awe, fan_data_eng, fan_data_play, fan_data_days, fan_row_train, fan_col_train, fan_test_data, fan_items_dict, fan_users_dict = split(0.2)
    """
    fan_train_awe = sparse.coo_matrix((fan_data_awe, (fan_row_train, fan_col_train)), dtype=np.float32)
    fan_train_eng = sparse.coo_matrix((fan_data_eng, (fan_row_train, fan_col_train)), dtype=np.float32)
    fan_train_play = sparse.coo_matrix((fan_data_play, (fan_row_train, fan_col_train)), dtype=np.float32)
    fan_train_days= sparse.coo_matrix((fan_data_days, (fan_row_train, fan_col_train)), dtype=np.float32)
    print ("TRAIN USERS", fan_train_play.shape)
    sparse.save_npz(os.path.join('data', 'lastfm_time', 'fan_train_data_awearnes_features_fidelity_05.npz'), fan_train_awe)
    sparse.save_npz(os.path.join('data', 'lastfm_time', 'fan_train_data_playcount_features_fidelity_05.npz'), fan_train_play)
    sparse.save_npz(os.path.join('data', 'lastfm_time', 'fan_train_data_engagement_features_fidelity_05.npz'), fan_train_eng)
    sparse.save_npz(os.path.join('data', 'lastfm_time', 'fan_train_data_days_features_fidelity_05.npz'), fan_train_days)
    pickle.dump(fan_test_data, open(os.path.join('data', 'lastfm_time','fan_test_data_features_fidelity_05.pkl'), 'wb'))
    pickle.dump(fan_items_dict, open(os.path.join('data','lastfm_time', 'fan_items_dict_features_fidelity_05.pkl'), 'wb'))
    pickle.dump(fan_users_dict, open(os.path.join('data','lastfm_time', 'fan_users_dict_features_fidelity_05.pkl'), 'wb'))
    """
