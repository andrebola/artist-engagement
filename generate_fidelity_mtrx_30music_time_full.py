import tqdm
import os
import random
import pickle
import pandas as pd
import json

from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
from collections import Counter, defaultdict

dataset_location = 'data/30music_full/relations/events.idomaar'
tracks_location = 'data/30music_full/entities/tracks.idomaar'
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
    track_artists = {}
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

    for line in tqdm.tqdm(open(tracks_location)):
        hists = line.strip().split('\t')
        values = json.loads(hists[4])
        artist_id = str(values["artists"][0]['id'])
        item_id = hists[1]
        track_artists[item_id] = artist_id

    print ("track", track_artists['3351207'])
    # user session item time artist
    for line in tqdm.tqdm(open(dataset_location)):
        hists = line.strip().split('\t')
        values = json.loads(hists[4])
        user_id = str(values['subjects'][0]['id'])
        item_id = str(values['objects'][0]['id'])
        user_pos[user_id] = count
        artist_id = track_artists[item_id]
        if artist_id not in artists_catalog:
            artists_catalog[artist_id] = set()
        artists_catalog[artist_id].add(item_id)
        if artist_id not in artists_users:
            artists_users[artist_id] = set()
        artists_users[artist_id].add(user_id)
        curr_time = int(hists[2])
        if curr_time > last_time:
            last_time = curr_time
        if curr_time < first_time:
            first_time = curr_time
        count += 1

    print ("FIRST COUNT", count)
    split_time = ((last_time - first_time)*0.8) + first_time
    print ('last_time', last_time)
    print ('first_time', first_time)
    print ('split_time', split_time)
    count = 0
    for line in tqdm.tqdm(open(dataset_location)):
        hists = line.strip().split('\t')
        values = json.loads(hists[4])
        user_id = str(values['subjects'][0]['id'])
        item_id = str(values['objects'][0]['id'])
        artist_id = track_artists[item_id]
        if user_id not in counts_dict:
            counts_dict[user_id] = {}
        curr_time = int(hists[2])
        if curr_time < split_time:
            if artist_id not in counts_dict[user_id]:
                counts_dict[user_id][artist_id] = {'s':set(), 't': 0, 'times':[]}
            counts_dict[user_id][artist_id]['s'].add(hists[2])
            counts_dict[user_id][artist_id]['t'] += 1
            counts_dict[user_id][artist_id]['times'].append(curr_time)
        else:
            if user_id not in user_test_split:
                user_test_split[user_id] = {}
            if artist_id not in user_test_split[user_id]:
                user_test_split[user_id][artist_id] = {'s':set(), 't': 0, 'times':[]}
            user_test_split[user_id][artist_id]['s'].add(hists[2])
            user_test_split[user_id][artist_id]['t'] += 1
            user_test_split[user_id][artist_id]['times'].append(curr_time)
            #if hists[0] not in user_test_split:
            #    user_test_split[hists[0]] = []
            #user_test_split[hists[0]].append(hists[4])

        last_user = user_id
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
                if play > 0:
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
    print ("FINAL COUNT", count)
    return fan_data_awe, fan_data_eng, fan_data_play, fan_data_days, fan_row_train, fan_col_train, fan_test_data, fan_items_dict, fan_users_dict

if __name__== "__main__":
    #users_w_features = pickle.load(open('data/users_w_features.pkl', 'rb'))

    fan_data_awe, fan_data_eng, fan_data_play, fan_data_days, fan_row_train, fan_col_train, fan_test_data, fan_items_dict, fan_users_dict = split(0.2)

    fan_train_awe = sparse.coo_matrix((fan_data_awe, (fan_row_train, fan_col_train)), dtype=np.float32)
    fan_train_eng = sparse.coo_matrix((fan_data_eng, (fan_row_train, fan_col_train)), dtype=np.float32)
    fan_train_play = sparse.coo_matrix((fan_data_play, (fan_row_train, fan_col_train)), dtype=np.float32)
    fan_train_days= sparse.coo_matrix((fan_data_days, (fan_row_train, fan_col_train)), dtype=np.float32)
    sparse.save_npz(os.path.join('data', '30music_time_full', 'fan_train_data_awearnes_features_fidelity_05.npz'), fan_train_awe)
    sparse.save_npz(os.path.join('data', '30music_time_full', 'fan_train_data_playcount_features_fidelity_05.npz'), fan_train_play)
    sparse.save_npz(os.path.join('data', '30music_time_full', 'fan_train_data_engagement_features_fidelity_05.npz'), fan_train_eng)
    sparse.save_npz(os.path.join('data', '30music_time_full', 'fan_train_data_days_features_fidelity_05.npz'), fan_train_days)
    pickle.dump(fan_test_data, open(os.path.join('data', '30music_time_full','fan_test_data_features_fidelity_05.pkl'), 'wb'))
    pickle.dump(fan_items_dict, open(os.path.join('data','30music_time_full', 'fan_items_dict_features_fidelity_05.pkl'), 'wb'))
    pickle.dump(fan_users_dict, open(os.path.join('data','30music_time_full', 'fan_users_dict_features_fidelity_05.pkl'), 'wb'))

