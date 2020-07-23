import tqdm
import struct
import os
import numpy as np
import pickle
import argparse

from lightfm import LightFM
from scipy import sparse
from scipy.stats import wilcoxon
from evaluate import evaluate, coverage
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import FunctionTransformer # normalize
from implicit.als import AlternatingLeastSquares

os.environ["OPENBLAS_NUM_THREADS"] = "1"

user_fidelity_features_filename = 'out_user_awear_features_{}.feats'
item_fidelity_features_filename = 'out_item_awear_features_{}.feats'
predictions_fidelity_filename = 'predicted_fidelity_features_{}.npy'
user_features_playcounts_filename = 'out_user_playcounts_als.feats'
item_features_playcounts_filename = 'out_item_playcounts_als.feats'
predictions_playcounts_filename = 'predicted_playcounts_als.npy'
user_features_file = '/home/andres/projects/implicit-bias/data/LFM-1b/LFM-1b_users.txt'

def evaluate2(iteration_tracks, items_dict, tracks_pop, N=100):
    max_pop = np.max(tracks_pop)
    all_songs = {}
    popularity = []
    for user in range(len(iteration_tracks)):
        if len(iteration_tracks[user]):
            curr_pop = 0
            for track in iteration_tracks[user][:N]:
                curr_pop += (1-(tracks_pop[0, track]/max_pop))
                if track not in all_songs:
                    all_songs[track] = 0
                all_songs[track] += 1
            popularity.append(curr_pop/N)

    popularity = np.mean(popularity)
    different_songs = len(all_songs)
    if different_songs > len(items_dict):
        np_counts = np.zeros(different_songs, np.dtype('float64'))
    else:
        np_counts = np.zeros(len(items_dict), np.dtype('float64'))
    np_counts[:different_songs] = np.array(list(all_songs.values())) 
    return gini(np_counts), different_songs/tracks_pop.shape[1], popularity

def gini(array):
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def load_feats(feat_fname, meta_only=False, nrz=False):
    with open(feat_fname, 'rb') as fin:
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16))
        if meta_only:
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32)
        feat = feat.reshape((R, C))
        if nrz:
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat

def save(keys, feats, out_fname):
        feats = np.array(feats, dtype=np.float32)
        with open(out_fname + '.tmp', 'wb') as fout:
            fout.write(b' '.join([k.encode() for k in keys]))
            fout.write(b'\n')
            R, C = feats.shape
            fout.write(struct.pack('qq', *(R, C)))
            fout.write(feats.tostring())
        os.rename(out_fname + '.tmp', out_fname)

def train_als(impl_train_data, dims, user_ids, item_ids, user_features_filem, item_features_file, save_res=True):
    model = AlternatingLeastSquares(factors=dims, iterations=100)
    model.fit(impl_train_data.T)

    user_vecs_reg = model.user_factors
    item_vecs_reg = model.item_factors
    #print("USER FEAT:", user_vecs_reg.shape)
    #print("ITEM FEAT:", item_vecs_reg.shape)

    if save_res==True:
        save(item_ids, item_vecs_reg, item_features_file)
        save(user_ids, user_vecs_reg, user_features_file)
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg


def train_warp(impl_train_data, dims, user_ids, item_ids, user_features_filem, item_features_file, save_res=True):
    model = LightFM(loss='warp', no_components=dims, max_sampled=30)
    model = model.fit(impl_train_data, epochs=50, num_threads=8)

    user_biases, user_embeddings = model.get_user_representations()
    item_biases, item_embeddings = model.get_item_representations()
    item_vecs_reg = np.concatenate((item_embeddings, np.reshape(item_biases, (1, -1)).T), axis=1)
    user_vecs_reg = np.concatenate((user_embeddings, np.ones((1, user_biases.shape[0])).T), axis=1)
    #print("USER FEAT:", user_vecs_reg.shape)
    #print("ITEM FEAT:", item_vecs_reg.shape)
    if save_res==True:
        save(item_ids, item_vecs_reg, item_features_file)
        save(user_ids, user_vecs_reg, user_features_file)
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg


def predict(item_vecs_reg, user_vecs_reg, prediction_file,impl_train_data, N=100, step=1000, save_res=True):
    listened_dict = impl_train_data
    predicted = np.zeros((user_vecs_reg.shape[0],N), dtype=np.uint32)
    for u in range(0,user_vecs_reg.shape[0], step):
        sims = user_vecs_reg[u:u+step].dot(item_vecs_reg.T)
        curr_users = listened_dict[u:u+step].todense() == 0
        topn = np.argsort(-np.multiply(sims,curr_users), axis=1)[:,:N]
        predicted[u:u+step, :] = topn
        #if u % 100000 == 0:
        #    print ("Precited users: ", u)
    if save_res==True:
        np.save(open(prediction_file, 'wb'), predicted)
    return predicted

def show_eval(predicted_x, fan_test_data,item_ids, sum_listen, beta=None, prev_results=None, print_pval=False):
    topn = predicted_x.shape[1]
    artists_res = {}
    fan_test_data_sorted = []
    all_res = {'test_engagement': [], 'test_awearnes': [], 'test_playcounts': [], 'pred_awearnes': {}, 'pred_engagement': {}, 'pred_playcounts': {}, 'test_songs': [],'test_days':[], 'pred_songs':{}, 'pred_days':{}}
    for cutoff in ('1', '3', '5', '10', '100'):
        for name in ('pred_awearnes', 'pred_engagement', 'pred_playcounts', 'pred_songs', 'pred_days'):
            all_res[name][cutoff] = []
            if name not in artists_res:
                artists_res[name] = {}
            artists_res[name][cutoff] = {}

    for i in range(len(fan_test_data)):
        if len(fan_test_data[i]) == 0:
            continue
        predicted_dict = {a:pos for pos, a in enumerate(predicted_x[i])}
        pred_in_test = [a for a in fan_test_data[i] if a[0] in predicted_dict]
        all_res['test_awearnes'].append(np.sum([a[2] for a in fan_test_data[i]]))
        all_res['test_engagement'].append(np.sum([a[1] for a in fan_test_data[i]]))
        all_res['test_playcounts'].append(np.sum([a[3] for a in fan_test_data[i]]))
        all_res['test_songs'].append(np.sum([a[4] for a in fan_test_data[i]]))
        all_res['test_days'].append(np.sum([a[5] for a in fan_test_data[i]]))
        #if len(pred_in_test) == 0:
        #    continue
        res = {'pred_awearnes': {}, 'pred_engagement': {}, 'pred_playcounts': {}, 'pred_songs':{}, 'pred_days':{}}
        for cutoff in ('1', '3', '5', '10', '100'):
            for name in ('pred_awearnes', 'pred_engagement', 'pred_playcounts', 'pred_songs', 'pred_days'):
                res[name][cutoff] = []

        for a, eng, aw, playcounts, different_songs, days  in pred_in_test:
            for j in ['10', '5', '3', '1']:
                if predicted_dict[a] < int(j):
                    res['pred_awearnes'][j].append(aw)
                    res['pred_engagement'][j].append(eng)
                    res['pred_playcounts'][j].append(playcounts)
                    res['pred_songs'][j].append(different_songs)
                    res['pred_days'][j].append(days)
                    if a not in artists_res['pred_awearnes'][j]:
                        artists_res['pred_awearnes'][j][a] = []
                        artists_res['pred_engagement'][j][a]= []
                        artists_res['pred_playcounts'][j][a] = []
                        artists_res['pred_songs'][j][a] = []
                        artists_res['pred_days'][j][a] = []
                    artists_res['pred_awearnes'][j][a].append(aw)
                    artists_res['pred_engagement'][j][a].append(eng)
                    artists_res['pred_playcounts'][j][a].append(playcounts)
                    artists_res['pred_songs'][j][a].append(different_songs)
                    artists_res['pred_days'][j][a].append(days)

        for cutoff in ('1', '3', '5', '10'):
            for name in ('pred_awearnes', 'pred_engagement', 'pred_playcounts', 'pred_songs', 'pred_days'):
                if len(res[name][cutoff]) > 0:
                    #all_res[name][cutoff].append(np.mean(res[name][cutoff]))
                    all_res[name][cutoff].append(np.sum(res[name][cutoff]))

        if len(pred_in_test) > 0:
            all_res['pred_awearnes']['100'].append(np.sum([a[2] for a in pred_in_test]))
            all_res['pred_engagement']['100'].append(np.sum([a[1] for a in pred_in_test]))
            all_res['pred_playcounts']['100'].append(np.sum([a[3] for a in pred_in_test]))
            all_res['pred_songs']['100'].append(np.sum([a[4] for a in pred_in_test]))
            all_res['pred_days']['100'].append(np.sum([a[5] for a in pred_in_test]))


    for name in ('pred_awearnes', 'pred_engagement', 'pred_playcounts', 'pred_songs', 'pred_days'):
       for cutoff in ('1', '3', '5', '10', '100'):
           all_res[name][cutoff] = [np.mean(v) for v in artists_res[name][cutoff].values()]

    for i in range(len(fan_test_data)):
        test_u_sorted_playcount = sorted(fan_test_data[i], key=lambda x: x[1])
        fan_test_data_sorted.append([a[0] for a in test_u_sorted_playcount])
        #fan_test_data_sorted.append(fan_test_data[i])

    metrics = ['map@10', 'precision@1', 'precision@3', 'precision@5', 'precision@10', 'r-precision', 'ndcg@10']
    results, all_results = evaluate(metrics, fan_test_data_sorted, predicted_x)
    gini_val,cov_val,pop_val = evaluate2(predicted_x, item_ids, sum_listen, 10)
    if beta == None:
        print_head = "BETA, "
        for metric in metrics:
            print_head += metric +", "
            if print_pval:
                print_head+= "p-val-"+metric+", "
        print_head+='GINI@10, pop@10, coverage@10'
        print_head+=',test_awearnes,test_engagement,test_playcounts'
        for cutoff in ('1', '3', '5', '10', '100'):
            for name in ('pred_awearnes', 'pred_engagement', 'pred_playcounts', 'pred_songs', 'pred_days'):
                print_head+= ','+name+'@'+cutoff
        print (print_head)
        print_str = "PLAYCOUNTS"
    else:
        print_str = str(beta)
    for metric in metrics:
        print_str+= ', {:.4f}'.format(results[metric])
        if prev_results != None and print_pval:
            stat, pval = wilcoxon(prev_results[metric], all_results[metric])
            print_str+= ', {:.4f}'.format(pval)
        elif print_pval:
            print_str+= ', '

    print_str+= ', {:.4f}, {:.4f}, {:.4f}'.format(gini_val, pop_val, cov_val)

    print_str+= ', {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(np.mean(all_res['test_awearnes']), np.mean(all_res['test_engagement']), np.mean(all_res['test_playcounts']), np.mean(all_res['test_songs']), np.mean(all_res['test_days']))
    for cutoff in ('1', '3', '5', '10', '100'):
        for name in ('pred_awearnes', 'pred_engagement', 'pred_playcounts', 'pred_songs', 'pred_days'):
            print_str+= ', {:.4f}'.format(np.mean(all_res[name][cutoff]))
    """
    all_res['test_awearnes']= np.mean(all_res['test_awearnes'])
    all_res['test_engagement']= np.mean(all_res['test_engagement'])
    all_res['test_playcounts']= np.mean(all_res['test_playcounts'])
    for cutoff in ('1', '3', '5', '10', '100'):
        for name in ('pred_awearnes', 'pred_engagement', 'pred_playcounts'):
            all_res[name][cutoff] = np.mean(all_res[name][cutoff])
    print (all_res)
    """

    print (print_str)
    return all_results


def predict_pop(pop_artists, impl_train_data, N=100):
    predicted = np.zeros((impl_train_data.shape[0],N), dtype=np.uint32)
    for u in range(0, impl_train_data.shape[0]):
        curr_val = 0
        for a in pop_artists:
            if impl_train_data[u,a] == 0:
                predicted[u,curr_val] = a
                curr_val += 1
            if curr_val == 100:
               break 
    return predicted


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Run model training and evaluation.')
    parser.add_argument('-f', "--split_folder", default=False)
    parser.add_argument('-d', "--dims", default=200)
    parser.add_argument('-r', "--reduce_data", default=False)
    parser.add_argument('-p', "--print_pval", default=False)
    parser.add_argument('-a', "--alg", default='als')
    args = parser.parse_args()
    split_folder = args.split_folder
    reduce_data = args.reduce_data
    print_pval = args.print_pval
    alg = args.alg
    dims = int(args.dims)


    print ("Dataset:", split_folder, 'Dimension:', dims, 'Reduce data: ', reduce_data)
    fan_train_data = sparse.load_npz(os.path.join('data', split_folder, 'fan_train_data_playcount_features_fidelity_05.npz')).tocsr()
    fan_train_days= sparse.load_npz(os.path.join('data', split_folder, 'fan_train_data_days_features_fidelity_05.npz')).tocsr()
    orig_fan_train_data = sparse.load_npz(os.path.join('data', split_folder, 'fan_train_data_playcount_features_fidelity_05.npz')).tocsr()

    sum_listen = fan_train_data.sum(axis=0)
    rank_listen = np.argsort(-sum_listen.flatten())
    pop_artists = rank_listen[0,:1000].tolist()[0]
    predicted_pop = predict_pop(pop_artists, fan_train_data)

    transformer = Binarizer().fit(fan_train_data)
    binary_mtrx = transformer.transform(fan_train_data)

    #ft = FunctionTransformer(np.log1p, accept_sparse=True)
    #fan_train_data = ft.fit_transform(fan_train_data)


    if reduce_data == '50':
        row_mask = fan_train_data.data < 3
        row_mask_indices = np.argwhere(fan_train_data.data < 3)
        non_zeros = np.random.choice(row_mask_indices[:,0], size=int(row_mask_indices.shape[0]/2))
        row_mask[non_zeros] = False
        rows = fan_train_data.nonzero()[0][row_mask]
        cols = fan_train_data.nonzero()[1][row_mask]
        fan_train_data[rows,cols] = 0
        binary_mtrx[rows,cols] = 0
    elif reduce_data == '0':
        row_mask_indices = np.argwhere(fan_train_data.data < 3)
        rows = fan_train_data.nonzero()[0][row_mask_indices]
        cols = fan_train_data.nonzero()[1][row_mask_indices]
        fan_train_data[rows,cols] = 0
        binary_mtrx[rows,cols] = 0

    #ft = FunctionTransformer(np.log1p, accept_sparse=True)
    #fan_train_data = ft.fit_transform(fan_train_data)

    #del fan_train_data

    fan_test_data = pickle.load(open(os.path.join('data', split_folder, 'fan_test_data_features_fidelity_05.pkl'), 'rb'))
    fan_items_dict = pickle.load(open(os.path.join('data', split_folder, 'fan_items_dict_features_fidelity_05.pkl'), 'rb'))
    fan_users_dict = pickle.load(open(os.path.join('data', split_folder,'fan_users_dict_features_fidelity_05.pkl'), 'rb'))
    #print ("Item", len(fan_items_dict))
    #print ("User", len(fan_users_dict))

    model_folder = 'models'
    user_features_file = os.path.join(model_folder, split_folder, user_features_playcounts_filename)
    item_features_file = os.path.join(model_folder, split_folder, item_features_playcounts_filename)
    # PLAYCOUNTS evaluation
    if alg == 'als':
        item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_data, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    elif alg=='warp':
        item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_warp(fan_train_data, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    #user_ids, user_vecs_reg = load_feats(user_features_file)
    #item_ids, item_vecs_reg = load_feats(item_features_file)
    predictions_file = os.path.join(model_folder, split_folder,predictions_playcounts_filename)
    predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, orig_fan_train_data, step=500)
    #predicted = np.load(predictions_file)
    prev_results = show_eval(predicted, fan_test_data, item_ids, rank_listen, print_pval=print_pval)

    # BINARY evaluation
    if alg == 'als':
        item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(binary_mtrx, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    elif alg=='warp':
        item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_warp(binary_mtrx, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, orig_fan_train_data, step=500)
    show_eval(predicted, fan_test_data, item_ids, rank_listen, beta="BINARY", print_pval=print_pval)

    item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_days, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, orig_fan_train_data, step=500)
    show_eval(predicted, fan_test_data, item_ids, rank_listen, beta="DAYS", print_pval=print_pval)

    for beta in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:#[::-1]:
        fan_awearnes_data = sparse.load_npz(os.path.join('data', split_folder, 'fan_train_data_awearnes_features_fidelity_05.npz')).tocsr()
        fan_engagement_data = sparse.load_npz(os.path.join('data', split_folder,'fan_train_data_engagement_features_fidelity_05.npz')).tocsr()

        qt = FunctionTransformer(np.log1p, accept_sparse=True)
        qt2 = FunctionTransformer(np.log1p, accept_sparse=True)

        #fan_awearnes_data = qt.fit_transform(fan_awearnes_data.tocsr())
        #fan_engagement_data = qt2.fit_transform(fan_engagement_data.tocsr())

        fan_train_data_fidelity = (fan_awearnes_data*(beta))+((1-beta)*fan_engagement_data)
        if reduce_data != False:
            fan_train_data_fidelity[rows, cols] = 0
            #fan_awearnes_data[rows, cols] = 0
            #fan_engagement_data[rows, cols] = 0


        del qt
        del qt2

        del fan_awearnes_data
        del fan_engagement_data

        user_features_file = os.path.join(model_folder, split_folder, user_fidelity_features_filename.format(beta))
        item_features_file = os.path.join(model_folder, split_folder, item_fidelity_features_filename.format(beta))
        if alg == 'als':
            item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_data_fidelity, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
        elif alg=='warp':
            item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_warp(fan_train_data_fidelity, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
        #user_ids, user_vecs_reg = load_feats(user_features_file)
        #item_ids, item_vecs_reg = load_feats(item_features_file)
        predictions_file = os.path.join(model_folder, split_folder,predictions_fidelity_filename.format(beta))
        predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, orig_fan_train_data, step=500)
        #predicted = np.load(predictions_file)

        show_eval(predicted, fan_test_data,item_ids, sum_listen, beta, prev_results, print_pval)
    #print ("POP: -->")
    #show_eval(predicted_pop, fan_test_data, beta)
       
