
import world
import numpy as np
import torch
import utils
from pprint import pprint
from utils import timer,getFileName,saveOutucome
from time import time
import model
import multiprocessing
import os
import json

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    # print(X)
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    similarities = X[2]  
    # print(sorted_items)
    # print(groundTrue)
    if sorted_items.ndim == 1:
        sorted_items = sorted_items.reshape(1, -1) 
    if type(groundTrue) == set:
        groundTrue_list = []
        groundTrue_list.append(groundTrue)
        groundTrue = groundTrue_list
    # if groundTrue.ndim == 1:
    #     groundTrue = groundTrue.reshape(1, -1)  
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg),
            'similarities': similarities}  
        
            
def Test(dataset, Recmodel, epoch, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testQueryDict: dict = dataset.testQueryDict
    testLabelDict: dict = dataset.testLabelDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    all_dict ={}
    with torch.no_grad():
        users = list(testQueryDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        similarity_list = [] 
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            groundTrue = [testLabelDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            rating = Recmodel.getUsersRating(batch_users_gpu)
            similarities = rating.cpu().numpy() 
            rating_K, rating_K_indices = torch.topk(rating, k=max_K) 
            rating = rating.cpu().numpy()
            del rating       
            for user,gt,rat,sim in zip(batch_users,groundTrue,rating_K_indices.cpu(),similarities):
                result = test_one_batch((rat,gt,sim))
                all_dict[user] = {}
                all_dict[user]['recall'] = result['recall'].tolist()
                all_dict[user]['precision'] = result['precision'].tolist()
                all_dict[user]['ndcg'] = result['ndcg'].tolist()
                all_dict[user]['rating'] = rat.tolist()
                all_dict[user]['groundTrue'] = list(gt)
        
                gt_similarities = {}
                for tool_id in gt:
                    if tool_id < len(sim):  
                        gt_similarities[str(tool_id)] = float(sim[tool_id])
                all_dict[user]['gt_similarities'] = gt_similarities
            users_list.append(batch_users)
            rating_list.append(rating_K_indices.cpu())
            groundTrue_list.append(groundTrue)
            similarity_list.append(similarities)  
        X = zip(rating_list, groundTrue_list, similarity_list)  
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        json_file = os.path.join(world.config['output_dir'],f"{world.config['phase']}_Retrieval_evaluation_results.csv")
        saveOutucome(epoch,results,json_file)
        all_dict_file = os.path.join(world.config['output_dir'],f"{world.config['phase']}_epoch_{epoch}_all_results.json")
        with open(all_dict_file,'w') as f:
            json.dump(all_dict,f)
        if multicore == 1:
            pool.close()
        print(results)
        return results
