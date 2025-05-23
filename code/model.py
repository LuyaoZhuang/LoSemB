import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
from sentence_transformers.util import cos_sim
import random
import pickle
from collections import Counter
from collections import defaultdict

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getquerysRating(self, querys):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, querys, pos, neg):
        """
        Parameters:
            querys: querys list 
            pos: positive items for corresponding querys
            neg: negative items for corresponding querys
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  

class PureMF(BasicModel):
    # This method does not utilize text relationships, only uses item and query interaction history.
    # While this is reasonable for recommendation scenarios, it is not suitable for tool retrieval scenarios.
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_querys  = dataset.n_querys
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_query = torch.nn.Embedding(
            num_embeddings=self.num_querys, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getquerysRating(self, querys):
        querys = querys.long()
        querys_emb = self.embedding_query(querys)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(querys_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, querys, pos, neg):
        querys_emb = self.embedding_query(querys.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(querys_emb*pos_emb, dim=1)
        neg_scores= torch.sum(querys_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(querys_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(querys))
        return loss, reg_loss
        
    def forward(self, querys, items):
        querys = querys.long()
        items = items.long()
        querys_emb = self.embedding_query(querys)
        items_emb = self.embedding_item(items)
        scores = torch.sum(querys_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        if world.config['add_tool'] > 0 and world.config['phase'] != 'orig':
            self.add_tool_idx_list = self.dataset.add_cid_list
            self.sudo_text_query_emb = world.config['sudo_query_emb']
            self.add_text_tool_emb = world.config['add_tool_emb']   
        self.train_text_query_emb = world.config['train_query_emb']
        self.text_tool_emb = world.config['item_emb']
    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.save = False
        self.batch_count = 0
        self.embedding_query = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_query.weight.data.copy_(self.config['train_query_emb'])
        self.embedding_item.weight.data.copy_(self.config['item_emb'])
        print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")
        # if wo
        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g 
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    def complete_final_query_emb(self,train_graph_query_emb):
        if world.config['without_tool_transfer'] == 0:
            sudo_query_rating = torch.matmul(self.sudo_text_query_emb, self.train_text_query_emb.t())
            sudo_query_sim_score,sudo_query_rating_I = torch.topk(sudo_query_rating, world.config['topI'])
            sudo_diff = train_graph_query_emb[sudo_query_rating_I] - self.train_text_query_emb[sudo_query_rating_I]
            sudo_weights = torch.nn.functional.softmax(sudo_query_sim_score, dim=1)
            sudo_weights = sudo_weights.unsqueeze(-1)
            sudo_weighted_diff=(sudo_diff *sudo_weights).sum(dim=1)
            sudo_graph_query_emb = torch.nn.functional.normalize((sudo_weighted_diff + self.sudo_text_query_emb),p=2,dim=1)
            final_graph_query_emb = torch.cat([train_graph_query_emb, sudo_graph_query_emb], dim=0)
            final_text_query_emb = torch.cat([self.train_text_query_emb, self.sudo_text_query_emb], dim=0)
        else:
            final_graph_query_emb = torch.cat([train_graph_query_emb, self.sudo_text_query_emb])
            final_text_query_emb = torch.cat([self.train_text_query_emb, self.sudo_text_query_emb], dim=0)
        return final_graph_query_emb,final_text_query_emb
    def complete_final_tool_emb(self,graph_tool_emb):
        add_graph_tool_emb = []
        for add_cid, tool_emb in zip(self.add_tool_idx_list, self.add_text_tool_emb):
            per_sudo_text_query_emb = self.sudo_text_query_emb[self.dataset.add_cid_sudo_query_idx_dict[add_cid]]
            sudo_query_rating = torch.matmul(per_sudo_text_query_emb, self.train_text_query_emb.t())
            _,sudo_query_rating_I = torch.topk(sudo_query_rating, world.config['topI'])
            used_tools = []
            for candidate_query_group in sudo_query_rating_I:
                for train_query_idx in candidate_query_group:
                    idx = int(train_query_idx)
                    used_tools.append(self.dataset.qid_cid_list_dict[idx])

            all_tools = [tool for sublist in used_tools for tool in sublist]
            tool_diffs = graph_tool_emb[all_tools] - self.text_tool_emb[all_tools]
            aggregated_diff = torch.mean(tool_diffs, dim=0)
            add_graph_tool_emb.append(tool_emb + aggregated_diff)

        add_graph_tool_emb = torch.nn.functional.normalize(torch.stack(add_graph_tool_emb), p=2, dim=1)
        final_graph_tool_emb = torch.zeros(
            graph_tool_emb.size(0) + add_graph_tool_emb.size(0), 
            graph_tool_emb.size(1), 
            device=graph_tool_emb.device
        )
        remaining_indices = [idx for idx in range(len(final_graph_tool_emb)) if idx not in self.add_tool_idx_list]
        final_graph_tool_emb[remaining_indices] = graph_tool_emb
        final_graph_tool_emb[self.add_tool_idx_list] = add_graph_tool_emb

        final_text_tool_emb = torch.zeros(
            self.text_tool_emb.size(0) + self.add_text_tool_emb.size(0), 
            self.text_tool_emb.size(1), 
            device=self.text_tool_emb.device
        )
        final_text_tool_emb[remaining_indices] = self.text_tool_emb
        final_text_tool_emb[self.add_tool_idx_list] = self.add_text_tool_emb
        return final_graph_tool_emb, final_text_tool_emb
    def complete_test_graph_query_emb(self,test_text_query_emb,train_text_query_emb,train_graph_query_emb):

        test_query_rating = torch.matmul(test_text_query_emb, train_text_query_emb.t())
        test_similarity_scores, test_query_rating_I = torch.topk(test_query_rating, world.config['topI'])
        test_diff =  train_graph_query_emb[test_query_rating_I] - train_text_query_emb[test_query_rating_I]
        test_weights = torch.nn.functional.softmax(test_similarity_scores, dim=1)
        test_weights = test_weights.unsqueeze(-1)
        test_weighted_diff=(test_diff *test_weights).sum(dim=1)
        test_graph_query_emb = torch.nn.functional.normalize((test_weighted_diff + test_text_query_emb),p=2,dim=1)
        return test_graph_query_emb,test_query_rating
    def mask_rating(self, rating, test_instruction_rating):
        test_similarity_scores, test_query_rating_I = torch.topk(test_instruction_rating,world.config['topT'] )
        for i, idx_list in enumerate(test_query_rating_I):
            current_candidates = set()                  
            for idx in idx_list:
                idx = idx.item()              
                if idx < len(self.dataset.qid_cid_list_dict):
                    for item in self.dataset.qid_cid_list_dict[idx]:
                        if world.config['add_tool'] == 0:
                            current_candidates.add(item)
                        elif world.config['add_tool'] > 0:
                            if world.config['add_tool_method'] == 'complete':
                                current_candidates.add(item)
                            elif world.config['add_tool_method'] == 'fussion':
                                current_candidates.add(self.dataset.ncid_ocid_dict[int(item)])
                else:
                    new_idx = idx - len(self.dataset.qid_cid_list_dict)
                    cid_list = self.dataset.sudo_idx_cid_list_dict[new_idx]
                    for item in cid_list:
                        # print('add sudo item', item)
                        current_candidates.add(item)
                    # cid_list = self.dataset.sudo_idx_cid_list_dict[new_idx]['cids']
                    # current_candidates.add(tuple(cid_list))
            mask = torch.zeros_like(rating[i], dtype=torch.bool) 
            mask[list(current_candidates)] = 1
            rating[i] = rating[i] * mask.float()           
        return rating
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        querys_emb = self.embedding_query.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([querys_emb, items_emb])
        if world.config['phase'] != 'orig':       
            embs = [all_emb]
            if self.config['dropout']:
                if self.training:
                    print("droping")
                    g_droped = self.__dropout(self.keep_prob)
                else:
                    g_droped = self.Graph        
            else:
                g_droped = self.Graph    
            for layer in range(self.n_layers):
                if self.A_split:
                    temp_emb = []
                    for f in range(len(g_droped)):
                        temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                    side_emb = torch.cat(temp_emb, dim=0)
                    all_emb = side_emb
                else:
                    all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            if not self.save and self.config['phase'] == 'test':
                self.save_embedding(users,items)
        else:
            users, items = torch.split(all_emb, [self.num_users, self.num_items])
        
        return users, items
    
    def getUsersRating(self, users):
        
        test_text_query_emb = world.config['test_query_emb'][users.long()]
        
        if world.config['phase'] == 'orig':
            rating = self.f(torch.matmul(test_text_query_emb, self.text_tool_emb.t()))
        elif world.config['phase'] != 'orig':
            train_graph_query_emb,graph_tool_emb = self.computer()   
            if world.config['add_tool'] > 0:
                if world.config['add_tool_method'] == 'complete':
                    if world.config['phase'] == 'gnn-off':
                        test_instruction_rating = torch.matmul(test_text_query_emb, self.train_text_query_emb.t())
                        rating = self.mask_rating(cos_sim(test_text_query_emb, self.text_tool_emb),test_instruction_rating)    
                    elif world.config['phase'] == 'train-off':
                        test_graph_query_emb, test_instruction_rating= self.complete_test_graph_query_emb(test_text_query_emb,self.train_text_query_emb,train_graph_query_emb)
                        rating = self.mask_rating(cos_sim(test_graph_query_emb, graph_tool_emb),test_instruction_rating)                                            
                    test_text_query_emb = torch.cat([test_text_query_emb,train_graph_query_emb[users.long()]],dim=1)
                elif world.config['add_tool_method'] == 'fussion':
                    if world.config['phase'] == 'gnn-off':
                        _,final_text_query_emb = self.complete_final_query_emb(train_graph_query_emb)
                        _,final_text_tool_emb =self.complete_final_tool_emb(graph_tool_emb)
                        test_instruction_rating = torch.matmul(test_text_query_emb, final_text_query_emb.t())
                        rating = self.mask_rating(cos_sim(test_text_query_emb, final_text_tool_emb),test_instruction_rating)    
                 
                    elif world.config['phase'] in ['train-off','train']:
                        final_graph_query_emb,final_text_query_emb = self.complete_final_query_emb(train_graph_query_emb)
                        final_graph_tool_emb,final_text_tool_emb = self.complete_final_tool_emb(graph_tool_emb)
                        test_graph_query_emb, test_instruction_rating = self.complete_test_graph_query_emb(test_text_query_emb,final_text_query_emb,final_graph_query_emb)
                        if world.config['without_logic'] == 1 and world.config['without_query_transfer'] == 1 and world.config['without_tool_transfer'] == 1:
                            final_combined_tool_emb = torch.zeros(
                                final_graph_tool_emb.size(), 
                                device=graph_tool_emb.device
                            )
                            remaining_indices = [idx for idx in range(len(final_combined_tool_emb)) if idx not in self.add_tool_idx_list]
                            final_combined_tool_emb[remaining_indices] = graph_tool_emb
                            final_combined_tool_emb[self.add_tool_idx_list] = self.add_text_tool_emb
                            rating = cos_sim(test_text_query_emb,final_combined_tool_emb)
                        elif world.config['without_logic'] == 0 and world.config['without_query_transfer'] == 1 and world.config['without_tool_transfer'] == 1:
                            final_combined_tool_emb = torch.zeros(
                                final_graph_tool_emb.size(), 
                                device=graph_tool_emb.device
                            )
                            remaining_indices = [idx for idx in range(len(final_combined_tool_emb)) if idx not in self.add_tool_idx_list]
                            final_combined_tool_emb[remaining_indices] = graph_tool_emb
                            final_combined_tool_emb[self.add_tool_idx_list] = self.add_text_tool_emb
                            rating = self.mask_rating(cos_sim(test_text_query_emb,final_combined_tool_emb),test_instruction_rating)
                        elif world.config['without_logic'] == 0 and world.config['without_query_transfer'] == 0 and world.config['without_tool_transfer'] == 1:
                            final_combined_tool_emb = torch.zeros(
                                final_graph_tool_emb.size(), 
                                device=graph_tool_emb.device
                            )
                            remaining_indices = [idx for idx in range(len(final_combined_tool_emb)) if idx not in self.add_tool_idx_list]
                            final_combined_tool_emb[remaining_indices] = graph_tool_emb
                            final_combined_tool_emb[self.add_tool_idx_list] = self.add_text_tool_emb
                            rating = self.mask_rating(cos_sim(test_graph_query_emb,final_combined_tool_emb),test_instruction_rating)
                        elif world.config['without_logic'] == 0 and world.config['without_query_transfer'] == 1 and world.config['without_tool_transfer'] == 0:
                            rating = self.mask_rating(cos_sim(test_text_query_emb,final_graph_tool_emb),test_instruction_rating)
                        elif world.config['without_logic'] == 1 and world.config['without_query_transfer'] == 0 and world.config['without_tool_transfer'] == 0:
                            rating = cos_sim(test_graph_query_emb,final_graph_tool_emb)
                        elif world.config['without_logic'] == 1 and world.config['without_query_transfer'] == 1 and world.config['without_tool_transfer'] == 0:
                            rating = cos_sim(test_text_query_emb,final_graph_tool_emb)
                        elif world.config['without_logic'] == 1 and world.config['without_query_transfer'] == 0 and world.config['without_tool_transfer'] == 1:
                            rating = cos_sim(test_graph_query_emb,final_text_tool_emb)
                        elif world.config['without_logic'] == 0 and world.config['without_query_transfer'] == 1 and world.config['without_tool_transfer'] == 1:
                            rating = self.mask_rating(cos_sim(test_text_query_emb,final_text_tool_emb),test_instruction_rating)
                        elif world.config['without_logic'] == 0 and world.config['without_query_transfer'] == 0 and world.config['without_tool_transfer'] == 0:
                            rating = self.mask_rating(cos_sim(test_graph_query_emb,final_graph_tool_emb),test_instruction_rating)
                        
            elif world.config['add_tool'] == 0:
                if world.config['phase'] == 'gnn-off':
                    test_instruction_rating = torch.matmul(test_text_query_emb, self.train_text_query_emb.t())
                    rating = self.mask_rating(cos_sim(test_text_query_emb, self.text_tool_emb),test_instruction_rating)    
                elif world.config['phase'] in ['train-off','train']:
                    test_graph_query_emb, test_instruction_rating= self.complete_test_graph_query_emb(test_text_query_emb,self.train_text_query_emb,train_graph_query_emb)
                    if world.config['without_logic'] == 1 and world.config['without_query_transfer'] == 1:
                        rating = cos_sim(test_text_query_emb,graph_tool_emb)
                    elif world.config['without_logic'] == 0 and world.config['without_query_transfer'] == 1:
                        rating = self.mask_rating(cos_sim(test_text_query_emb,graph_tool_emb),test_instruction_rating)                   
                    elif world.config['without_logic'] == 1 and world.config['without_query_transfer'] == 0:
                        rating = cos_sim(test_graph_query_emb,graph_tool_emb)
                    else:
                        rating = self.mask_rating(cos_sim(test_graph_query_emb,graph_tool_emb),test_instruction_rating)      
        return rating
    
    def getEmbedding(self, querys, pos_items, neg_items):
        all_querys, all_items = self.computer()
        querys_emb = all_querys[querys]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        querys_emb_ego = self.embedding_query(querys)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return querys_emb, pos_emb, neg_emb, querys_emb_ego, pos_emb_ego, neg_emb_ego
  
    def bpr_loss(self, querys, pos, neg):
        (querys_emb, pos_emb, neg_emb, 
        queryEmb0,  posEmb0, negEmb0) = self.getEmbedding(querys.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(queryEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(querys))
        pos_scores = torch.mul(querys_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(querys_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, querys, items):
        all_querys, all_items = self.computer()
        querys_emb = all_querys[querys]
        items_emb = all_items[items]
        inner_pro = torch.mul(querys_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    def save_embedding(self,querys,items):
        query_path = world.config['load_file'].replace('pth','_query.pkl')
        item_path = world.config['load_file'].replace('pth','_item.pkl')
        with open(query_path,'wb') as u_f:
            pickle.dump(querys,u_f)
        with open(item_path,'wb') as i_f:
            pickle.dump(items,i_f)
        self.save = True

