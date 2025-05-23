from collections import OrderedDict
import os
from os.path import join
import sys
import itertools
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time
from sentence_transformers import SentenceTransformer, models
import json
import csv
import random
import pickle
from data.llm import ChatGpt
from tqdm import tqdm
class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.path = path

        qid_query_dict,cid_tool_dict, qid_cid_list_dict = self.initialize_dataset(path,'train')
        if world.config['history'] == 1:
            cid_tool_dict = self.add_history(cid_tool_dict,qid_cid_list_dict,qid_query_dict)
        retiever_model = self.initialize_retriever(world.config['retrieval_type'], world.config['retriever_path'])
        if world.config['add_tool'] > 0:
            if world.config['add_tool_method'] == 'fussion':
                if world.config['phase'] != 'orig':
                    old_qid_query_dict,old_cid_tool_dict,old_qid_cid_list_dict = qid_query_dict.copy(),cid_tool_dict.copy(), qid_cid_list_dict.copy()  
                    new_qid_query_dict, new_cid_tool_dict, new_qid_cid_list_dict  = self.delete_add_tool(cid_tool_dict,qid_cid_list_dict,qid_query_dict,retiever_model)
                    self.process_sudo_query(retiever_model)
                    trainUserText,ItemText =self.process_graph(new_qid_query_dict, new_cid_tool_dict,new_qid_cid_list_dict)            
                    self.qid_cid_list_dict = {k:new_qid_cid_list_dict[k] for k in sorted(list(new_qid_cid_list_dict.keys()))}

                elif world.config['phase'] == 'orig':
                    self.qid_cid_list_dict = qid_cid_list_dict
                    trainUserText,ItemText = self.process_graph(qid_query_dict, cid_tool_dict, qid_cid_list_dict)
            elif  world.config['add_tool_method'] == 'complete':
                self.qid_cid_list_dict = qid_cid_list_dict
                trainUserText,ItemText = self.process_graph(qid_query_dict, cid_tool_dict, qid_cid_list_dict)
        elif world.config['add_tool'] == 0:
            self.qid_cid_list_dict = qid_cid_list_dict
            trainUserText,ItemText = self.process_graph(qid_query_dict, cid_tool_dict, qid_cid_list_dict)
        self.testQueryDict,self.testLabelDict = self.__build_test()
        testUserText = [value for key, value in sorted(self.testQueryDict.items(), key=lambda item: item[0])]
        self.getInitialEmbeding(trainUserText,testUserText,ItemText)
        
        print(f"{world.dataset} is ready to go")
    def add_history(self,cid_tool_dict,qid_cid_list_dict,qid_query_dict):
        cid_qid_list_dict = {}
        for qid in qid_cid_list_dict.keys():
            for cid in qid_cid_list_dict[qid]:
                if cid not in cid_qid_list_dict:
                    cid_qid_list_dict[cid] = []
                cid_qid_list_dict[cid].append(qid)
        if world.config['add_tool'] == 0:
            
            # print(cid_qid_list_dict.keys())
            for cid,tool in cid_tool_dict.items():
                queries = []
                try:
                    for qid in cid_qid_list_dict[cid]:
                        # print(qid_query_dict[qid])
                        queries.append(qid_query_dict[qid])
                    cid_tool_dict[cid] = tool  + "Usage example:\n" + '\n'.join([str(q) for q in queries])
                    # print(f"add history for {cid}")
                except:
                    print(f"add history for {cid} error")
                    pass
        else:
            unuse_query_list = set()
            index_file = world.config['add_path']
            with open(index_file, 'r') as f:
                self.add_cid_list = sorted(list(map(int,json.load(f))))
            for qid,cid_list in qid_cid_list_dict.items():
                for cid in cid_list:
                    if cid in self.add_cid_list:
                        unuse_query_list.add(qid)
            print(len(cid_tool_dict.keys()))
            for cid,tool in cid_tool_dict.items():
                if cid not in self.add_cid_list:
                    queries = []
                    try:
                        for qid in cid_qid_list_dict[cid]:
                            if qid not in unuse_query_list:
                                queries.append(qid_query_dict[qid])
                    except:
                        print(f"add history for {cid} error")
                        pass
                    cid_tool_dict[cid] = tool + "Usage example:\n" + '\n'.join([str(q) for q in queries])
                else:
                    add_cid_state_file = world.config['add_cid_state_path']
                    with open(add_cid_state_file, 'r') as f:
                        add_cid_state_dict = json.load(f)
                    # for cid in sorted(add_cid_state_dict.keys()):
                    try:
                        state = add_cid_state_dict[str(cid)]
                        queries = state['sudo_query_list']
                        cid_tool_dict[cid] = tool + "Usage example:\n" + '\n'.join([str(q) for q in queries])
                    except:
                        print(f"add history for {cid} error2")
                        pass
        return cid_tool_dict


    def process_graph(self,qid_query_dict,cid_tool_dict, qid_cid_list_dict ):
        self.n_user = 0
        self.m_item = 0
        trainUniqueUsers, trainItem, trainUser,ItemText,trainUserText = [], [], [],[],[]
        self.traindataSize = 0
        self.testDataSize = 0

        trainUserText = [value for key, value in sorted(qid_query_dict.items(), key=lambda item: item[0])]
        ItemText = [value for key, value in sorted(cid_tool_dict.items(), key=lambda item: item[0])]

        for qid, query in qid_query_dict.items():
            trainUniqueUsers.append(qid)
            items = qid_cid_list_dict[qid]
            trainUser.extend([qid] * len(items))
            trainItem.extend([item for item in items])
            self.traindataSize += len(items)

        self.n_user = len(qid_query_dict)
        self.m_item = len(cid_tool_dict)
        print(f"{self.n_user} users and {self.m_item} items in the training set")
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize) / self.n_users / self.m_items}")
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        print(f"UserItemNet shape: {self.UserItemNet.shape}")
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        return trainUserText, ItemText
    def process_sudo_query(self,retiever_model):
        add_cid_state_file = world.config['add_cid_state_path']
        with open(add_cid_state_file, 'r') as f:
            add_cid_state_dict = json.load(f)

        if world.config['sudo_no_same'] == 1:         
            unique_queries = OrderedDict() 
            cid_list_map = {}              
            for cid in sorted(add_cid_state_dict.keys(), key=lambda x: int(x)):
                state = add_cid_state_dict[cid]
                for query, cid_list in zip(state['sudo_query_list'], state['sudo_cid_list']):
                    if query not in unique_queries:
                        idx = len(unique_queries)
                        unique_queries[query] = idx
                        cid_list_map[query] = set(cid_list)
            sudo_queries = list(unique_queries.keys())
            
            print("Generating embeddings for all unique pseudo users...")
            world.config['sudo_query_emb'] = torch.nn.functional.normalize(
                retiever_model.encode(sudo_queries, batch_size=512, convert_to_tensor=True, show_progress_bar=True),
                p=2, dim=1
            )
            
            self.sudo_idx_cid_list_dict = {}
            for query, idx in unique_queries.items():
                self.sudo_idx_cid_list_dict[idx] = cid_list_map[query]
            
            self.add_cid_sudo_query_idx_dict = {}
            for cid in add_cid_state_dict.keys():
                state = add_cid_state_dict[cid]
                cid = int(cid)
                indices = []
                for query in state['sudo_query_list']:
                    indices.append(unique_queries[query])
                self.add_cid_sudo_query_idx_dict[cid] = indices
        else:
            add_cid_state_dict = {int(k): add_cid_state_dict[k] for k in sorted(add_cid_state_dict.keys(),key=lambda x: int(x))}
            sudo_query_list = [add_cid_state_dict[k]['sudo_query_list'] for k in add_cid_state_dict.keys()]
            print("Generating embeddings for all pseudo users")
            sudo_queries = list(itertools.chain.from_iterable(sudo_query_list))
            world.config['sudo_query_emb'] = torch.nn.functional.normalize(retiever_model.encode(sudo_queries, batch_size=512, convert_to_tensor=True, show_progress_bar=True),p=2, dim=1)       
            sudo_query_idx = 0
            self.sudo_idx_cid_list_dict = {}
            self.add_cid_sudo_query_idx_dict = {}
            for cid,state in add_cid_state_dict.items():
                sig_sudo_query_list = state['sudo_query_list']
                sig_sudo_cid_list = state['sudo_cid_list']
                self.add_cid_sudo_query_idx_dict[cid] = []
                for sudo_query,sudo_cid_list in zip(sig_sudo_query_list,sig_sudo_cid_list):
                    self.sudo_idx_cid_list_dict[sudo_query_idx] = sudo_cid_list    
                    self.add_cid_sudo_query_idx_dict[cid].append(sudo_query_idx)
                    sudo_query_idx += 1

    def delete_add_tool(self,cid_tool_dict,qid_cid_list_dict,qid_query_dict,retiever_model):
        index_file = world.config['add_path']
        with open(index_file, 'r') as f:
            self.add_cid_list = sorted(list(map(int,json.load(f))))
            print(self.add_cid_list)
        print(f"Generating initial embeddings for added items")
        add_ItemText = [cid_tool_dict[key] for key in self.add_cid_list]
        world.config['add_tool_emb'] =  torch.nn.functional.normalize(retiever_model.encode(add_ItemText, batch_size=512, convert_to_tensor=True, show_progress_bar=True),p=2, dim=1)
        for cid in self.add_cid_list:
            del cid_tool_dict[cid]
        queries_to_remove = set()
        for qid, cid_list in qid_cid_list_dict.items():
            if any(cid in self.add_cid_list for cid in cid_list):
                queries_to_remove.add(qid)  
        print(f"Number of queries to remove: {len(queries_to_remove)}")
        for qid in queries_to_remove:
            del qid_cid_list_dict[qid]
            del qid_query_dict[qid]

        ocid_ncid_dict = {old_cid: new_cid for new_cid, old_cid in enumerate(cid_tool_dict.keys())}
        self.ncid_ocid_dict = {new_cid: old_cid for old_cid, new_cid in ocid_ncid_dict.items()}
        new_cid_tool_dict = {ocid_ncid_dict[old_cid]: value for old_cid, value in cid_tool_dict.items()}   
        
        oqid_nqid_dict = {old_qid: new_qid for new_qid, old_qid in enumerate(qid_query_dict.keys())}
        new_qid_query_dict = {oqid_nqid_dict[old_qid]: value for old_qid, value in qid_query_dict.items()}
        
        new_qid_cid_list_dict = {}
        for old_qid, old_cid_list in qid_cid_list_dict.items():
            new_qid = oqid_nqid_dict.get(old_qid)
            new_cid_list = [ocid_ncid_dict[cid] for cid in old_cid_list]
            new_qid_cid_list_dict[new_qid] = new_cid_list
        return new_qid_query_dict, new_cid_tool_dict, new_qid_cid_list_dict
    def initialize_dataset(self,data_dir,phase):
        """
        return  ir_query,qid->query,
                ir_corpus, cid->tool 
                ir_relevant_docs qid->cid
        """
        if phase == 'train':
            if world.config['add_tool'] > 0:
                add_cid_list_file = world.config['add_path']
                add_cid_name = os.path.basename(add_cid_list_file).split('.')[0]
                if world.config['add_tool_method'] == 'complete':
                    if 'percent' in add_cid_list_file:
                        queries_df = pd.read_csv(os.path.join(data_dir,'percent',str(world.config['add_tool']),add_cid_name,f'sudo_{phase}.query.txt'), quoting=csv.QUOTE_NONE, sep='\t', names=['qid', 'query'])
                        labels_df = pd.read_csv(os.path.join(data_dir,'percent',str(world.config['add_tool']),add_cid_name,f'sudo_qrels.{phase}.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
                    elif 'chunk' in add_cid_list_file:
                        queries_df = pd.read_csv(os.path.join(data_dir,'chunk',str(world.config['add_tool']),add_cid_name,f'sudo_{phase}.query.txt'), quoting=csv.QUOTE_NONE, sep='\t', names=['qid', 'query'])
                        labels_df = pd.read_csv(os.path.join(data_dir,'chunk',str(world.config['add_tool']),add_cid_name,f'sudo_qrels.{phase}.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
                elif world.config['add_tool_method'] == 'fussion':
                    add_cid_list_file = world.config['add_path']
                    add_cid_name = os.path.basename(add_cid_list_file).split('.')[0]
                    queries_df = pd.read_csv(os.path.join(data_dir,f'{phase}.query.txt'), quoting=csv.QUOTE_NONE, sep='\t', names=['qid', 'query'])
                    labels_df = pd.read_csv(os.path.join(data_dir, f'qrels.{phase}.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
                    if 'percent' in add_cid_list_file:
                        world.config['add_cid_state_path'] = os.path.join(data_dir,'percent',str(world.config['add_tool']),add_cid_name,'generate_cid_sudo_dict.json')
                    else:
                        world.config['add_cid_state_path'] = os.path.join(data_dir,'chunk',str(world.config['add_tool']),add_cid_name,'generate_cid_sudo_dict.json')
            elif world.config['add_tool'] == 0:
                queries_df = pd.read_csv(os.path.join(data_dir,f'{phase}.query.txt'), quoting=csv.QUOTE_NONE, sep='\t', names=['qid', 'query'])
                labels_df = pd.read_csv(os.path.join(data_dir, f'qrels.{phase}.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
        elif phase == "test":
            queries_df = pd.read_csv(os.path.join(data_dir,f'{phase}.query.txt'), quoting=csv.QUOTE_NONE, sep='\t', names=['qid', 'query'])
            labels_df = pd.read_csv(os.path.join(data_dir, f'qrels.{phase}.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
        qid_query_dict = {}
        for row in queries_df.itertuples():
            qid_query_dict[int(row.qid)] = row.query
        qid_cid_list_dict = {}
        cid_qid_list_dict = {}
        for row in labels_df.itertuples():
            qid_cid_list_dict.setdefault(int(row.qid), set()).add(int(row.docid))
        
        cid_tool_dict = {}
        documents_df = pd.read_csv(os.path.join(data_dir,'corpus.tsv'), sep='\t')
        if 'bank' or 'UltraTool' or 'utral' in data_dir:
            for row in documents_df.itertuples():
                cid_tool_dict[int(row.docid)] = row.document_content
        else:
            cid_tool_dict,_= self.process_retrieval_ducoment(documents_df)
        
        
        return qid_query_dict, cid_tool_dict, qid_cid_list_dict


    def initialize_retriever(self,retriever_type,retriever_path):
        if retriever_type == 'bert':
            print(retriever_path)
            word_embedding_model = models.Transformer(retriever_path, max_seq_length=world.config['max_seq_length'])
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            reward_model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device=world.config['device'])
            return reward_model
        # elif retriever_type == 'ada':

    def process_retrieval_ducoment(self,documents_df):
        ir_corpus = {}
        corpus2tool = {}
        for row in documents_df.itertuples():
            # print(row)
            doc = json.loads(row.document_content)
            ir_corpus[row.docid] = (doc.get('category_name', '') or '') + ', ' + \
            (doc.get('tool_name', '') or '') + ', ' + \
            (doc.get('api_name', '') or '') + ', ' + \
            (doc.get('api_description', '') or '') + \
            ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
            ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
            ', return_schema: ' + json.dumps(doc.get('template_response', ''))
            corpus2tool[(doc.get('category_name', '') or '') + ', ' + \
            (doc.get('tool_name', '') or '') + ', ' + \
            (doc.get('api_name', '') or '') + ', ' + \
            (doc.get('api_description', '') or '') + \
            ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
            ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
            ', return_schema: ' + json.dumps(doc.get('template_response', ''))] = doc['category_name'] + '\t' + doc['tool_name'] + '\t' + doc['api_name']
        return ir_corpus, corpus2tool

    def getInitialEmbeding(self, train_users, test_users, items):
        if world.config['retrieval_type'] == 'bert':
            retiever_model = self.initialize_retriever(world.config['retrieval_type'], world.config['retriever_path'])        
            print(f"Generating initial embeddings for training users")
            train_query_emb = retiever_model.encode(train_users, batch_size=512, convert_to_tensor=True, show_progress_bar=True)
            train_query_emb = torch.nn.functional.normalize(train_query_emb, p=2, dim=1)
            world.config['train_query_emb'] = train_query_emb
            print(f"Generating initial embeddings for test users")
            test_query_emb = retiever_model.encode(test_users, batch_size=512, convert_to_tensor=True, show_progress_bar=True)
            test_query_emb = torch.nn.functional.normalize(test_query_emb, p=2, dim=1)
            world.config['test_query_emb'] = test_query_emb
            item_emb = retiever_model.encode(items, batch_size=512, convert_to_tensor=True, show_progress_bar=True)
            item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)
            world.config['item_emb'] = item_emb
        elif world.config['retrieval_type'] == 'ada':
            print(f"Generating initial embeddings for training users")
            test_query_emb = self.create_ada_embedding(test_users,phase='test').to(world.device)
            train_query_emb = self.create_ada_embedding(train_users,phase='train').to(world.device)  
            item_emb = self.create_ada_embedding(items,phase='tool').to(world.device)
            print(train_query_emb.shape)
            train_query_emb = torch.nn.functional.normalize(train_query_emb, p=2, dim=1)
            
            world.config['train_query_emb'] = train_query_emb
            test_query_emb = torch.nn.functional.normalize(test_query_emb, p=2, dim=1)
            world.config['test_query_emb'] = test_query_emb
            item_emb = torch.nn.functional.normalize(item_emb, p=2, dim=1)
            world.config['item_emb'] = item_emb
    def create_ada_embedding(self,text_list,phase='train',type='save'):
        batch_size = 64
        ada_path = os.path.join(self.path,f'{phase}_ada_embeding.json')
        if os.path.exists(ada_path):
            print(f'load embedding {ada_path}')
            with open(ada_path, 'r') as f:
                id_emb_dict = json.load(f)  
            sorted_values = id_emb_dict.values()           
        else:
            id_emb_dict = {}
            model =  ChatGpt("text-embedding-ada-002")
            print(f'start to generate embedding {ada_path}')
            for i in tqdm(range(0,len(text_list),batch_size)):
                batch_text = text_list[i:min(i+batch_size,len(text_list))]
                batch_id = list(range(i, min(i+batch_size, len(text_list))))
                batch_id_text_dict = dict(zip(batch_id,batch_text))
                model.batch_embedding_generate(batch_id_text_dict,id_emb_dict)
            sorted_id_list = sorted(id_emb_dict.keys())
            sorted_emb_list = [id_emb_dict[i] for i in sorted_id_list]
            sorted_id_emb_dict = dict(zip(sorted_id_list,sorted_emb_list))
            with open(ada_path, 'w') as f:
                json.dump(sorted_id_emb_dict, f,indent=4)
            sorted_values = sorted_id_emb_dict.values()
        torch_sorted_values = [torch.tensor(emb, dtype=torch.float) for emb in sorted_values]
        concatenated_embedding_torch = torch.stack(torch_sorted_values, dim=0)
        return concatenated_embedding_torch
    #    
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        
    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {qid:query}
            dict: {qid: [cids]}
        """
        test_ir_query,_, test_ir_relevant_docs = self.initialize_dataset(self.path,'test')
        return test_ir_query,test_ir_relevant_docs

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems