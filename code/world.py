
import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'zly_code')
DATA_PATH = join(ROOT_PATH, 'data')
FILE_PATH = join(CODE_PATH, 'checkpoints')
OUTPUT_PATH = join(CODE_PATH, 'outputs')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}

all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['retrieval_type'] = args.retrieval_type
config['retriever_path'] = args.retrieval_path
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['max_seq_length'] = args.max_seq_length
config['phase'] = args.phase
config['topI'] = args.topI
config['topT'] = args.topT
config['add_tool_method'] = 'fussion'
config['load_file'] = args.load_file
config['add_tool'] = args.n_tool
config['add_path'] = args.add_path
config['output_dir'] = args.output_dir
config['all_pseudo_path'] = args.all_pseudo_path
config['without_logic'] = args.without_logic
config['without_query_transfer'] = args.without_query_transfer
config['without_tool_transfer'] = args.without_tool_transfer
config['sudo_no_same']= args.sudo_no_same
config['history']=args.history
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
config['device'] = device
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

