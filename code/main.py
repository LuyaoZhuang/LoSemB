import world
import utils
from world import cprint
import torch
import numpy as np
import Procedure
from os.path import join
import os
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
os.environ["TOKENIZERS_PARALLELISM"] = "false"
Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = world.config['output_dir']
print(f"load and save to {weight_file}")
Neg_k = 1

if world.config['phase'] == 'train':
    for epoch in range(world.TRAIN_epochs):
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch,world.config['multicore'])
            torch.save(Recmodel.state_dict(), os.path.join(weight_file, f"epoch_{epoch}.pth"))
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        
elif world.config['phase'] == 'test':
    cprint("[TEST]")
    Recmodel.load_state_dict(torch.load(world.config['load_file']))
    world.cprint(f"loaded model weights from {weight_file}")
    Procedure.Test(dataset, Recmodel, 0, world.config['multicore'])
    
elif world.config['phase'] == 'orig' or world.config['phase'] == 'train-off' or world.config['phase'] == 'gnn-off' :
    cprint("[TEST]")
    Procedure.Test(dataset, Recmodel, 0, world.config['multicore'])