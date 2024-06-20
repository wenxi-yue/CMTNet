import torch
from train import Trainer, Finetuner
import os
import argparse
import random
import shutil  
import os.path as osp 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="cholec80")
parser.add_argument('--N', type=int, default=2)
parser.add_argument('--M',type=int,default=3)
parser.add_argument('--margin',type=float,default=0.2)
parser.add_argument('--d_model',type=int,default=128)
args = parser.parse_args()


features_path = osp.join("data",args.dataset,"resnet_features")    
gt_path =  osp.join("data",args.dataset,"groundtruth")    
mapping_file = osp.join("data",args.dataset,"mapping.txt")    
model_dir = osp.join("best_models",args.dataset)   
results_dir = osp.join("results",args.dataset)
checkpoint_dir = osp.join("checkpoints", args.dataset)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# generate action dictionary using the mapping file
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

# generate number of classes from the action dictionary
num_classes = len(actions_dict)

# dictionary to store the final results for each run 1: 2: 3: 4:
results_dict = {}

# repeat the experiments for four different seeds 1 2 3 4 
repeat = 4
for i in range(repeat):
    
    # we simply select seeds = 1,2,3,4 to run experiments and take the average. We use seeds to make it easier to reproduce the results. 
    # we select seed = 1, 2, 3, 4 to show that we did not tune the seed. 
    seed = [1,2,3,4][i]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    print(f"============> Start with Seed {seed} <============")
    
    # define the trainer and fine-tuner for first step training and second step fine-tuning
    trainer = Trainer(args.N, args.M, num_classes, args.dataset, args.margin, args.d_model)
    finetuner = Finetuner(args.N, args.M, num_classes, args.dataset, args.margin, args.d_model)

    # train and then finetune
    n_epoch, l_fit, _ = trainer.train_val(gt_path, features_path, checkpoint_dir, dataset=args.dataset, N = args.N, M = args.M)

    acc,prec,rec,jac = finetuner.finetune_test(gt_path, features_path, checkpoint_dir, model_dir, num_epochs=1000, dataset=args.dataset, n_epoch=n_epoch, l_fit=l_fit, N = args.N, M = args.M, seed=seed)

    # store results into dictionary
    results_dict[str(i)] = [acc,prec,rec,jac]

    # clear checkpoint for next experiment run
    shutil.rmtree(checkpoint_dir)
    
print(results_dict)

# compute the mean results
mean_results = [0,0,0,0]
for _,value in results_dict.items():
    mean_results = [x + y for (x, y) in zip(mean_results, value)]
print([round(j/repeat,1) for j in mean_results])

