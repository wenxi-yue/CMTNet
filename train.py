import torch
import torch.nn as nn
from torch import optim
import numpy as np
from data import create_dataloader
from model import CMTNet
from loss import refinement_loss
from eval import official_evaluation
import os.path as osp


class Trainer:
    def __init__(self, N, M, num_classes, dataset, margin, d_model):

        self.model = CMTNet(N = N, d_model=d_model, d_ff=d_model, num_classes = num_classes, M=M)
        
        if dataset == "cholec80":
            loss_weight = [
            1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618]

        elif dataset == "m2cai":
            loss_weight = [
            0.2639588012080849,    
            0.14090451483001626,
            1.0,
            0.38530937814605437,
            0.8951057074266243,
            0.1306822581894215,
            0.4749477270967242,
            0.3697049485015101]

        loss_weight  = torch.from_numpy(np.asarray(loss_weight)).float().cuda()

        self.ce = nn.CrossEntropyLoss(weight = loss_weight) 
        self.refinement_loss = refinement_loss(margin)
        
        self.num_classes = num_classes
        

    def train_val(self, gt_path, features_path, save_dir, dataset, N, M):
        train_dataloader, val_dataloader, train_size, val_size = create_dataloader(gt_dir= gt_path, feat_dir=features_path, dataset=dataset, phase="train") 
                   
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        
        
        self.model.cuda()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        
        for epoch in range(1,2):
            epoch_loss = 0
            correct = 0
            total = 0
        
            self.model.train()
            for batch_input,batch_target in train_dataloader:

                with torch.set_grad_enabled(True):
                    batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
                    optimizer.zero_grad()
                    
                    batch_input = torch.transpose(batch_input,1,2).contiguous()
                    _,T,_ = batch_input.size()
                    predictions = self.model(batch_input).squeeze(1)
                    
                    loss = 0
                    for p in predictions:
                        loss += self.ce(p, batch_target.view(-1))                        
                        
                    loss += self.refinement_loss(predictions,batch_target)[0]

                    for i in range(M):
                        for j in range(N):
                            for v in range(2):
                                atten = self.model.stages[i].phaseencoder.layers[j].self_attn.attn.squeeze(0).squeeze(-2)[v]
                                attention_loss = self.ce(atten, batch_target.view(-1)[::4])                   
                                loss += attention_loss*0.04
                    
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float().squeeze(1)).sum().detach().item()
                total += T 

            print(("TRAIN: [epoch %d]: loss = %f,   acc = %f" % (epoch, epoch_loss / train_size, float(correct)/total)))
            
            train_loss = epoch_loss / train_size
            train_accuracy = float(correct)/total

            # begin validation
            epoch_loss = 0
            correct = 0
            total = 0
            
            self.model.eval()
            with torch.no_grad():
                for batch_input,batch_target in val_dataloader:
                    batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
                    
                    batch_input = torch.transpose(batch_input,1,2).contiguous()
                    _,T,_ = batch_input.size()
                    predictions  = self.model(batch_input).squeeze(1)
                    loss = 0
                    
                    for p in predictions:
                        stage_cls_loss = self.ce(p.view(-1, self.num_classes), batch_target.view(-1)) 
                        loss += stage_cls_loss
                        
                    loss += self.refinement_loss(predictions,batch_target)[0]

                    for i in range(M):
                        for j in range(N):
                            for v in range(2): # number of heads
                                
                                atten = self.model.stages[i].phaseencoder.layers[j].self_attn.attn.squeeze(0).squeeze(-2)[v]
                                attention_loss = self.ce(atten, batch_target.view(-1)[::4]) 
                                loss += attention_loss*0.04


                    epoch_loss += loss.detach().item()
                    
                    _, predicted = torch.max(predictions[-1].data, 1)

                    correct += ((predicted == batch_target).float().squeeze(1)).sum().detach().item()
                    total += T

                    
            print(("VAL: [epoch %d]: loss = %f,   acc = %f" % (epoch, epoch_loss / val_size, float(correct)/total)))
                
            if float(correct)/total > self.model.best_acc:
                self.model.best_acc = float(correct)/total
                self.model.best_epoch = epoch
                self.val_loss_best_acc = epoch_loss / val_size
                self.train_loss_best_acc = train_loss
                self.train_acc_best_acc = train_accuracy

                torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, osp.join(save_dir, f"model_{epoch}.pt"))  
            
            print("BEST_VAL_ACC: {} in epoch {}; Train_loss: {}; Val_loss: {}; Train_acc: {}".format(self.model.best_acc, \
            self.model.best_epoch,  self.train_loss_best_acc, self.val_loss_best_acc,self.train_acc_best_acc))
        
        return self.model.best_epoch, self.train_loss_best_acc, self.train_acc_best_acc
    
class Finetuner:
    def __init__(self, N, M, num_classes, dataset, margin, d_model):
        
        self.model = CMTNet(N = N, d_model=d_model, d_ff=d_model, num_classes = num_classes, M=M)
        
        # loss weight for class imbalance 
        if dataset == "cholec80":
            loss_weight = [
            1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,
        ]       
        elif dataset == "m2cai":
            loss_weight = [0.2639588012080849,   
            0.14090451483001626,
            1.0,
            0.38530937814605437,
            0.8951057074266243,
            0.1306822581894215,
            0.4749477270967242,
            0.3697049485015101]
            
        loss_weight  = torch.from_numpy(np.asarray(loss_weight)).float().cuda()
         
        self.ce = nn.CrossEntropyLoss(weight = loss_weight) 
        self.refinement_loss = refinement_loss(margin)
        
        self.num_classes = num_classes
        

    def finetune_test(self,gt_path,features_path, save_dir, model_dir, num_epochs, dataset, n_epoch, loss_fit, N, M, seed):
        # dataset
        train_dataloader, _, train_size, _ = create_dataloader(gt_dir= gt_path, feat_dir=features_path, dataset=dataset, phase="finetune") 

        # model       
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        
        
        self.model.cuda()
        start_point = torch.load(osp.join(save_dir, f"model_{n_epoch}.pt"))
        self.model.load_state_dict(start_point['model_state_dict'])
        
        # optimiser
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer.load_state_dict(start_point['optimizer_state_dict'])
        
        # start fine-tuning
        for epoch in range(1,num_epochs+1):
            epoch_loss = 0
            correct = 0
            total = 0
            
            self.model.train()
            for batch_input, batch_target in train_dataloader:

                with torch.set_grad_enabled(True):
                    batch_input, batch_target = batch_input.cuda(), batch_target.cuda()
                    
                    optimizer.zero_grad()

                    batch_input = torch.transpose(batch_input,1,2).contiguous()
                    _,T,_ = batch_input.size()
                    predictions = self.model(batch_input).squeeze(1)
                    
                    # compute loss 
                    loss = 0
                    
                    for p in predictions:
                        stage_cls_loss = self.ce(p.view(-1, self.num_classes), batch_target.view(-1)) #
                        loss += stage_cls_loss
                        
                    loss += self.refinement_loss(predictions,batch_target)[0] 

                    for i in range(M):
                        for j in range(N):
                            for v in range(2):
                                atten = self.model.stages[i].phaseencoder.layers[j].self_attn.attn.squeeze(0).squeeze(-2)[v]
                                
                                attention_loss = self.ce(atten, batch_target.view(-1)[::4]) #
                                loss += attention_loss*0.04

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.detach().item()

                _, predicted = torch.max(predictions[-1].data, 1)
                
                correct += ((predicted == batch_target).float().squeeze(1)).sum().detach().item()
                total += T 

           
            print(("TRAIN: [epoch %d]: epoch loss = %f,   acc = %f" % (epoch, epoch_loss / train_size,float(correct)/total)))

            average_loss = epoch_loss / train_size
            print(average_loss - loss_fit)
                    
            if average_loss - loss_fit<0:
                print("*"*100)
                
                torch.save(self.model.state_dict(), osp.join(model_dir,f"best_acc_{seed}.model"))
                
                if dataset == "cholec80":
                    dictionary = {'Preparation': 0, 'CalotTriangleDissection': 1, 'ClippingCutting': 2, 'GallbladderDissection': 3, 'GallbladderPackaging': 4, 'CleaningCoagulation': 5, 'GallbladderRetraction': 6}
                elif dataset == "m2cai":
                    dictionary = {'TrocarPlacement': 0, 'Preparation': 1, 'CalotTriangleDissection': 2,'ClippingCutting': 3, 'GallbladderDissection': 4, 'GallbladderPackaging': 5, 'CleaningCoagulation': 6, 'GallbladderRetraction': 7}

                acc,prec,rec,jac,acc_std,prec_std,rec_std,jac_std = self.evaluate(osp.join("best_models",dataset), osp.join("results",dataset), f"./data/{dataset}/resnet_features/",\
                "./data/{}/test.bundle".format(dataset), dictionary, 1, dataset, seed)

                return acc,prec,rec,jac
    
    def predict(self, model_dir, results_dir, features_path, vid_list_file, actions_dict, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.cuda()
            self.model.load_state_dict(torch.load(model_dir))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
    
                # get input features 
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = torch.transpose(input_x,1,2).contiguous()
                input_x = input_x.cuda()

                # get output 
                predictions = self.model(input_x)
                predictions = torch.transpose(predictions,2,3)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()


    def evaluate(self, all_model_dir, results_dir, features_path, vid_list_file, actions_dict, sample_rate,dataset, seed):
        model_dir = osp.join(all_model_dir, f"best_acc_{seed}.model")
        print(model_dir)
        self.predict(model_dir, results_dir, features_path, vid_list_file, actions_dict, sample_rate)
        acc,prec,rec,jac,acc_std,prec_std,rec_std,jac_std = official_evaluation(dataset)
        
        return acc,prec,rec,jac,acc_std,prec_std,rec_std,jac_std
