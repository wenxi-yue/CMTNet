import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class refinement_loss(nn.Module):
    def __init__(self,margin):
        super(refinement_loss, self).__init__()  
        self.ranking_loss_conf = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_smth = nn.MarginRankingLoss(margin=margin)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, prediction,groundtruth):
        
        loss = 0
        prob = list()
        stab = list()

        T = groundtruth.size()[1]
        target = torch.FloatTensor([-1 for m in range(T)]).cuda()

        target_sta = torch.FloatTensor([1]).cuda()

        for pred in prediction:
            pred = pred.squeeze()
            index_correct_cls = [(count,i.item()) for count,i in enumerate(groundtruth[0])]

            prob.append(torch.cat([torch.nn.functional.softmax(pred,dim=-1)[index].unsqueeze(0) for index in index_correct_cls]))
            
            pred = pred.unsqueeze(0)
            pred = torch.transpose(pred,1,2)
            mse_error = torch.mean(torch.clamp(self.mse(F.log_softmax(pred[:, :, 1:], dim=1), F.log_softmax(pred.detach()[:, :, :-1], dim=1)), min=0, max=9))
            
            stab.append(mse_error.unsqueeze(0))
        
        for i in range(len(prob)-1):

           
            loss += self.ranking_loss_conf(prob[i],prob[i+1],target)
            loss += self.ranking_loss_smth(stab[i],stab[i+1],target_sta)

        return loss,prob
