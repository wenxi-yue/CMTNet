import numpy as np
from skimage import measure


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def official_evaluation(dataset):

    ground_truth_path = "./data/{}/groundtruth/".format(dataset)
    recog_path = "./results/{}/".format(dataset)
    file_list = "./data/{}/test.bundle".format(dataset)

    list_of_videos = read_file(file_list).split('\n')[:-1]
    acc_mtx = []
    
    if dataset == "cholec80":
        jac_mtx = np.zeros((7,40))
        rec_mtx = np.zeros((7,40))
        prec_mtx = np.zeros((7,40))

    elif dataset == "m2cai":
        jac_mtx = np.zeros((8,14))
        rec_mtx = np.zeros((8,14))
        prec_mtx = np.zeros((8,14))
  

    for count,vid in enumerate(list_of_videos):

        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()

        if dataset == "cholec80":
            action_dic = {'Preparation':1,  'CalotTriangleDissection':2, 'ClippingCutting':3, 'GallbladderDissection':4, \
                        'GallbladderPackaging':5, 'CleaningCoagulation':6, 'GallbladderRetraction':7}
            
        elif dataset == "m2cai":
            action_dic = {'TrocarPlacement':1, 'Preparation':2,  'CalotTriangleDissection':3, 'ClippingCutting':4, 'GallbladderDissection':5, \
                        'GallbladderPackaging':6, 'CleaningCoagulation':7, 'GallbladderRetraction':8}
        
        predLabelID = np.array([action_dic[i] for i in recog_content])
        gtLabelID = np.array([action_dic[i] for i in gt_content])
        
        oriT = 10

        diff = predLabelID - gtLabelID
        
        updatedDiff = [0 for i in range(len(diff))]

        if dataset == "cholec80":
            for iPhase in range(1,8):
                gtConn = measure.label(gtLabelID==iPhase,connectivity=1)
                for iConn in range(1,max(gtConn)+1):
                    startIdx = np.where(gtConn==iConn)[0][0]
                    endIdx = np.where(gtConn==iConn)[0][-1]
                    curDiff = diff[startIdx:endIdx+1]

                    t = oriT
                    if t>len(curDiff):
                        t = len(curDiff)
                    
                    if iPhase == 4 or iPhase == 5:
                        curDiff = [0 if num<t and i == -1 else i for num,i in enumerate(curDiff)]
                        curDiff = [0 if num>(len(curDiff)-1-t) and i in [1,2] else i for num,i in enumerate(curDiff)]
                
                    if iPhase == 6 or iPhase == 7:
                        curDiff = [0 if num<t and i in [-1,-2] else i for num,i in enumerate(curDiff)]
                        curDiff = [0 if num>(len(curDiff)-1-t) and i in [1,2] else i for num,i in enumerate(curDiff)]
                    
                    else:
                        curDiff = [0 if num<t and i == -1 else i for num,i in enumerate(curDiff)]
                        curDiff = [0 if num>(len(curDiff)-1-t) and i==1 else i for num,i in enumerate(curDiff)]
                    
                    updatedDiff[startIdx:endIdx+1] = curDiff

        elif dataset == "m2cai":
            for iPhase in range(1,9):
                gtConn = measure.label(gtLabelID==iPhase,connectivity=1)
                for iConn in range(1,max(gtConn)+1):
                    startIdx = np.where(gtConn==iConn)[0][0]
                    endIdx = np.where(gtConn==iConn)[0][-1]
                    curDiff = diff[startIdx:endIdx+1]

                    t = oriT
                    if t>len(curDiff):
                        t = len(curDiff)
                    
                    if iPhase == 5 or iPhase == 6:
                        curDiff = [0 if num<t and i == -1 else i for num,i in enumerate(curDiff)]
                        curDiff = [0 if num>(len(curDiff)-1-t) and i in [1,2] else i for num,i in enumerate(curDiff)]
                
                    if iPhase == 7 or iPhase == 8:
                        curDiff = [0 if num<t and i in [-1,-2] else i for num,i in enumerate(curDiff)]
                        curDiff = [0 if num>(len(curDiff)-1-t) and i in [1,2] else i for num,i in enumerate(curDiff)]
                    
                    else:
                        curDiff = [0 if num<t and i == -1 else i for num,i in enumerate(curDiff)]
                        curDiff = [0 if num>(len(curDiff)-1-t) and i==1 else i for num,i in enumerate(curDiff)]
                    
                    updatedDiff[startIdx:endIdx+1] = curDiff
            
        jac = []
        prec = []
        rec = []

        if dataset == "cholec80":
            for iPhase in range(1,8):
                gtConn = measure.label(gtLabelID==iPhase,connectivity=1)
                predConn = measure.label(predLabelID==iPhase,connectivity=1)
                

                if max(gtConn) == 0:
                    jac.append(float("nan"))
                    prec.append(float("nan"))
                    rec.append(float("nan"))
                    continue

                set1 = {i for i in np.where(gtConn!=0)[0]}
                set2 = {i for i in np.where(predConn!=0)[0]}

                iPUnion = set.union(set1,set2)
                tp = sum([updatedDiff[i]==0 for i in iPUnion])
                jaccard = 100*tp/len(iPUnion)
                jac.append(jaccard)
                
                sumPred = sum([i == iPhase for i in predLabelID])
                sumGT = sum([i == iPhase for i in gtLabelID])
        
                if sumPred == 0:
                    prec.append(100)
                else:
                    prec.append(tp * 100 / sumPred)
                
                rec.append(tp * 100 / sumGT)
        
        elif dataset == "m2cai":
            for iPhase in range(1,9):
                gtConn = measure.label(gtLabelID==iPhase,connectivity=1)
                predConn = measure.label(predLabelID==iPhase,connectivity=1)
                

                if max(gtConn) == 0:
                    jac.append(float("nan"))
                    prec.append(float("nan"))
                    rec.append(float("nan"))
                    continue

                set1 = {i for i in np.where(gtConn!=0)[0]}
                set2 = {i for i in np.where(predConn!=0)[0]}

                iPUnion = set.union(set1,set2)
                tp = sum([updatedDiff[i]==0 for i in iPUnion])
                jaccard = 100*tp/len(iPUnion)
                jac.append(jaccard)
                
                sumPred = sum([i == iPhase for i in predLabelID])
                sumGT = sum([i == iPhase for i in gtLabelID])
        
                if sumPred == 0:
                    prec.append(100)
                else:
                    prec.append(tp * 100 / sumPred)
                
                rec.append(tp * 100 / sumGT)

        acc = 100*sum([i == 0 for i in updatedDiff])/len(gtLabelID)
        
        jac = [100 if i>100 else i for i in jac]
        prec = [100 if i>100 else i for i in prec]
        rec = [100 if i>100 else i for i in rec]

        acc_mtx.append(acc)
        jac_mtx[:,count] = jac
        prec_mtx[:,count] = prec
        rec_mtx[:,count] = rec


    jac_per_phase =  np.nanmean(jac_mtx, 1)
    prec_per_phase = np.nanmean(prec_mtx, 1)
    rec_per_phase = np.nanmean(rec_mtx, 1)

    acc = np.mean(acc_mtx)
    jac = np.mean(jac_per_phase)
    prec = np.mean(prec_per_phase)
    rec = np.mean(rec_per_phase)
    
    acc_std = np.std(acc_mtx)
    jac_std = np.std(jac_per_phase)
    prec_std = np.std(prec_per_phase)
    rec_std = np.std(rec_per_phase)

    print("Accuracy: %.13f +/- %.1f" % (acc,acc_std))
    print("Precision: %.13f +/- %.1f" % (prec,prec_std))
    print("Recall: %.13f +/- %.1f" % (rec,rec_std))
    print("Jaccard: %.13f  +/- %.1f" % (jac,jac_std))

    return acc,prec,rec,jac,acc_std,prec_std,rec_std,jac_std
