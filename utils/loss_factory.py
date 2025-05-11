import torch
import math 
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score, f1_score, auc, classification_report, precision_recall_fscore_support, confusion_matrix

def nll_loss(hazards, S, Y, c, alpha=0., eps=1e-7):
    batch_size = len(Y)
    #print(Y)
    Y = Y.view(batch_size, 1).long()  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    #print(S_padded)
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)   
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

def nllsurv(out, gt, alpha=0.):
    #loss = 0
    #for haz, s in zip(out['hazards'], out['S']):
    #    loss += nll_loss(haz, s, gt['label'], gt['c'], alpha=alpha)
    #print(out['hazards'], out['S'], gt['label'])
    loss = nll_loss(out['hazards'], out['S'], gt['label'], gt['c'], alpha=alpha)
    return loss
    

def CLSLoss1(out, gt, alpha=0.):
    loss = 0
    for pred in out['pred1']:
        loss += F.cross_entropy(pred, gt['label'].long())
    #print(out['pred'], lbl)
    return loss

def CLSLoss(out, gt, alpha=0.):
    #loss = 0
    #for pred in out['pred']:
    #print(out['pred'].shape, gt['label'].shape)
    #lbl = F.one_hot(gt['label'].long(), num_classes=5)
    loss = F.cross_entropy(out['cls'], gt['label'].long())
    #print(out['pred'], lbl)
    return loss
    
def BCLSLoss(out, gt, alpha=0.):
    #loss = 0
    #for pred in out['pred']:
    #print(out['cls'].shape, gt['label'].shape)
    #lbl = F.one_hot(gt['label'].long(), num_classes=5)
    loss = F.binary_cross_entropy(out['cls'], gt['label'].float()).mean()
    #print(out['pred'], lbl)
    return loss

def SIMLoss(out, gt, alpha=0.):
    lf, rf = out['align']
    loss = 1 - F.cosine_similarity(lf, rf).mean()
    #print(lf.shape, rf.shape, loss.shape)
    return loss

def ReembLoss(out, gt, alpha=0.):
    return out['reemb']
    
def contrastive(out, gt, temperature=0.1):
    feats, labels = out['contrastive']    # feats shape: [B, D]
    feats = torch.concat(feats, dim=0)
    labels = torch.tensor(labels).cuda()
    #labels = outputs['labels']    # labels shape: [B]
    #print(feats.shape, labels.shape)

    feats = F.normalize(feats, dim=-1, p=2)

    logits_mask = torch.eye(feats.shape[0]).float().cuda()
    mask = torch.eq(labels.view(-1, 1), labels.contiguous().view(1, -1)).float() - logits_mask

    # compute logits
    logits = torch.matmul(feats, feats.T) / temperature
    logits = logits - logits_mask * 1e9

    # optional: minus the largest logit to stablize logits
    #logits = stablize_logits(logits)
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()

    # compute ground-truth distribution
    p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
    #loss = compute_cross_entropy(p, logits)
    logits = F.log_softmax(logits, dim=-1)
    loss = torch.sum(p * logits, dim=-1)

    return -loss.mean()

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def topk_cluster(feature,supports,scores,p,k=3):
    #p: outputs of model batch x num_class
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    topk_sim_matrix,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near].detach().clone()  #batch x K x num_class
    #print(p.shape, scores_near.shape)
    diff_scores = torch.sum((p.unsqueeze(1) - scores_near)**2,-1)
    
    loss = -1.0* topk_sim_matrix * diff_scores
    return loss.mean()
    
def select_supports(ent_s, y_hat, filter_K=100, n_cls=5):
    #ent_s = self.ent
    y_hat = y_hat.argmax(dim=1).long()
    #filter_K = self.filter_K
    if filter_K == -1:
        indices = torch.LongTensor(list(range(len(ent_s))))

    indices = []
    indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
    for i in range(n_cls):
        #print(y_hat.shape, ent_s.shape)
        _, indices2 = torch.sort(ent_s[y_hat == i])
        indices.append(indices1[y_hat==i][indices2][:filter_K])
    indices = torch.cat(indices)

    return indices
    #self.supports = self.supports[indices]
    #self.labels = self.labels[indices]
    #self.ent = self.ent[indices]
    #self.scores = self.scores[indices]
    
    #return self.supports, self.labels

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div        
        

def prototype_loss(z,p,labels=None,use_hard=False,tau=1):
    #z [batch_size,feature_dim]
    #p [num_class,feature_dim]
    #labels [batch_size,]        
    z = F.normalize(z,1)
    p = F.normalize(p,1)
    dist = z @ p.T / tau
    if labels is None:
        _,labels = dist.max(1)
    if use_hard:
        """use hard label for supervision """
        #_,labels = dist.max(1)  #for prototype-based pseudo-label
        labels = labels.argmax(1)  #for logits-based pseudo-label
        loss =  F.cross_entropy(dist,labels)
    else:
        """use soft label for supervision """
        #print(labels.shape, dist.shape)
        loss = softmax_kl_loss(labels.detach(),dist).sum(1).mean(0)  #detach is **necessary**
        #loss = softmax_kl_loss(dist,labels.detach()).sum(1).mean(0) achieves comparable results
    return dist,loss

def attloss(out, gt, alpha=0.):
    #print(out['att'].shape)
    loss = (out['att'].softmax(1) * out['att'].log_softmax(1)).sum(1).mean()
    return loss
    
def tentloss(out, gt, alpha=0.):
    #print(out['pred'].shape)
    loss = softmax_entropy(out['pred']).mean()
    return loss
    
    
def pseloss(out, gt, alpha=0.):
    
    loss = F.binary_cross_entropy(out['cls'], out['pse']).mean()
    #loss = softmax_entropy(out['pred']).mean()
    return loss
    
feat_bank = {}
logit_bank = {}
for label in range(5):
    feat_fold = 'data/mem_bank/fold4/feature/{}'.format(label)
    logit_fold = 'data/mem_bank/fold4/logit/{}'.format(label)
    feat_bank[label] = []
    logit_bank[label] = []
    
    pt_files = os.listdir(feat_fold)
    for pt_file in pt_files:
        #print(feat_fold, pt_file)
        feat = torch.load(os.path.join(feat_fold, pt_file), weights_only=True).cuda()
        logit = torch.load(os.path.join(logit_fold, pt_file), weights_only=True).cuda()
        feat_bank[label].append(feat)
        logit_bank[label].append(logit)
        
    
def tsdloss(out, gt, n_cls=5):
    loss = 0
    z = out['feature']
    p = out['pred']

    yhat = F.one_hot(p.argmax(1), num_classes=n_cls).float()
    yent = softmax_entropy(p)
    yscore = F.softmax(p,1)

    with torch.no_grad():
        supports = torch.zeros((0, 256)).cuda()
        labels = torch.zeros((0, n_cls)).cuda()
        ents = torch.zeros((0)).cuda()
        scores = torch.zeros((0)).cuda()
        for label in range(5):
            feats = torch.cat(feat_bank[label], dim=0)
            logits = torch.cat(logit_bank[label], dim=0)
            
            label = F.one_hot(logits.argmax(1), num_classes=n_cls).float()
            ent = softmax_entropy(logits)
            score = F.softmax(logits,1)
            #print(prob)
            #att = 1.5 + torch.sum(prob * torch.log(prob + 1e-10), dim=-1, keepdims=True)
            
            supports = torch.cat([supports, feats], dim=0)
            labels = torch.cat([labels, label], dim=0)
            ents = torch.cat([ents, ent], dim=0)
            scores = torch.cat([scores, logits], dim=0)
        
        supports = torch.cat([supports, z], dim=0)
        labels = torch.cat([labels, yhat], dim=0)
        ents = torch.cat([ents, yent], dim=0)
        scores = torch.cat([scores, yscore], dim=0)
        
        indices = select_supports(ents, labels, filter_K=100, n_cls=n_cls)
        
        supports = supports[indices]
        labels = labels[indices]
        ents = ents[indices]
        scores = scores[indices]
        
        supports = F.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
    dist,loss = prototype_loss(z,weights.T,yscore,use_hard=False)
    loss += topk_cluster(z.detach().clone(),supports,scores,p,k=3)
    #loss += F.cross_entropy(out['pred'], yhat).mean() * 2
    return loss
    

def ttaloss(out, gt, n_cls=5):
    loss = 0
    z = out['feature']
    p = out['pred']

    yhat = F.one_hot(p.argmax(1), num_classes=n_cls).float()
    yent = softmax_entropy(p)

    with torch.no_grad():
        supports = torch.zeros((0, 256)).cuda()
        labels = torch.zeros((0, n_cls)).cuda()
        ents = torch.zeros((0)).cuda()
        for label in range(5):
            feats = torch.cat(feat_bank[label], dim=0)
            logits = torch.cat(logit_bank[label], dim=0)
            
            label = F.one_hot(logits.argmax(1), num_classes=n_cls).float()
            ent = softmax_entropy(logits)
            #print(prob)
            #att = 1.5 + torch.sum(prob * torch.log(prob + 1e-10), dim=-1, keepdims=True)
            
            supports = torch.cat([supports, feats], dim=0)
            labels = torch.cat([labels, label], dim=0)
            ents = torch.cat([ents, ent], dim=0)
        
        supports = torch.cat([supports, z], dim=0)
        labels = torch.cat([labels, yhat], dim=0)
        ents = torch.cat([ents, yent], dim=0)
        
        indices = select_supports(ents, labels, filter_K=100, n_cls=n_cls)
        
        supports = supports[indices]
        labels = labels[indices]
        ents = ents[indices]
        
        supports = F.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
    loss += (z @ F.normalize(weights, dim=0)).mean()
    return loss

class Loss_factory(nn.Module):
    def __init__(self, args):
        super(Loss_factory, self).__init__()
        loss_item = args.loss.split(',')
        self.loss_collection = {}
        for loss_im in loss_item:
            tags = loss_im.split('_')
            self.loss_collection[tags[0]] = float(tags[1]) if len(tags) == 2 else 1.
            
        
    def forward(self, preds, target):
        loss_sum = 0
        ldict = {}
        for loss_name, weight in self.loss_collection.items():
            loss = eval(loss_name + '(preds, target) * weight')
            ldict[loss_name] = loss
            loss_sum += loss
        return loss_sum, ldict

def metric(results, task, testset='zfy'):
    if task == 'cls':
        #out['cls'], pred_cls, label
        all_cls = []
        all_lbls = []
        all_scores = []
        for res in results:
            if testset == 'tcga':
                #print(res[0].shape)
                res[0][0, 1] = res[0][0, 1] + res[0][0, 2]
                res[0] = np.concatenate([res[0][:, :2], res[0][:, 3:]], axis=-1)
                if res[1] > 1:
                    res[1] = res[1] - 1
                if res[2] > 1:
                    res[2] = res[2] - 1
                    
            all_scores.append(res[0])
            all_cls.append(res[1])
            all_lbls.append(res[2])
        
        if testset == 'tcga':
            subtype_list = ['LUMA', 'LUMB', 'HER2', 'TNBC']
        else:
            subtype_list = ['LUMA', 'LUMB1', 'LUMB2', 'HER2', 'TNBC']
        
        return cls_metric(np.concatenate(all_cls, axis=0), np.concatenate(all_lbls, axis=0), np.concatenate(all_scores, axis=0), subtype_list)
    elif task == 'bcls':
        all_cls = []
        all_lbls = []
        all_scores = []
        for res in results:
            all_scores.append(res[0])
            all_cls.append(res[1])
            all_lbls.append(res[2])
        
        subtype_list = ['Others', 'TNBC']
        
        return cls_metric(np.concatenate(all_cls, axis=0), np.concatenate(all_lbls, axis=0), np.concatenate(all_scores, axis=0), subtype_list)
    else:
        #risk, status, event_time
        all_risk = []
        all_status = []
        all_time = []
        for res in results:
            all_risk.append(res[0])
            all_status.append(res[1])
            all_time.append(res[2])
        cindex = concordance_index_censored((1-np.concatenate(all_status, axis=0)).astype(bool), np.concatenate(all_time, axis=0), np.concatenate(all_risk, axis=0), tied_tol=1e-08)[0]
        return {'cindex': cindex}
        

#res = metric(all_preds, all_lbls, all_scores)
def cls_metric(pred, label, score, subtype_list):
    #print(pred.shape, label.shape, score.shape)
    acc = np.sum(pred == label) / len(label)
    #auc_score = roc_auc_score(label, score, average='macro', multi_class='ovr')
    auc_score = roc_auc_score(label, score, average=None, multi_class='ovr')
    #fp, tp, thresholds = roc_auc_score(label, score, multi_class='ovr')
    #roc_auc = auc(fp, tp)
    #f1score = f1_score(label, pred, average='weighted')
    
    #classification_report(y_true, y_pred, target_names=subtype_list)
    #print(classification_report(label, pred, target_names=subtype_list))
    
    n_bootstraps = 1000

    # Bootstrap to calculate the confidence interval
    auc_scores = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(range(len(label)), len(label), replace=True)
        auc_bootstrap = roc_auc_score(label[indices], score[indices])
        auc_scores.append(auc_bootstrap)

    # Calculate the 95% confidence interval
    confidence_interval = np.percentile(auc_scores, [2.5, 97.5])

    #print("AUC:", auc)
    print("95% Confidence Interval:", confidence_interval)
    
    tp, fn, fp, tn = confusion_matrix(label, pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    sen = (tp)/(tp+fn)
    sp = (tn)/(tn+fp)
    ppv = (tp)/(tp+fp)
    npv = (tn)/(tn+fn)
    f1 = 2*(sen*ppv)/(sen+ppv)
    fpr = (fp)/(fp+tn)
    tpr = (tp)/(tp+fn)
    res = { 'ACC': round(acc, 4),
            'AUC': round(auc_score, 4),
            'Sens': round(sen, 4),
            'Spec': round(sp, 4),
            'PPV/Precision': round(ppv, 4),
            'NPV': round(npv, 4),
            'F1score': round(f1, 4),
            'False positive rate': round(fpr, 4),
            'True positive rate': round(tpr, 4),
            #"95% Confidence Interval": np.percentile(auc_scores, [2.5, 97.5]),
          }
    return res
    '''
    res = []
    for l in range(len(subtype_list)):
        prec,recall,f1score,support = precision_recall_fscore_support(np.array(label)==l,
                                                          np.array(pred)==l,
                                                          pos_label=True,
                                                          average=None)
        #print(prec,recall,f1score,support)
        res.append([subtype_list[l],recall[0],recall[1]])
        # ['class','specificity','sensitivity']
    #print(res)
    return {'ACC': acc, 'AUC': auc_score, 'F1score': f1score[1], 'Spec': res[-1][1], 'Sens': res[-1][2]}
    '''