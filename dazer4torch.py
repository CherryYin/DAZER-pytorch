# --------------------------------------------------------
# DAZER for Torch
# Copyright (c) 2022 Ubisoft Chengdu Studio
# Licensed under The MIT License [see LICENSE for details]
# Written by Yin Juan
# It's the program for paper "A Deep Relevance Model for Zero-Shot Document Filtering"
# --------------------------------------------------------

from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import fasttext
import numpy as np
import io
from torch.utils.tensorboard import SummaryWriter

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
    return data

class DAZER(nn.Module):
    def __init__(self, device, 
                 vocab_dict, 
                 emb_path, 
                 vocab_num, 
                 phase="train"):
        super(DAZER, self).__init__()
        self.device = device
        self.emb_size = 300  # The dimonsion of latent variable
        self.seq_len = 100
        self.kernel_width= 5  #CNN Kernel width
        self.kernel_num = 50
        self.maxpooling_num = 3
        self.decoder_mlp1_num = 75
        self.decoder_mlp2_num = 1
        self.train_class_num = 20
        self.regular_term = 0.01
        self.adv_term = 0.1
        self.BCE = torch.nn.BCELoss()
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.KLD = torch.nn.KLDivLoss(reduction='mean')
        self.SL1 = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
        
        """Embedding layer"""
        if phase == "train":
            print("load embedding vectors...")
            org_emb_model = load_vectors(emb_path)
            emb_weight = np.random.random((vocab_num, self.emb_size))
            for w, idx in vocab_dict.items():
                if w in org_emb_model:
                    emb_weight[idx] = org_emb_model[w]
            emb_weight = torch.FloatTensor(emb_weight)
            self.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=False)
            del emb_weight, org_emb_model
        else:
            self.embedding = nn.Embedding(vocab_num+2, emb_size)

        """CONV (w1, b1)"""
        self.covn = nn.Conv2d(1, self.kernel_num, kernel_size=(self.kernel_width, self.emb_size*3), stride=1, bias=True)
        # self.neg_covn = nn.Conv2d(1, self.kernel_num, kernel_size=(self.kernel_width, self.emb_size*3), stride=1, bias=True)
        
        """Category-specific Gating Mechanism (w2, b2)"""
        self.query_gate = nn.Sequential(nn.Linear(self.emb_size, self.kernel_num), 
                           nn.Sigmoid())
        
        """hidden feature linear (w3, b3)"""
        self.hidden = nn.Sequential(nn.Linear(self.kernel_num*self.maxpooling_num, self.decoder_mlp1_num),   # set input and output shape
                                        nn.Tanh())
        """self.neg_hidden = nn.Sequential(nn.Linear(self.kernel_num*self.maxpooling_num, self.decoder_mlp1_num), 
                                        nn.Tanh())"""
        
        """relevancy score (w, w)"""
        self.score = nn.Sequential(nn.Linear(self.decoder_mlp1_num, self.decoder_mlp2_num),   # set input and output shape
                                     nn.Tanh())
        """self.neg_score = nn.Sequential(nn.Linear(self.decoder_mlp1_num, self.decoder_mlp2_num),   # set input and output shape
                                     nn.Tanh())"""
        
        """Adversarial Learning (w4, b4)"""
        self.adv = nn.Sequential(nn.Linear(self.decoder_mlp1_num, self.train_class_num),
                                 nn.Softmax())
        
    def kmax_pooling(self, x, dim, k):     
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)
    
    def get_class_vec_and_query(self, input_q):
        # get class vector
        class_vec = self.embedding(input_q)
        class_vec = torch.mean(class_vec, axis=1)
        #qurey gate
        query_gate = self.query_gate(class_vec)
        
        return class_vec, query_gate
    
    def gen_adv_query_mask(self, q_ids):
        q_mask = np.zeros((len(q_ids), self.train_class_num))
        for batch_num, c_index in enumerate(q_ids):
            q_mask[batch_num][c_index] = 1
        return q_mask
    
    def forward_base(self, class_vec, query_gate, input_d):
        batch_size = input_d.size()[0]
        
        class_vec = class_vec.unsqueeze(1)
        
        # Interaction embdding
        emb = self.embedding(input_d)    
        mult_info = class_vec * emb
        sub_info = class_vec - emb
        conv_input = torch.cat([emb,mult_info,sub_info], axis=-1)
        conv_input = conv_input.unsqueeze(1)
        
        # CNN
        conv = self.covn(conv_input)
        conv = conv.squeeze()
        
        # Gate
        query_gate = query_gate.unsqueeze(-1)
        gated_conv = query_gate * conv
        gated_conv = gated_conv.permute(0, 2, 1)
        
        # k-max pooling
        index = gated_conv.topk(self.maxpooling_num, dim=1)[1].sort(dim=1)[0]
        kmaxpooling = gated_conv.gather(1, index)
        
        # decoder
        kmaxpooling = torch.cat([kmaxpooling[:,k,:] for k in range(self.maxpooling_num)], axis=-1)
        decoder_mlp1 = self.hidden(kmaxpooling)
        score = self.score(decoder_mlp1)
        
        return decoder_mlp1, score
    
    def pred_process(self, input_q, input_pos_d):
        class_vec, query_gate = self.get_class_vec_and_query(input_q)
        pos_decoder_mlp1, pos_score = self.forward_base(class_vec, query_gate, input_pos_d)
        
        return pos_decoder_mlp1, pos_score
    
    
    def train_process(self, input_q, input_pos_d, input_neg_d, input_q_index):
        class_vec, query_gate = self.get_class_vec_and_query(input_q)
        pos_decoder_mlp1, pos_score = self.forward_base(class_vec, query_gate, input_pos_d)
        neg_decoder_mlp1, neg_score = self.forward_base(class_vec, query_gate, input_neg_d)
        
        hinge_loss =  torch.mean(torch.maximum(torch.FloatTensor([0.0]).to(self.device), (1.0 - pos_score + neg_score)))
        adv_prob = self.adv(pos_decoder_mlp1)
        log_adv_prob = torch.log(adv_prob)
        adv_loss = torch.mean(torch.sum(log_adv_prob * input_q_index.float(), 1))
        loss1 = -1.0 * adv_loss
        loss2 = hinge_loss + (adv_loss * self.adv_term)
        
        return pos_score, loss1, loss2, neg_score
    
def get_model(vocab_dict, emb_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DAZER(device=device, vocab_dict=vocab_dict, emb_path=emb_path, vocab_num=len(vocab_dict)).to(device)
    torch.backends.cudnn.enabled = False
    return model



def get_adv_optimizer(param_optimizer, dr, lr=1e-3):
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if "adv" in n], 'weight_decay':  dr}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr)
    return optimizer

def get_model_optimizer(param_optimizer, dr, lr=1e-3):
    optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if "adv" not in n], 'weight_decay':  dr}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr)
    return optimizer

def train_epoch(epoch, data_loader, model, optimizers,  writer, global_step, logger):
    print("#"*20)
    print("epoch: {}".format(epoch))
    model.cuda()
    model.train()
    device = model.device
    print(device)
    for optimizer in optimizers:
        optimizer.zero_grad()

    batch_loss = 0
    print_score1, print_score2, print_loss = 0.0, 0.0, 0.0
    label_ids = []
    for i, (q, d, d_aux, q_index, y) in enumerate(data_loader):
        global_step += 1
        q = torch.LongTensor(q).to(device)
        d = torch.LongTensor(d).to(device)
        d_aux = torch.LongTensor(d_aux).to(device)
        query_mask = torch.DoubleTensor(model.gen_adv_query_mask(q_index)).to(device)
        label_ids = torch.DoubleTensor(y).to(device)
        pos_score, loss1, loss2, neg_score = model.train_process(q, d, d_aux, query_mask)
        writer.add_scalar('log/loss/train_loss1', loss1, global_step)
        writer.add_scalar('log/loss/train_loss2', loss2, global_step)
        print_score1 += torch.mean(pos_score).item()
        print_score2 += torch.mean(neg_score).item()
        print_loss += loss2.item()
        
        # optimization
        """for name, param in model.named_parameters():
            if "adv" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False"""
        for param in model.parameters():
            param.requires_grad = False
        for param in model.adv.parameters():
            param.requires_grad = True   
        loss1.backward(retain_graph=True)
        for param in model.parameters():
            param.requires_grad = True
        for param in model.adv.parameters():
            param.requires_grad = False
        loss2.backward()
        optimizers[0].step()
        optimizers[0].zero_grad()
        
        optimizers[1].step()
        optimizers[1].zero_grad()      
        
        if (i+1) % 300 == 0:
            logger.info("Step {} : training pos_score {}, neg_score {}, model loss {}".format(global_step, print_score1/300, print_score2/300, print_loss/300))
            print_score1, print_score2, print_loss = 0.0, 0.0, 0.0
            
    return global_step


def evaluate(data_loader, model, device, writer, epoch, relevancy):
    BEC = torch.nn.BCEWithLogitsLoss()
    total_loss, total_pred_t, total_pred_f = 0.0, 0, 0
    num = 0
    total_acc = 0
    TP, FP, NF = 0, 0, 0
    for i, (q, d, y) in enumerate(data_loader):
        instance_num = len(q)
        q = torch.LongTensor(q).to(device)
        d = torch.LongTensor(d).to(device)
        # label_ids = torch.LongTensor(y).float().to(device)
        pos_decoder_mlp1, pos_score = model.pred_process(q, d)
        # preds = F.sigmoid(pos_score)
        hinge_loss =  torch.mean(torch.maximum(torch.FloatTensor([0.0]).to(device), (1.0 - pos_score)))
        inst_preds = torch.gt(pos_score, torch.tensor([[0.5] for i in range(pos_score.size(0))]).to(device)).long()
        
        
        acc = torch.sum((inst_preds[:, 0]==relevancy).int())
        acc = acc /50.0
        pred_t = torch.sum((inst_preds[:, 0]==1).int())
        pred_f = instance_num - pred_t
        
        total_loss += hinge_loss.item()
        total_acc += acc.item()
        total_pred_t += pred_t.item()
        total_pred_f += pred_f.item()
        num += 1 
        
    loss_avg = total_loss / num
    acc_avg = total_acc / num

    writer.add_scalar('log/loss/test', loss_avg, epoch)
    writer.add_scalar('log/loss/accurracy', acc_avg, epoch)
    
    return loss_avg, acc_avg, total_pred_t, total_pred_f