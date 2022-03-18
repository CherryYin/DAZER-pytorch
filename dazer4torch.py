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
        self.adv_term = 0.2
        self.BCE = torch.nn.BCELoss()
        self.MSE = torch.nn.MSELoss(reduction='mean')
        self.KLD = torch.nn.KLDivLoss(reduction='mean')
        self.SL1 = torch.nn.SmoothL1Loss(reduce=False, size_average=False)
        
        """Embedding layer"""
        if phase == "train":
            print("load embedding vectors...")
            org_emb_model = load_vectors(emb_path)
            emb_weight = np.random.random((vocab_num+2, self.emb_size))
            for w, idx in vocab_dict.items():
                if w in org_emb_model:
                    emb_weight[idx] = org_emb_model[w]
            emb_weight = torch.FloatTensor(emb_weight)
            self.embedding = nn.Embedding.from_pretrained(emb_weight, freeze=False)
            del emb_weight, org_emb_model
        else:
            self.embedding = nn.Embedding(vocab_num+2, emb_size)

        """CONV (w1, b1)"""
        self.pos_covn = nn.Conv2d(1, self.kernel_num, kernel_size=(self.kernel_width, self.emb_size*3), stride=1, bias=True)
        self.neg_covn = nn.Conv2d(1, self.kernel_num, kernel_size=(self.kernel_width, self.emb_size*3), stride=1, bias=True)
        
        """Category-specific Gating Mechanism (w2, b2)"""
        self.query_gate = nn.Sequential(nn.Linear(self.emb_size, self.kernel_num), 
                           nn.Sigmoid())
        
        """hidden feature linear (w3, b3)"""
        self.pos_hidden = nn.Sequential(nn.Linear(self.kernel_num*self.maxpooling_num, self.decoder_mlp1_num),   # set input and output shape
                                        nn.Tanh())
        self.neg_hidden = nn.Sequential(nn.Linear(self.kernel_num*self.maxpooling_num, self.decoder_mlp1_num), 
                                        nn.Tanh())
        
        """relevancy score (w, w)"""
        self.pos_score = nn.Sequential(nn.Linear(self.decoder_mlp1_num, self.decoder_mlp2_num),   # set input and output shape
                                     nn.Tanh())
        self.neg_score = nn.Sequential(nn.Linear(self.decoder_mlp1_num, self.decoder_mlp2_num),   # set input and output shape
                                     nn.Tanh())
        
        """Adversarial Learning (w4, b4)"""
        self.adv = nn.Sequential(nn.Linear(self.decoder_mlp1_num, self.train_class_num),
                                 nn.Softmax())
        
    def kmax_pooling(self, x, dim, k):     
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)
    
    def get_class_vec_and_query(self, input_q):
        # get class vector
        class_vec = self.embedding(input_q)
        class_vec = torch.mean(class_vec, axis=0)
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
        
        # Interaction embdding
        emb = self.embedding(input_d)
        mult_info = class_vec * emb
        sub_info = class_vec - emb
        conv_input = torch.concat([emb,mult_info,sub_info], axis=-1)
        conv_input = conv_input.unsqueeze(1)
        
        # CNN
        conv = self.covn(conv_input)
        conv = conv.squeeze()
        
        # Gate
        gated_conv = query_gate * conv
        gated_conv = gated_conv.permute(0, 2, 1)
        
        # k-max pooling
        index = gated_conv.topk(self.maxpooling_num, dim=2)[1].sort(dim=2)[0]
        kmaxpooling = gated_conv.gather(2, index)
        
        # decoder
        encoder = torch.reshape(kmaxpooling, (batch_size, -1))
        decoder_mlp1 = self.hidden(encoder)
        score = self.score(encoder)
        
        return decoder_mlp1, score
    
    def pred_process(self, input_q, input_pos_d):
        class_vec, query_gate = self.get_class_vec_and_query(input_q)
        pos_decoder_mlp1, pos_score = forward_base(class_vec, query_gate, input_pos_d)
        
        return pos_decoder_mlp1, pos_score
    
    
    def train_process(self, input_q, input_pos_d, input_neg_d, input_q_index):
        class_vec, query_gate = self.get_class_vec_and_query(input_q)
        pos_decoder_mlp1, pos_score = forward_base(class_vec, query_gate, input_pos_d)
        neg_decoder_mlp1, neg_score = forward_base(class_vec, query_gate, input_neg_d)
        
        hinge_loss =  torch.mean(torch.max(0.0, 1 - score_pos + score_neg))
        adv_prob = self.adv(pos_decoder_mlp1)
        log_adv_prob = torch.log(adv_prob)
        adv_loss = torch.mean(torch.sum(log_adv_prob * input_q_index.float()), axis=1, keep_dims=True)
        loss1 = -1 * adv_loss
        loss2 = hinge_loss + (adv_loss * self.adv_term)
        
        return pos_score, loss1, loss2
    
def get_model(vocab_dict, emb_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DAZER(device=device, vocab_dict=vocab_dict, emb_path=emb_path, vocab_num=len(vocab_dict)).to(device)
    torch.backends.cudnn.enabled = False
    return model



def get_optimizer(param_optimizer, decays, dr, lr=5e-5, phase="in"):
    if phase == "in":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in decays)], 'weight_decay':  dr},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in decays)], 'weight_decay': 0.0}]
    elif phase == "notin":
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in decays)], 'weight_decay':  0.0},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in decays)], 'weight_decay': dr}]
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr)
    
    return optimizer

def train_epoch(epoch, data_loader, model, optimizers,  writer, global_step):
    print("#"*20)
    print("epoch: {}".format(epoch))
    model.cuda()
    model.train()
    device = model.device
    print(device)
    for optimizer in optimizers:
        optimizer.zero_grad()

    batch_loss = 0
    print_loss1, print_loss2 = 0.0, 0.0
    label_ids = []
    for i, (q, d, d_aux, q_index, y) in enumerate(data_loader):
        global_step += 1
        q = torch.LongTensor(q).to(device)
        d = torch.LongTensor(d).to(device)
        d_aux = torch.LongTensor(d_aux).to(device)
        query_mask = torch.FloatTensor(model.get_adv_mask(q_index)).to(device)
        label_ids = torch.LongTensor(label).float().to(device)
        pos_score, loss1, loss2 = model.train_process(q, d, d_aux, query_mask)
        writer.add_scalar('log/loss/train', loss, global_step)
        print_loss1 += loss1.item()
        print_loss2 += loss2.item()
        
        # optimization
        for name, param in model.named_parameters():
            if "adv" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        loss1.backward()
        optimizers[0].step()
        optimizers[0].zero_grad()
        
        for name, param in model.named_parameters():
            if "adv" not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        loss2.backward()
        optimizers[1].step()
        optimizers[1].zero_grad()
        
        if (i+1) % 300 == 0:
            logging.info("Step {} : training loss is {}".format(global_step, print_loss/300))
            print_loss1, print_loss2 = 0.0, 0.0
            
    return global_step


def evaluate(test_loader, model, device, writer, epoch, relevancy):
    BEC = torch.nn.BCEWithLogitsLoss()
    total_loss, total_pred_t, total_pred_t = 0.0, 0, 0
    num = 0
    total_acc = 0
    TP, FP, NF = 0, 0, 0
    keywords = torch.LongTensor(keywords).to(device)
    for i, (q, d, d_aux, q_str, y) in enumerate(data_loader):
        global_step += 1
        q = torch.LongTensor(q).to(device)
        d = torch.LongTensor(d).to(device)
        d_aux = torch.LongTensor(d_aux).to(device)
        query_indexs = torch.FloatTensor(query_indexs).to(device)
        label_ids = torch.LongTensor(label).float().to(device)
        pos_score = model.predict(q, d, y)
        hinge_loss =  torch.mean(torch.max(0.0, 1 - score_pos))
        inst_preds = torch.gt(yi, torch.tensor([[0.5] for i in range(yi.size(0))]).to(device)).long()
        
        acc = torch.sum((inst_preds[:, 0]==relevancy).int()) / float(torch.sum(inst_preds[:, 0])) 
        pred_t = torch.sum((inst_preds[:, 0]==1).int())
        pred_f = torch.sum(inst_preds[:, 0]) - pred_t
        
        total_loss += hinge_loss.item()
        total_acc += acc.item()
        total_pred_t += pred_t.item()
        total_pred_f += pred_f.item()
        num += 1 
        
    loss_avg = total_loss / num
    acc_avg = total_acc / num

    writer.add_scalar('log/loss/test', loss, epoch)
    writer.add_scalar('log/loss/accurracy', acc_avg, epoch)
    
    return loss_avg, acc_avg, total_pred_t, total_pred_f



            


