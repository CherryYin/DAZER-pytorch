import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class DAZER_Training_Dataset(Dataset):
    def __init__(self, datas, vocab_size, max_q_len, max_d_len, label_index_path):
        self.vocab_size = vocab_size
        self.max_q_len = max_q_len
        self.l_q, self.l_d, self.l_d_aux, self.l_q_index, self.l_y = [], [], [], [], []
        with open(label_index_path, "rb") as f:
            label_ids_dict, reverse_label_dict, label_index_dict = pickle.load(f)
            
        for line in datas:
            cols = line.strip().split('\t')
            y = float(1.0)
            q_str = cols[0]
            q = np.array([int(t) for t in cols[0].split(',') if int(t) < self.vocab_size])
            t1 = np.array([int(t) for t in cols[1].split(',') if int(t) < self.vocab_size])
            t2 = np.array([int(t) for t in cols[2].split(',') if int(t) < self.vocab_size])
            q_index = reverse_label_dict[q_str]
            q_index = label_index_dict[q_index]

            #padding
            v_q = np.array([self.vocab_size-1 for i in range(self.max_q_len)])
            v_d = np.array([self.vocab_size-1 for i in range(max_d_len)])
            v_d_aux = np.array([self.vocab_size-1 for i in range(max_d_len)])

            v_q[:min(q.shape[0], self.max_q_len)] = q[:min(q.shape[0], self.max_q_len)]
            v_d[:min(t1.shape[0], max_d_len)] = t1[:min(t1.shape[0], max_d_len)]
            v_d_aux[:min(t2.shape[0], max_d_len)] = t2[:min(t2.shape[0], max_d_len)]

            self.l_q.append(v_q)
            self.l_d.append(v_d)
            self.l_d_aux.append(v_d_aux)
            self.l_q_index.append(q_index)
            self.l_y.append(y)

            
    def __getitem__(self, index):
        return self.l_q[index], self.l_d[index], self.l_d_aux[index], self.l_q_str[index], self.y[index]
                  

    def __len__(self):
        return len(self.l_d)
    
    
class DAZER_Test_Dataset(Dataset):
    def __init__(self, datas, vocab_size, max_q_len, max_d_len, label_index_path, relevency):
        self.vocab_size = vocab_size
        self.max_q_len = max_q_len
        self.l_q, self.l_d, self.l_y = [], [], []
        with open(label_index_path, "rb") as f:
            label_ids_dict, reverse_label_dict, label_index_dict = pickle.load(f)
            
        for line in datas:
            cols = line.strip().split('\t')
            y = float(relevent)
            q_str = cols[0]
            q = np.array([int(t) for t in cols[0].split(',') if int(t) < self.vocab_size])
            t1 = np.array([int(t) for t in cols[1].split(',') if int(t) < self.vocab_size])

            #padding
            v_q = np.array([self.vocab_size-1 for i in self.max_q_len])
            v_d = np.array([self.vocab_size-1 for i in self.max_q_len])
            v_q[:min(q.shape[0], self.max_q_len)] = q[:min(q.shape[0], self.max_q_len)]
            v_d[:min(t1.shape[0], self.max_d_len)] = t1[:min(t1.shape[0], self.max_d_len)]

            self.l_q.append(v_q)
            self.l_d.append(v_d)
            self.l_y.append(y)

            
    def __getitem__(self, index):
        return self.l_q[index], self.l_d[index], self.y[index]
                  

    def __len__(self):
        return len(self.l_d)