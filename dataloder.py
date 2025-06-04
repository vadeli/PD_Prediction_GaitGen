import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from os.path import join as pjoin
from torch.nn.utils.rnn import pad_sequence
import torch
import joblib

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from consts import paths

class BaseClinicalMotionDataset(Dataset):
    def __init__(self, db, split='train'):
        self.db = db
        self.split = split
        self.joints_num = 22
        self.min_lengh = 30
        self.data_root = paths[db]['data_root']
        self.motion_dir = pjoin(self.data_root, paths[db]['data_folder'])
        self.rep_dir = None
        self.data, self.labels = self.load_data()
        feat_bias = 5
        self.meta_dir = pjoin('./data', db)
        os.makedirs(self.meta_dir, exist_ok=True)
        self.mean, self.std = self.get_mean_std(self.motion_dir, feat_bias)
        
    def mean_variance(self, data_dir, save_dir, joints_num):
        if data_dir is not None:
            file_list = os.listdir(data_dir)
            data_list = []
            for file in file_list:
                data = np.load(pjoin(data_dir, file))
                if np.isnan(data).any():
                    print(file)
                    continue
                data_list.append(data)
        else:
            data_list = self.data
        data = np.concatenate(data_list, axis=0)
        print(data.shape)
        Mean = data.mean(axis=0)
        Std = data.std(axis=0)
        Std[0:1] = Std[0:1].mean() / 1.0
        Std[1:3] = Std[1:3].mean() / 1.0
        Std[3:4] = Std[3:4].mean() / 1.0
        Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
        Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
        Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
        Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0
        assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]
        np.save(pjoin(save_dir, 'Mean.npy'), Mean)
        np.save(pjoin(save_dir, 'Std.npy'), Std)
    
    
    def get_mean_std(self, motion_dir, feat_bias):         
        try:
            mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
            std = np.load(pjoin(self.meta_dir, 'std.npy'))
        except:
            try:
                mean = np.load(pjoin(motion_dir, 'Mean.npy'))
                std = np.load(pjoin(motion_dir, 'Std.npy'))
            except:
                self.mean_variance(self.rep_dir, motion_dir, self.joints_num)
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (self.joints_num - 1) * 3] = std[4: 4 + (self.joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (self.joints_num - 1) * 3: 4 + (self.joints_num - 1) * 9] = std[4 + (self.joints_num - 1) * 3: 4 + (self.joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (self.joints_num - 1) * 9: 4 + (self.joints_num - 1) * 9 + self.joints_num * 3] = std[4 + (self.joints_num - 1) * 9: 4 + (self.joints_num - 1) * 9 + self.joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (self.joints_num - 1) * 9 + self.joints_num * 3:] = std[4 + (self.joints_num - 1) * 9 + self.joints_num * 3:] / feat_bias

            assert 4 + (self.joints_num - 1) * 9 + self.joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(self.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(self.meta_dir, 'std.npy'), std)
        return mean, std

    def load_data(self):
        raise NotImplementedError("Subclasses should implement this method")
    
    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        motion = self.data[idx]
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        m_length = motion.shape[0]
        
        if m_length < 200:
                motion = np.concatenate([motion,
                                         np.zeros((200 - m_length, motion.shape[1]))
                                         ], axis=0)
        elif m_length > 200:
            motion = motion[:200]
            m_length = 200
        if self.db in ['tri_pd', 'pdgam', 'gaitgen']:
            label = self.labels[idx]
            return motion, m_length, label
        if self.db == 'tri':
            name = self.sample_name[idx]
            return motion, m_length, name
        
    
class TRI(BaseClinicalMotionDataset):
    def load_data(self):
        print('info:', f"Loading {self.db} dataset...")
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        self.lengths = []
        self.sample_name = []
        data = []
        id_list = []
        id_list = os.listdir(self.rep_dir)
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name))
                if motion.shape[0] < self.min_lengh: #TODO: Check if this is true for us
                    continue
                self.lengths.append(motion.shape[0]) 
                self.sample_name.append(name)
                data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        print('info', f"Total number of motions {len(data)}") 
        return data, []
    
class TRI_PD(BaseClinicalMotionDataset):
    def load_data(self):
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        self.annot_file = pjoin(self.data_root, 'ALL_PD_SCORES_AND_CLINICAL_DATA.xlsx')
        annotations = pd.read_excel(self.annot_file)
        
        self.lengths = []
        data, labels = [], []
        id_list = os.listdir(self.rep_dir)
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name))
                if motion.shape[0] < self.min_lengh: 
                    continue
                self.lengths.append(motion.shape[0])
                data.append(motion)
                # read score labels
                walk = name.split('.npy')[0]
                score = annotations[annotations['File Number/Title '] == walk]['UPDRS__gait'].values[0]
                labels.append(score)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass
        
        print('info', f"Total number of motions {len(data)}") 
        return data, labels
    
class PDGaMDataset(BaseClinicalMotionDataset):
    def load_data(self):
        self.rep_dir = pjoin(self.motion_dir, 'new_joint_vecs')
        self.annot_file = pjoin(self.motion_dir, f'{self.split}.csv')
        annotations = pd.read_csv(self.annot_file, names=['walk', '#frames', 'score'])
        self.split_file = pjoin(self.motion_dir, f'{self.split}.txt')
        
        data, data_M, labels = [], [], []
        id_list = []
        with open(self.split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
                
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.rep_dir, name + '.npy'))
                data.append(motion)
                # read score labels
                if 'mixmatch' in name:
                    score = 3
                else:
                    subject_id = name.split('_')[0][:3]
                    visit_ID = name.split('_')[0][4:]
                    annot_format_ID = f'{visit_ID}_{subject_id}'
                    score = annotations[annotations['walk'] == annot_format_ID]['score'].values[0]
                labels.append(score)
            except Exception as e:
                print(e)
                pass
        
        print('info', f"Total number of motions {len(data)}, snippets {-1}")
        return data, labels
    
class GAITGenDataset(BaseClinicalMotionDataset):
    def load_data(self):
        self.motion_dir = paths['gaitgen']['data_root']
        
        print(f'----> Using file {paths["gaitgen"]["file_name"]} for gaitgen dataset')
        
        data, data_M, labels = [], [], []
        samples = joblib.load(pjoin(self.motion_dir, paths['gaitgen']['file_name']))
        
        samples_num = len(samples)

                
        for k in tqdm(samples.keys()):
            sample = samples[k]
            try:
                motion = sample['feat_data']
                data.append(motion)
                # # read score labels
                # if 'mixmatch' in name:
                #     score = 3
                # else:
                #     subject_id = name.split('_')[0][:3]
                #     visit_ID = name.split('_')[0][4:]
                #     annot_format_ID = f'{visit_ID}_{subject_id}'
                #     score = annotations[annotations['walk'] == annot_format_ID]['score'].values[0]
                labels.append(sample['updrs_score'])
            except Exception as e:
                print(e)
                pass
        
        print('info', f"Total number of motions {len(data)}, snippets {-1}")
        return data, labels

    
def get_dataloader(db, batch_size, split='test'):
    if db == 'tri':
        dataset = TRI(db)
    elif db == 'tri_pd':
        dataset = TRI_PD(db)
    elif db == 'pdgam':
        dataset = PDGaMDataset(db, split=split)
    elif db == 'gaitgen':
        dataset = GAITGenDataset(db)
    shuffle = (split == 'train')
    drop_last = (split == 'train')
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=drop_last, num_workers=4,
        shuffle=shuffle, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_fn
    )
    return data_loader, dataset




def custom_collate_fn(batch):
    motions, m_lens, labels = zip(*batch)
    motions = [torch.tensor(motion, dtype=torch.float32) for motion in motions]
    motions_padded = pad_sequence(motions, batch_first=True)
    m_lens = torch.tensor(m_lens, dtype=torch.int64)
    labels = torch.tensor(labels)
    return motions_padded, m_lens, labels