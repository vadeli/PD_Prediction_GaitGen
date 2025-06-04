import argparse
import os
import numpy as np
import pandas as pd
import torch
import joblib
import wandb
import random  
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from os.path import join as pjoin
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataloder import get_dataloader, custom_collate_fn
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from consts import paths, set_gaitgen_file_name


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        

class MotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MotionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def train_classifier(classifier, train_loader, val_loader, eval_wrapper, lr, num_epochs=20, device='cuda'):
    classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        
        for motions, m_lens, labels in train_loader:
            motions, labels, m_lens = motions.to(device), labels.to(device), m_lens.to(device)
            # Get motion embeddings from eval_wrapper
            embeddings = eval_wrapper.get_motion_embeddings_ordered(motions, m_lens) # (batch_size, 512)
            
            # Train the classifier
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
        epoch_loss = running_loss / len(train_loader.dataset)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        
        wandb.log({'epoch': epoch, 'train_loss': epoch_loss, 'train_macroF1': macro_f1})
        
        eval_f1, eval_loss = evaluate_classifier(classifier, val_loader, eval_wrapper, criterion, device)
        wandb.log({'val_loss': eval_loss, 'val_macroF1': eval_f1})
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Macro F1: {macro_f1:.4f} | Val Loss: {eval_loss:.4f} | Val Macro F1: {eval_f1:.4f}')
    
    return classifier


def evaluate_classifier(classifier, data_loader, eval_wrapper, criterion='', device='cuda', test=False):
    classifier.to(device)
    classifier.eval()
    val_running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for motions, m_lens, labels in data_loader:
            motions, m_lens, labels = motions.to(device), m_lens.to(device), labels.to(device)
            embeddings = eval_wrapper.get_motion_embeddings_ordered(motions, m_lens)
            
            outputs = classifier(embeddings)
            if criterion != '':
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * embeddings.size(0)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    if test:        
        report = classification_report(all_labels, all_predictions)
        return report, None
    else:
        val_loss = val_running_loss / len(data_loader.dataset)
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        return macro_f1, val_loss

# python downstream_task_train_test.py --model checkpoints/classifier_pdgam_synthetic.pth --exp_dataname pdgam_synthetic
# python downstream_task_train_test.py --model checkpoints/classifier_pdgam.pth --exp_dataname pdgam

# python downstream_task_train_test.py --exp_dataname pdgam_synthetic --numepoch 4000 --lr 0.00001
# python downstream_task_train_test.py --exp_dataname pdgam --numepoch 1500 --lr 0.0001
def arg_parse():
    parser = argparse.ArgumentParser(description='Downstream Task')
    parser.add_argument("--dataset_name", default='pdgam', help="datasets e.g., ['pdgam', 'gaitgen', 'tri', 'tri_pd']")
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--exp_dataname', default='pdgam_synthetic', type=str, help='dataset name "pdgam" (orig), pdgam_synthetic')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--model', type=str, default=None, help='path to the trained model')
    parser.add_argument('--numepoch', type=int, default=2000, help='number of epochs to train')
    
    
    opt = parser.parse_args()
    
    return opt

if __name__ == "__main__":
    opt = arg_parse()
    fixseed(3407)
    set_gaitgen_file_name('data_gaitgen.pkl')
    
    opt.outdir = pjoin('./exp', opt.exp_dataname)
    opt.checkpoints_dir = opt.model
    os.makedirs(opt.outdir, exist_ok=True)
    test_only = False
    
    
    classifier = MotionClassifier(512, opt.num_classes)
    classifier.to(opt.device)
    # Load the trained weights
    if opt.model is not None:
        model_path = opt.checkpoints_dir
        state_dict = torch.load(model_path, map_location=opt.device)
        classifier.load_state_dict(state_dict)
        classifier.eval() 
        test_only = True 
        print("====================================================")
        print(f"Loaded model from {model_path}")
        print("Running in test mode only.")
        print("====================================================")
    
    if not test_only:
        train_loader, train_dataset = get_dataloader(opt.dataset_name, opt.batch_size, split='train')
        print (f"Train dataset size: {len(train_loader)}")
    val_loader, val_dataset = get_dataloader(opt.dataset_name, opt.batch_size, split='test')
    
    if opt.exp_dataname == 'pdgam_synthetic' and not test_only:
        print("Using synthetic data for training")
        gaitgen_loader, gaitgen_dataset = get_dataloader('gaitgen', opt.batch_size, split='train')
        combined_train_dataset = ConcatDataset([train_dataset, gaitgen_dataset])
        train_loader = DataLoader(combined_train_dataset, batch_size=opt.batch_size, 
        drop_last=True, num_workers=1, shuffle=True, pin_memory=True, persistent_workers=True, collate_fn=custom_collate_fn)
        print (f"Train dataset size: {len(train_loader)}")

    
    dataset_opt_path = f'./checkpoints/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    
    wbname = f'{opt.exp_dataname}_lr{opt.lr}'
    print(f'Running experiment: {wbname}')
    wandb.init(project='GAITGen_downstream', name=wbname, config=vars(opt))
    
    if not test_only:
        classifier = train_classifier(classifier, train_loader, val_loader, eval_wrapper, opt.lr, num_epochs=opt.numepoch, device=opt.device)
    
    report, _ = evaluate_classifier(classifier, val_loader, eval_wrapper, device=opt.device, test=True)
    
    print(report)
    
    os.makedirs(pjoin(opt.outdir), exist_ok=True)
    with open(pjoin(opt.outdir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Save the model
    if not test_only:
        print(f"Saving classifier model to {pjoin('checkpoints', f'classifier_{opt.exp_dataname}.pth')}")
        torch.save(classifier.state_dict(), pjoin('checkpoints', f'classifier_{opt.exp_dataname}.pth'))
    
    wandb.finish()