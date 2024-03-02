import sys
import torch
import json
from torch.utils.data import DataLoader
from bleu_eval import BLEU
import pickle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import numpy as np
import random
from scipy.special import expit
import sys
import os
import json
from collections import Counter
import re
import pickle
from torch.utils.data import DataLoader, Dataset
from models import Models, Attention, EncoderLSTM, DecoderLSTMWithAttention


class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
            
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]

def generate_predictions(test_loader, model, idx_to_word):
    model.eval()
    predictions = []
    
    for batch_idx, batch in enumerate(test_loader):
        ids, avi_feats = batch
        avi_feats = avi_feats.float().cuda()
        
        seq_log_probs, seq_predictions = model(avi_feats, mode='inference')
        test_predictions = seq_predictions
        
        result = [[idx_to_word[x.item()] if x.item() in idx_to_word else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        rr = zip(ids, result)
        for r in rr:
            predictions.append(r)
    return predictions

# Load the model
model_path = 'S2S_Model.h5'
model = torch.load(model_path, map_location=lambda storage, loc: storage)
print(model)

# Load the testing dataset
test_dataset_path = sys.argv[1]
test_data = test_data(test_dataset_path)
testing_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# Load the index to word mapping
index_to_word_path = 'index_to_word.pickle'
with open(index_to_word_path, 'rb') as handle:
    index_to_word = pickle.load(handle)

model = model.cuda()

predictions = generate_predictions(testing_loader, model, index_to_word)

# Save predictions to file
output_file_path = sys.argv[2]
with open(output_file_path, 'w') as f:
    for id, caption in predictions:
        f.write('{},{}\n'.format(id, caption))

test_labels = json.load(open("testing_label.json"))

predicted_captions = {}
with open(output_file_path,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        predicted_captions[test_id] = caption

# Calculate BLEU scores
bleu=[]
for item in test_labels:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(predicted_captions[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("Average bleu score is " + str(average))