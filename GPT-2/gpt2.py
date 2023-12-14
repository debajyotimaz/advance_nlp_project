# -*- coding: utf-8 -*-

#@title Specifying Upsampling Option
UPSAMPLE = True #@param {type:"boolean"}

#!pip install Keras-Preprocessing

from torch import nn
from keras_preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output

import tarfile
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, recall_score
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import KFold
# from sklearn.model_selection import StratifiedKFold
#!pip install imbalanced-learn
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score, roc_auc_score

"""# **Loading data**

Connects google drive to this notebook to get the dataset, and then unzip it. Then, we load the specified columns (docID, keyword, country, paragraph and label) into the df.
"""

# Unzip the dataset
#!unzip "/content/dontpatronizeme_v1.3.zip" -d "/content/"

# Opening the file from MyDrive
file = open(r'/data1/debajyoti/adv_nlp_project/dontpatronizeme_pcl.tsv')
reader = csv.reader(file, delimiter="\t")
data = []
for row in reader:
  data.append(row)

# Create a df using the specified columns
df = pd.DataFrame(data[5:],  columns = ['serial', 'docID', 'keyword', 'country', 'paragraph', 'label' ] )
df

"""# **Exploring data**

Get the length of each data instance and plot its histogram.
"""

# Length of text
def length (txt):
  length = len(txt.split())
  return length

txt_length = df['paragraph'].apply(lambda x: length(x))
txt_length.sort_values(ascending = False)

"""None of the column contains missing values."""

#checking for missing values
print('Is null: \n', df.isnull().sum() )

"""There are 5 labels in this dataset. The label number represents the degree of patronizing and condescending language (PCL).  Label 0 are sentences that do not contain patronizing nor condescending language, where as Label 4 are sentences with highly PCL. Based on the value counts below, the majority of data is negative with a label 0, whereas only 946 records are positive. Positive data contain labels from 1 to 4.

In this project, we choose task 1 that is a binary classification of PCL. Therefore, we convert the labels into 0 and 1, where 0 means negative and 1 means positive. The negative data contain label 0 originally, and the positive data contain labels 1 to 4.
"""

# Convert label data type to string
df['label'] = df['label'].astype(str)
df['label']

# One hot encoding labels
label_dic = {'0':0,
             '1':0,
             '2':1,
             '3':1,
             '4':1}
df['label'] = df['label'].map(label_dic)
print(df['label'].value_counts())
sns.countplot(x='label', data=df)
plt.show()

"""# **Data Partition**

Once the data is thoroughly explored, we proceed to partition the shuffled data into training and testing sets with a ratio of 80-20.
"""

# Splitting the data into training (80%) and test set(20%)
from sklearn.model_selection import train_test_split
X = df['paragraph']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split (X, y, train_size = 0.8, random_state = 42, shuffle = True, stratify=y)
print ('Shapes of X_train, y_train: ', X_train.shape, y_train.shape)
print ('Shapes of X_test, y_test: ', X_test.shape, y_test.shape)
print(y_train.value_counts())

"""# **Create helper functions for printing and recording performance measures**

This part creates 2 helper function, where printing_eval_scores is responsible for printing out the model performance and get_roc_curve is responsible for extracting different metrics such as false positive rate (FPR), recall, macro-f1 scores and auc score. These 2 functions take the ground truth labels and predicted labels as inputs.

Since the data is very imbalanced, we decide to use macro-F1 score as a standard metric to compare among the candidate models. Macro-F1 score takes both major class and minor class into account regardless of their sizes. Therefore, using this metric will give us a more precise baseline to compare among the models.
"""

# Printing model performance 
def printing_eval_scores (y_true, y_pred):
  print('accuracy score: {}'.format(sklearn.metrics.accuracy_score(y_true, y_pred)))
  print('precision score: {}'.format(sklearn.metrics.precision_score(y_true, y_pred, average = 'macro', zero_division=1)))
  print('recall score: {}'.format(sklearn.metrics.recall_score(y_true, y_pred,  average = 'macro', zero_division=1)))
  print('F1 score: {}'.format(f1_score(y_true, y_pred,  average = 'macro', zero_division=1)))
  print('\nConfusion Matrix:\n', confusion_matrix(y_true, y_pred))
  print('\n', classification_report(y_true, y_pred))

# Get the measurements of ROC curve for each model
def get_roc_cuve (y_true, y_pred):
  # Get arrays of FPR and recall using roc_curve
  FPR, recall, threshold = sklearn.metrics.roc_curve(y_true, y_pred)

  # Get testing accuracy:
  acc = accuracy_score( y_test,y_pred)

  # Get testing macro-f1:
  f1 = f1_score(y_test, y_pred,  average = 'macro', zero_division=1)

  # Get auc score
  auc = sklearn.metrics.auc(FPR, recall)
  roc = {'fpr': FPR, 'tpr': recall, 'auc': auc, 'accuracy': acc, 'macro-F1': f1}
  return roc

"""# **GPT-2**

**Get maximum length of a sentence**

 If it's greater than 512, MAX_LEN is set to be 225. We tried to set it with 512, but we run out of GPU memory, so we decide to set it as 225.
"""

# get max len in tokenized train text to set the tokens length in the next step
MAX_LEN = max(map(len, X_train))  # 
print('MAX LEN of trainning sentence is:', MAX_LEN, '\nMAX LEN > 512 is ', MAX_LEN>512)

# Update MAX LEN if it's > 512, set it to be 225 
## 512 is is the maximum seq len of BERT_BASE. But we cannot allow the seq len to be 512 since we'll run out of GPU memory --> Use max len of 225
MAX_LEN = 225 if MAX_LEN > 512 else MAX_LEN

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

#!pip install transformers

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# Load GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-small')

"""Since we use a MAX_LEN of 225, some sentences will be shorter than that. Therefore, we need to embed its empty space on the right after tokenizing the data."""

# Padding sequences from the right to MAX_LEN (225) and tokenize train set
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
train_tokens = tokenizer(X_train.to_list(), return_tensors='pt', truncation=True, padding=True, max_length = MAX_LEN)
test_tokens  = tokenizer(X_test.to_list(),  return_tensors='pt', truncation=True, padding=True, max_length = MAX_LEN)

# This is the embedded word vector of each training instance
train_tokens.input_ids

"""Upsampling option is available at the top of this notebook. If var UPSAMPLE is checked, the data will be upsampled using SMOTE."""

# If Upsampling option for GPT is on, then upsample data using SMOTE first.
# Then, convert List of words to list of numbers. (Words are replaced by their index in the dictionary)
if UPSAMPLE:
  oversampler = SMOTE(random_state=42)
  train_tokens_ids, train_labels = oversampler.fit_resample(train_tokens.input_ids, y_train)
else:
  train_tokens_ids = train_tokens.input_ids
  train_labels = y_train

# Do the same thing to the test set
test_tokens_ids = test_tokens.input_ids

train_tokens_ids.shape, train_labels.shape

# Mask the paddings with 0 and words with 1
train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
test_masks = [[float(i > 0) for i in ii] for ii in test_tokens_ids]

# Define GPT binary classifier
class GTP2BinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(GTP2BinaryClassifier, self).__init__()
        self.gtp2 = GPT2ForSequenceClassification.from_pretrained('microsoft/DialoGPT-small')
      
    def train_m(self,x,y,train_mask,epochs,batchsize):
      train_tokens_tensor = torch.tensor(x)
      train_y_tensor = torch.tensor(y.reshape(-1, 1)).long()
      train_masks_tensor = torch.tensor(train_mask)

      train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
      train_sampler = RandomSampler(train_dataset)
      train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batchsize) 


      # param_optimizer = list(self.gtp2.parameters()) 
      # optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
      optimizer = Adam(self.gtp2.parameters(), lr=5e-5)
      for epoch_num in range(epochs):
          self.gtp2.train() # Training Flag
          train_loss = 0
          for step_num, batch_data in enumerate(train_dataloader):
              
              # Load batch on device memory
              token_ids, masks, labels = tuple(t.to(device) for t in batch_data)
              self.zero_grad()

              # Get the output of the model for provided input
              outputs = self.gtp2(token_ids,attention_mask=masks,labels=labels)
              loss, logits = outputs[:2]
              # logits = self(token_ids, masks)
              
              # Total Loss
              train_loss += loss.item()
              
              # Backward pass the loss
              loss.backward()
              torch.nn.utils.clip_grad_norm_(self.gtp2.parameters(), 1.0)
              
              optimizer.step()
              logits = logits.cpu().detach().numpy()

              clear_output(wait=True)
        
              print('Epoch: ', epoch_num + 1)
              print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_labels) / batchsize, train_loss / (step_num + 1)))

# Initiate GPT Classifier using cuda
gtp_clf = GTP2BinaryClassifier()
gtp_clf = gtp_clf.cuda()

EPOCHS = 3
BATCH_SZ = 4

# Configure the Padding token id
gtp_clf.gtp2.config.pad_token_id = tokenizer.eos_token_id

# Train GPT
gtp_clf.train_m(train_tokens_ids, train_labels.to_numpy(), train_masks, EPOCHS, BATCH_SZ)

train_tokens_ids.shape, train_labels.to_numpy().shape, len(train_masks)

"""**Evaluate on Testing Set**"""

## Converting test token ids, test labels and test masks to a tensor and the create a tensor dataset out of them.
# Convert token ids to tensor 
test_tokens_tensor = torch.tensor(test_tokens_ids)

# Convert labels to tensors
test_y_tensor = torch.tensor( y_test.to_numpy().reshape(-1, 1) ).long()

# Convert masks to tensor
test_masks_tensor = torch.tensor(test_masks)

# Load Token, token mask and label into Dataloader
test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)

# Define sampler
test_sampler = SequentialSampler(test_dataset)

# Define test data loader
test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size=16)

# Evaluate Model
gtp_clf.eval() # Define eval
gpt_predicted = [] # Store Result
with torch.no_grad():
    for step_num, batch_data in enumerate(test_dataloader):

        token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

        # ----------------------------------------------------------------
        outputs = gtp_clf.gtp2(token_ids, attention_mask=masks, labels=labels)
        loss, logits = outputs[:2]
        numpy_logits = logits.detach().cpu().numpy()
        # ----------------------------------------------------------------
        gpt_predicted +=list(numpy_logits.argmax(axis=-1).flatten().tolist())

rocs = {}
# Get ROC curve measurements
rocs['GPT'] = get_roc_cuve(y_test, gpt_predicted)

# Prin performance
print('----------------------------GPT performance---------------------------')
printing_eval_scores(y_test, gpt_predicted)

"""#**Visualize all models with ROC curves**

The model performances are visualized through an overlaid ROC curves. In this plot, the higher recall and the lower FPR that the model can achieve, the better that model is. A model that randomly classify data is represented as a straight dotted line in black.
"""

def graph_multi_ROC (rocs):
  # Set color for each model
  colors = {'LGBM': 'lightcoral','LR': 'darkorange', 'SVM':'lime', 'NB': 'steelblue',
            'XGB': 'purple','DT': 'magenta','RF': 'deeppink','KNN': 'darkturquoise',
            'BERT': 'darkred', 'GPT': 'blue'}
  # Set marker for each model          
  markers = {'LGBM':'1--','LR': 'v--', 'SVM': '^--', 'XGB': '*--', 'DT': 'o--', 'RF': '+--', 'KNN': '.--', 'NB': 'x--', 'BERT':'<--', 'GPT': '>--'}
  
  plt.figure(figsize=(9,6))
  for model in rocs:
    plt.plot( rocs[model]['fpr'], rocs[model]['tpr'], markers[model], color=colors[model], label= model+' - AUC=' + str(rocs[model]['auc'].round(3)) )
  
  plt.plot([0,1], [0,1], 'k--', label='Random Chances')
  plt.xlim([0.0,1.0])
  plt.ylim([0.0,1.02])
  plt.ylabel('Recall')
  plt.xlabel('False Positive Rate (1-Specificity)')
  plt.legend(loc='lower right') 
  plt.title( 'ROC Curves of all models')
  plt.show()

graph_multi_ROC(rocs)

# Export performance to a txt file

# Set file name according to Upsampling option
if UPSAMPLE == 1: name = 'GPT_SMOTE'
else: name = 'GPT'

txtfile = open( name + '.txt','w')
txtfile.write(name + '='+ str(rocs['GPT'])+'\n')

txtfile.close()

rocs
