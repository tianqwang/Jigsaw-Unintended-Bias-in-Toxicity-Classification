# General imports
import numpy as np
import pandas as pd
import os, sys, gc, re, warnings, pickle, itertools, emoji, psutil, random

# sklearn imports
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold

# fastai imports
import fastai
from fastai.train import Learner, DataBunch
from fastai.callbacks import *
from fastai.basic_data import DatasetType
from fastai.text import *

# keras for preprocessing and embeding model sequences
import keras
from keras import backend as K
from keras.preprocessing import text, sequence

# custom imports
from scipy import sparse                # Minifying np.array
from multiprocessing import Pool        # Multiprocess Runs

warnings.filterwarnings('ignore')
pd.options.display.max_columns = 10
pd.options.display.max_colwidth = 20


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
 
   
## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


## Validation
# :validation_df - DataFrame to make validation # type: pandas DataFrame
# :preds_df - DataFrame with predictions        # type: pandas DataFrame (columns ['id','prediction'])
# :verbose - print or not full report           # type: bool
def local_validation(validation_df, preds_df, verbose=False):
    validation_df = validation_df.merge(preds_df[['id', 'prediction']], on=['id'], how='left').dropna()
    print('Validation set size:', len(validation_df))
       
    for col in ['target'] + identity_columns:
        validation_df[col] = np.where(validation_df[col] >= 0.5, True, False)

    SUBGROUP_AUC = 'subgroup_auc'
    BPSN_AUC = 'bpsn_auc'  
    BNSP_AUC = 'bnsp_auc'  
    TOXICITY_COLUMN = 'target'

    def compute_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan
    
    def compute_subgroup_auc(df, subgroup, label, model_name):
        return compute_auc(df[df[subgroup]][label], df[df[subgroup]][model_name])
    
    def compute_bpsn_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
        subgroup_negative_examples = df[df[subgroup] & ~df[label]]
        non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
        examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        return compute_auc(examples[label], examples[model_name])
    
    def compute_bnsp_auc(df, subgroup, label, model_name):
        """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
        subgroup_positive_examples = df[df[subgroup] & df[label]]
        non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
        examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        return compute_auc(examples[label], examples[model_name])
    
    def compute_bias_metrics_for_model(dataset, subgroups, model, label_col, include_asegs=False):
        """Computes per-subgroup metrics for all subgroups and one model."""
        records = []
        for subgroup in subgroups:
            record = {
                'subgroup': subgroup,
                'subgroup_size': len(dataset[dataset[subgroup]])
            }
            record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
            record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
            record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
            records.append(record)
        return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
    
    def calculate_overall_auc(df, model_name):
        return metrics.roc_auc_score(df[TOXICITY_COLUMN], df[model_name])
    
    def power_mean(series, p):
        return np.power(sum(np.power(series, p)) / len(series), 1 / p)
    
    def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
        bias_score = np.average([
            power_mean(bias_df[SUBGROUP_AUC], POWER),
            power_mean(bias_df[BPSN_AUC], POWER),
            power_mean(bias_df[BNSP_AUC], POWER)
        ])
        return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

    bias_metrics_df = compute_bias_metrics_for_model(validation_df, identity_columns, 'prediction', 'target')
    if verbose:
        print(bias_metrics_df)
    print(get_final_metric(bias_metrics_df, calculate_overall_auc(validation_df, 'prediction')))

def validate_df(df, preds, verbose=True, val_df='train'):
    df = df.copy()
    df['prediction'] = preds
    if val_df=='train':
        local_validation(train, df, verbose)  
    else:
        local_validation(test, df, verbose)  
## ----------------------------------------------------------------------------------------------------


## ----------------------------------------------------------------------------------------------------
"""
Tokinization and embedings helpers. Very simple and clear.
We use keras for tokenization.
"""
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)

## Prepare Final embedings matrix
# :emb_dict - dict                              # type: dict {'NAME': str: ['file_path': str, 'vector_dimension': int]}
# :tokenizer - tokenizer                        # type: keras tokenizer
def get_matrix(emb_dict, tokenizer):
    for i, matrix in enumerate(emb_dict):
        tmp_matrix, tmp_unknown_words = build_matrix(emb_dict[matrix][0], emb_dict[matrix][1], tokenizer)
        if i==0: embedding_matrix = tmp_matrix.copy().astype(np.float32)
        else: embedding_matrix = np.concatenate([embedding_matrix.astype(np.float32), tmp_matrix.astype(np.float32)], axis=-1).astype(np.float32)
        print('Unknown words', matrix, len(tmp_unknown_words))
    print('Check embedding_matrix Shape', embedding_matrix.shape)    
    return embedding_matrix.astype(np.float32)
    
## Builds matrix from path and do matching with tokinizer
# :path - file path for embedding vectors       # type: str
# :emb_dim - embedding vector dimension         # type: int
def build_matrix(path, emb_dim, tokenizer):
    word_index = tokenizer.word_index
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
    unknown_words = set(word_index).difference(embedding_index)
    lower_word_index = {k:v for k, v in word_index.items() if (k in unknown_words) and (k.lower() in embedding_index)}
    word_index = {k:v for k, v in word_index.items() if k not in unknown_words}

    # add normal vector
    for word, i in word_index.items():
        embedding_matrix[i] = embedding_index[word]
    
    return embedding_matrix.astype(np.float32), list(set(unknown_words).difference(set(lower_word_index)))

# Prepare sequences
def make_sequences(x_train, x_test):
    tokenizer = text.Tokenizer(lower=False, filters='')
    tokenizer.fit_on_texts(list(x_train)+list(x_test))
    x_train, x_test = tokenizer.texts_to_sequences(x_train), tokenizer.texts_to_sequences(x_test)
    x_train, x_test = sequence.pad_sequences(x_train, maxlen=MAX_LEN), sequence.pad_sequences(x_test, maxlen=MAX_LEN)
    return x_train, x_test, tokenizer
## ----------------------------------------------------------------------------------------------------

## ----------------------------------------------------------------------------------------------------
"""
Embedding Helpers and Model. 
"""
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

## Disable progress bar for FastAi
import fastprogress
from fastprogress import force_console_behavior
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DHU, DHU)
        self.linear2 = nn.Linear(DHU, DHU)
        
        self.linear_out = nn.Linear(DHU, 1)
        self.linear_aux_out = nn.Linear(DHU, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out
        
def custom_loss(data, targets):

    np_weights = (targets.data).cpu().numpy()
    loss_weight = np_weights[:,1].mean()

    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])
    return (bce_loss_1 * loss_weight) + bce_loss_2

def get_loaders(trn_idx, val_idx):

    x_train_torch = torch.tensor(x_train.A[trn_idx], dtype=torch.long)
    valid_torch = torch.tensor(x_train.A[val_idx], dtype=torch.long)
    x_test_torch = torch.tensor(x_test.A, dtype=torch.long)

    y_train_torch = torch.tensor(y_train_final, dtype=torch.float32)
    
    test_dataset = data.TensorDataset(x_test_torch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    train_dataset = data.TensorDataset(x_train_torch, y_train_torch[trn_idx])
    valid_dataset = data.TensorDataset(valid_torch, y_train_torch[val_idx])    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader)

    return databunch, test_loader

def train_model(learn, output_dim, lr=0.001, batch_size=512, n_epochs=4):

    learn.fit_one_cycle(n_epochs, max_lr=lr)

    val_preds = np.zeros((len(val_idx), output_dim))    
    test_preds = np.zeros((len(test), output_dim))    
            
    for i, x_batch in enumerate(test_loader):
        X = x_batch[0].cuda()
        y_pred = sigmoid(learn.model(X).detach().cpu().numpy())
        test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred
            
    for i, x_batch in enumerate(databunch.valid_dl):
        X = x_batch[0].cuda()
        y_pred = sigmoid(learn.model(X).detach().cpu().numpy())
        val_preds[i * batch_size:(i+1) * batch_size, :] = y_pred
        
    validate_df(train.iloc[val_idx], val_preds[:,0], verbose=False)
        
    return val_preds, test_preds

def make_embedings():

    model = NeuralNet(embedding_matrix, y_aux_train.shape[-1])
    learn = Learner(databunch, model, loss_func=custom_loss)
    val_preds, test_preds = train_model(learn, output_dim=y_aux_train.shape[-1]+1, batch_size=BATCH_SIZE, n_epochs=N_EPOCH)    

    return val_preds, test_preds
## ----------------------------------------------------------------------------------------------------

## ----------------------------------------------------------------------------------------------------
def update_comment_text(train, test, TEST_PREPROCESS):
    tt_1 = pd.read_pickle('../input/jigsaw-preprocess-collections/'+TEST_PREPROCESS+'_x_train.pkl')
    tt_2 = pd.read_pickle('../input/jigsaw-preprocess-collections/'+TEST_PREPROCESS+'_x_test.pkl')
    preprocessed_comments = pd.concat([tt_1, tt_2])
    
    x_train = train.merge(preprocessed_comments, on='id', how='left').set_index(train.index)['p_comment']
    x_test = test.merge(preprocessed_comments, on='id', how='left').set_index(test.index)['p_comment']

    train['comment_text'] = x_train.astype(str)
    test['comment_text'] = x_test.astype(str)
    return train, test
## ----------------------------------------------------------------------------------------------------

## ----------------------------------------------------------------------------------------------------
def split_text(text, part):
    text = text.split()
    if len(text)<=MAX_LEN:
        if part=='first':
            return ' '.join(text)
        elif part=='second':
            return ''         
    
    else:
        cutoff = MAX_LEN
        for i in range(MAX_LEN-50,MAX_LEN):
            if (text[i]=='.') or (text[i]=='!') or (text[i]=='?'):
               cutoff = i+1
    
        if part=='first':
            text = text[:cutoff] 
        elif part=='second':
            text = text[cutoff:] 
        return ' '.join(text)    
## ----------------------------------------------------------------------------------------------------

## ----------------------------------------------------------------------------------------------------
def make_split(train, test):
    l_train_max = train['comment_text'].apply(lambda x: len([i for i in x.split()]))
    l_test_max = test['comment_text'].apply(lambda x: len([i for i in x.split()]))
    
    test = pd.concat([test, train[l_train_max>MAX_LEN]]).fillna(0).reset_index(drop=True)
    train = train[l_train_max<=MAX_LEN].reset_index(drop=True)
    
    out_of_len_1, out_of_len_2, out_of_len_3 = test.copy(), test.copy(), test.copy()
    
    out_of_len_1['comment_text'] = test['comment_text'].apply(lambda x: split_text(x, 'first'))
    out_of_len_2['comment_text'] = test['comment_text'].apply(lambda x: split_text(x, 'second'))
    out_of_len_3['comment_text'] = out_of_len_2['comment_text'].apply(lambda x: split_text(x, 'second'))
    out_of_len_2['comment_text'] = out_of_len_2['comment_text'].apply(lambda x: split_text(x, 'first'))
    
    test = pd.concat([out_of_len_1, out_of_len_2, out_of_len_3]).reset_index(drop=True)
    test = test[test['comment_text']!='']
    
    return train, test
## ----------------------------------------------------------------------------------------------------











 

########################### Initial vars
#################################################################################
CRAWL_EMBEDDING_PATH = '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'
GOOGLE_EMBEDDING_PATH = '../input/mod-google-new/mod_google_new.pkl'
ENGLISH_EMBEDDING_PATH = '../input/english-col-embedding/mod_english_col.pkl'

EMBEDDING_DICT = {
#    'english':  [ENGLISH_EMBEDDING_PATH, 100],
    'fasttext': [CRAWL_EMBEDDING_PATH, 300],
    'glove':    [GLOVE_EMBEDDING_PATH, 300],
#    'google':   [GOOGLE_EMBEDDING_PATH, 300],
}


LOCAL_TEST  = False         ## Local test - for test performance on train set only
SEED        = 42            ## Seed for enviroment
seed_everything(SEED)       ## Seed everything

MAX_LEN     = 300           ## Length of the sequinces - less is faster but underfit
NFOLDS      = 5             ## CV folds for NN and LGBM
folds       = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

N_EPOCH     = 5
LSTM_UNITS  = 128           ## torch vars
DHU         = 4 * LSTM_UNITS    ## Dense Hidden Units
BATCH_SIZE  = 128           ## Global batch size

# OPTIONS
# :'classic_preprocess'
# :'classic_modified'
# :'mod_exp'
# :'mod_embedding'
# :'mod_bert'
TEST_PREPROCESS = 'classic_modified'

## Identity columns for bias ROC-AUC metric 
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

good_cols = ['id', 'target', 'comment_text',
            'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit', 
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
            
            
########################### DATA LOAD
#################################################################################
print('1.1. Load Data')
if LOCAL_TEST:
    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', nrows=200000).fillna(0)
    test =  pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', skiprows=range(1,200000), nrows=100000).fillna(0)   
else:
    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv').fillna(0)  
    test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
    train = train[good_cols]


########################### Text Cleaning
#################################################################################
## Get preprocessed train and level 1 test data
## For 2nd level test data we have to do preprocessing in "live" mode
train, test = update_comment_text(train, test, TEST_PREPROCESS)


########################### Split Train dataset
#################################################################################
print('1.x. Split data set by MAX-LEN')
if MAX_LEN<200:
    train, test = make_split(train, test)


########################### Target and weights
#################################################################################
print('1.x. Targets and Weights')
## Item weights (from some public kernel) - please see if we can improve it
cut_threshold = 0.5     # In some tests lower threshold gave significant boost
weights = np.ones((len(train),)) / 4
weights += (train[identity_columns].fillna(0).values>=cut_threshold).sum(axis=1).astype(bool).astype(np.int) / 4
weights += (( (train['target'].values>=cut_threshold).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values<cut_threshold).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
weights += (( (train['target'].values<cut_threshold).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values>=cut_threshold).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
loss_weight = 1.0 / weights.mean()


########################### Embeddings Model
#################################################################################
## Target and Aux Targets
y_train = np.vstack([train['target'].values,weights]).T
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']]
y_train_final = np.hstack([y_train, y_aux_train])
OUTPUT_DIM = y_train_final.shape[-1]-1
print('Memory in Gb', get_memory_usage())
    
    
########################### Do tokenization
#################################################################################
print('1.x. Tokenization')
x_train, x_test, tokenizer = make_sequences(train['comment_text'], test['comment_text'])
max_features = len(tokenizer.word_index) + 1
print('Memory in Gb', get_memory_usage())    
        
    
########################### Prepare Embedding Matrix
#################################################################################
print('1.x. Emedding Matrix')
embedding_matrix = get_matrix(EMBEDDING_DICT, tokenizer)
print('Memory in Gb', get_memory_usage())


########################### Cleaning
#################################################################################
print('1.x. Cleaning')
token_text = 0
tokenizer = 0

del train['comment_text'], test['comment_text']
train.iloc[:,1:] = train.iloc[:,1:].astype(np.float16) # We dont need float64 as we will use it only for validation 
del token_text, tokenizer;gc.collect()

for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))
print('Memory in Gb', get_memory_usage())
    
# Convert to sparse matrix
x_train = sparse.csr_matrix(x_train)
x_test = sparse.csr_matrix(x_test)
print('Memory in Gb', get_memory_usage())
        
    
########################### Embeddings Model
#################################################################################
print('1.6. Embeddings Model')
    
oof = np.zeros((len(train), OUTPUT_DIM), dtype=np.float32)
predictions = np.zeros((len(test), OUTPUT_DIM), dtype=np.float32)
    
for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train, np.where((train['target']>=0.5),1,0))):
    print('Fold:',fold_)
    databunch, test_loader = get_loaders(trn_idx, val_idx)
    print('Memory in Gb', get_memory_usage())
        
    vl_p, tt_p = make_embedings()
    del databunch, test_loader

    oof[val_idx] = vl_p.astype(np.float32)
    predictions += tt_p.astype(np.float32)/NFOLDS
        
    del vl_p, tt_p
    gc.collect()

print('CV BIASED AUC:')    
validate_df(train, oof[:,0], verbose=True)

train_results = pd.DataFrame(np.column_stack((train['id'], oof)), columns=['id']+['var_'+str(k) for k in range(oof.shape[-1])])
test_results = pd.DataFrame(np.column_stack((test['id'], predictions)), columns=['id']+['var_'+str(k) for k in range(oof.shape[-1])])

output_set = pd.concat([train_results, test_results]).reset_index(drop=True)
output_set['id'] = output_set['id'].astype(int)

if not LOCAL_TEST:
    output_set.to_pickle('global_out_put_'+ str(SEED) + '_' + TEST_PREPROCESS + '_' + str(MAX_LEN) +'.pkl')
    
    

    


########################### Validation and Output
#################################################################################
g_parameters = {
    'LOCAL_TEST': LOCAL_TEST,
    'TEST_PREPROCESS': TEST_PREPROCESS,
    'SEED': SEED, 
    'MAX_LEN': MAX_LEN,
    'NFOLDS': NFOLDS,
    'N_EPOCH': N_EPOCH,
    'BATCH_SIZE': BATCH_SIZE,
}
    
print('#'*10, 'GLOBAL Parameters')
for i in g_parameters:
    print(i,'-',g_parameters[i])
print('#'*20,'\n')
