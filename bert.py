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
Bert Helpers. I don't fully understand this magic
"""
package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
    
class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len] + ["[SEP]"]
    
class BertTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)
    
class BertNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)
    
def get_bert_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    """
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    """
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]
    
class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)
    
def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]

def bert_custom_loss(data, targets):
    # Calculate batch weight
    np_weights = (targets.data).cpu().numpy()
    loss_weight = np_weights[:,1].mean()

    bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:,1])(data[:,0],targets[:,0])
    bce_loss_2 = nn.BCEWithLogitsLoss()(data[:,1],targets[:,1])
    bce_loss_3 = nn.BCEWithLogitsLoss()(data[:,2:],targets[:,2:])
        
    return (bce_loss_1 * loss_weight) + bce_loss_3 
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
LOCAL_TEST  = False         ## Local test - for test performance on train set only
SEED        = 42            ## Seed for enviroment
seed_everything(SEED)       ## Seed everything

MODEL_TYPE  = 'bert'        ## Type of the model 'embeddings' or 'bert'
BERT_MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE  = 26
MAX_LEN     = 260           ## Length of the sequinces - less is faster but underfit
MAX_LR      = 1e-5          ## LR For bert

NFOLDS      = 8             ## CV folds for NN and LGBM
N_EPOCH     = 1             
SINGLE_FOLD = True          ## Due time limit we can do 1 fold per kernel for bert model
MAKE_FOLD   = 3             ## Number of the fold to process for bert model
CUR_STEP    = 4

folds       = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

# OPTIONS
# :'classic_preprocess'
# :'classic_modified'
# :'mod_exp'
# :'mod_embedding'
# :'mod_bert'
TEST_PREPROCESS = 'mod_bert'

## File name for model and for output
FILE_NAME   = '_'.join([MODEL_TYPE,BERT_MODEL_NAME,TEST_PREPROCESS,str(MAX_LEN),'fold',str(MAKE_FOLD)])

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
    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', nrows=100000).fillna(0)
    test =  pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', skiprows=range(1,100000), nrows=100000).fillna(0)   
else:
    train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv').fillna(0)  
    test =  pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
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
train['target_main'] = train['target']
train['weights'] = weights
test['weights'] = 0



########################### Embeddings Model
#################################################################################
## Target and Aux Targets
label_cols = ['target_main', 'weights', 
              'target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']
OUTPUT_DIM = len(label_cols)
print('Memory in Gb', get_memory_usage())


########################### Bert Model config (for single fold)
#################################################################################
bert_model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=OUTPUT_DIM)
bert_tok = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=MAX_LEN), pre_rules=[], post_rules=[])
fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))



########################### Bert Model
#################################################################################
oof = np.zeros((len(train), OUTPUT_DIM), dtype=np.float32)
predictions = np.zeros((len(test), OUTPUT_DIM), dtype=np.float32)

bert_train = train[label_cols]
bert_train['comment_text'] = train['comment_text']
    
bert_test = pd.DataFrame()
bert_test['comment_text'] = test['comment_text']
    
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, np.where((train['target']>=0.5),1,0))):
    print('Fold:',fold_)
    if (not SINGLE_FOLD) or (MAKE_FOLD==(fold_+1)):
        databunch = TextDataBunch.from_df('.', bert_train.iloc[trn_idx,:].sample(frac=1,random_state=SEED+CUR_STEP), bert_train.iloc[val_idx,:], bert_test,
                      tokenizer=fastai_tokenizer,
                      vocab=fastai_bert_vocab,
                      include_bos=False,
                      include_eos=False,
                      text_cols='comment_text',
                      label_cols=label_cols,
                      bs=BATCH_SIZE,
                      collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
                    )     
                
        learner = Learner(databunch, bert_model, loss_func=bert_custom_loss)         
        if CUR_STEP!=1:
            learner.load('/kaggle/input/freeze-bert-1-s-uc-260ml-3e-8f-s-'+str(CUR_STEP-1)+'-f-'+str(MAKE_FOLD)+'/models/' + FILE_NAME)

        learner.fit_one_cycle(N_EPOCH, max_lr=MAX_LR)

        oof[val_idx] = get_preds_as_nparray(DatasetType.Valid).astype(np.float32)
        predictions += get_preds_as_nparray(DatasetType.Test).astype(np.float32)/NFOLDS

        validate_df(train.iloc[val_idx], oof[val_idx, 0], verbose=True)

        learner.save(FILE_NAME)


print('CV BIASED AUC:')    
validate_df(train, oof[:,0], verbose=True)

train_results = pd.DataFrame(np.column_stack((train['id'], oof)), columns=['id']+['var_'+str(k) for k in range(oof.shape[-1])])
test_results = pd.DataFrame(np.column_stack((test['id'], predictions)), columns=['id']+['var_'+str(k) for k in range(oof.shape[-1])])

output_set = pd.concat([train_results, test_results])
output_set['id'] = output_set['id'].astype(int)

if not LOCAL_TEST:
    output_set.to_pickle('global_output_bert_'+ str(SEED) + '_' + TEST_PREPROCESS + '_' + str(MAX_LEN) +'.pkl')
    
    

    


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