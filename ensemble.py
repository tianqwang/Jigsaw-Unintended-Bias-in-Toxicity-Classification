import numpy as np
import pandas as pd
import os, sys, gc, re, time, warnings, pickle, itertools, psutil, random
start = time.time()

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

# custom imports
from multiprocessing import Pool        # Multiprocess Runs

# preprocess imports
import emoji, unicodedata
from gensim.utils import deaccent
from collections import Counter
from bs4 import BeautifulSoup
import collections as ct

# Torch imports
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

# Fastai imports
import fastai
from fastai.train import Learner, DataBunch
from fastai.callbacks import *
from fastai.basic_data import DatasetType
from fastai.text import *

# keras for preprocessing and embeding model sequences
import keras
from keras import backend as K
from keras.preprocessing import text, sequence

## Disable progress bar for FastAi
import fastprogress
from fastprogress import force_console_behavior
fastprogress.fastprogress.NO_BAR = True
master_bar, progress_bar = force_console_behavior()
fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar


warnings.filterwarnings('ignore')

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
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 
        
## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

## Multiprocessing Run.
# :df - DataFrame to split                      # type: pandas DataFrame
# :func - Function to apply on each split       # type: python function
# This function is NOT 'bulletproof', be carefull and pass only correct types of variables.
def df_parallelize_run(df, func):
    num_partitions, num_cores = psutil.cpu_count(), psutil.cpu_count()  # number of partitions and cores
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def get_preds_as_nparray(learner,databunch,ds_type) -> np.ndarray:
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]
## ----------------------------------------------------------------------------------------------------

## ----------------------------------------------------------------------------------------------------
def make_split(processed_comments, cur_len):
    df = test.copy()
    df['comment_text'] = processed_comments

    out_of_len_1, out_of_len_2, out_of_len_3 = df.copy(), df.copy(), df.copy()
    out_of_len_1['comment_text'] = df['comment_text'].apply(lambda x: ' '.join(x.split()[:cur_len]))
    out_of_len_2['comment_text'] = df['comment_text'].apply(lambda x: ' '.join(x.split()[cur_len:cur_len*2]))
    out_of_len_3['comment_text'] = df['comment_text'].apply(lambda x: ' '.join(x.split()[cur_len*2:]))

    out_of_len_1 = out_of_len_1[out_of_len_1['comment_text']!='']
    out_of_len_2 = out_of_len_2[out_of_len_2['comment_text']!='']
    out_of_len_3 = out_of_len_3[out_of_len_3['comment_text']!='']
    
    df = pd.concat([out_of_len_1, out_of_len_2, out_of_len_3]).reset_index(drop=True)
    
    return df
## ----------------------------------------------------------------------------------------------------






########################### Initial vars
#################################################################################
SEED                = 42            ## Seed for enviroment
NFOLDS              = 5             ## CV folds for LGBM
folds               = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
seed_everything(SEED)               ## Seed everything

## Identity columns for bias ROC-AUC metric 
identity_columns    = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                       'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

label_cols_set_1    = ['target_main', 'weights', 
                       'target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']


########################### DATA LOAD
#################################################################################
print('1.1. Load Data')
train =  pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv', usecols=['id','target']+identity_columns).fillna(0)
test =  pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 1: Loading Data - Used time - {time.time()-start:.2f}s')
print('#'*20)
## ----------------------------------------------------------------------------------------------------



#################################################################################
#################################################################################
########################### Preprocess Data
#################################################################################
#################################################################################
print('1.2. Preprocess Data')
## -------------------------------------
def mod_bert(data):

    def domain_search(text):
        re_3986_enhanced = re.compile(r"""
            # Parse and capture RFC-3986 Generic URI components.
            ^                                    # anchor to beginning of string
            (?:  (?P<scheme>    [^:/?#\s]+):// )?  # capture optional scheme
            (?:(?P<authority>  [^/?#\s]*)  )?  # capture optional authority
                 (?P<path>        [^?#\s]*)      # capture required path
            (?:\?(?P<query>        [^#\s]*)  )?  # capture optional query
            (?:\#(?P<fragment>      [^\s]*)  )?  # capture optional fragment
            $                                    # anchor to end of string
            """, re.MULTILINE | re.VERBOSE)
    
        re_domain =  re.compile(r"""
            # Pick out top two levels of DNS domain from authority.
            (?P<domain>[^.]+\.[A-Za-z]{2,6})  # $domain: top two domain levels.
            (?::[0-9]*)?                      # Optional port number.
            $                                 # Anchor to end of string.
            """, 
            re.MULTILINE | re.VERBOSE)
        try:
            return re_domain.search(re_3986_enhanced.match(text).group('authority')).group('domain')
        except:
            return 'url'
        
    def check_vocab(c_list, vocabulary, response='default'):
        try:
            words = set([w for line in c_list for w in line.split()])
            u_list = words.difference(set(vocabulary))
            k_list = words.difference(u_list)
        
            if response=='default':
                print('Unknown words:', len(u_list), '| Known words:', len(k_list))
            elif response=='unknown_list':
                return list(u_list)
            elif response=='known_list':
                return list(k_list)
        except:
            return []
            
    ## Load helper helper))
    def load_helper_file(HELPER_PATH, filename):
        with open(HELPER_PATH+filename+'.pickle', 'rb') as f:
            temp_obj = pickle.load(f)
        return temp_obj
    
    ## Preprocess helpers
    def place_hold(w):
        WPLACEHOLDER            = 'word_placeholder'
        return WPLACEHOLDER + '['+re.sub(' ', '___', w)+']'
    
    def check_replace(w):
        WPLACEHOLDER            = 'word_placeholder'
        return not bool(re.search(WPLACEHOLDER, w))
    
    def make_cleaning(s, c_dict):
        if check_replace(s):
            s = s.translate(c_dict)
        return s
      
    def make_dict_cleaning(s, w_dict):
        if check_replace(s):
            s = w_dict.get(s, s)
        return s
        
    HELPER_PATH             = '../input/stage-2-general-helpers/'
    local_vocab             = load_helper_file(HELPER_PATH,'helper_bert_cased_vocabulary')
    local_vocab_b           = load_helper_file(HELPER_PATH,'helper_bert_uncased_vocabulary')
    
    #local_vocab             = load_helper_file(HELPER_PATH,'helper_bert_uncased_vocabulary')
    bert_char_list          = list(set([c for line in local_vocab+local_vocab_b for c in line]))

    white_list_chars        = load_helper_file(HELPER_PATH,'helper_white_list_chars')
    white_list_punct        = " '*-.,?!/:;_()[]{}<>=" + '"'
    normalized_chars        = load_helper_file(HELPER_PATH,'helper_normalized_chars')
    html_tags               = load_helper_file(HELPER_PATH,'helper_html_tags')
    url_extensions          = load_helper_file(HELPER_PATH,'helper_url_extensions')
    pictograms_to_emoji     = load_helper_file(HELPER_PATH,'helper_pictograms_to_emoji')
    toxic_misspell_dict     = load_helper_file(HELPER_PATH,'helper_toxic_misspell_dict')        
    helper_contractions     = load_helper_file(HELPER_PATH,'helper_contractions')

    #data = test['comment_text']
    data = data.astype(str)
    
    global_chars_list = list(set([c for line in data for c in line]))
        

    # Normalize chars and dots - SEE HELPER FOR DETAILS
    data = data.apply(lambda x: ' '.join([make_cleaning(i,normalized_chars) for i in x.split()]))
    data = data.apply(lambda x: re.sub('\(dot\)', '.', x))
    data = data.apply(lambda x: deaccent(x))

    # Remove 'control' chars / hrefs
    chars_dict = {c:'' for c in global_chars_list if unicodedata.category(c)[0]=='C'}
    data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))
    data = data.apply(lambda x: re.sub(re.findall(r'\<a(.*?)\>', x)[0], '', x) if (len(re.findall(r'\<a (.*?)\>', x))>0) and ('href' in re.findall(r'\<a (.*?)\>', x)[0]) else x)

    # Convert or remove Bad Symbols
    chars = ''.join([c for c in global_chars_list if (c not in bert_char_list) and (c not in emoji.UNICODE_EMOJI) and (c not in white_list_chars)])
    chars_dict = {}
    for char in chars:
        try:
            new_char = unicodedata.name(char).split()[-1:][0].lower()
            if len(new_char)==1:
                chars_dict[ord(char)] = new_char
            else:
                chars_dict[ord(char)] = ''
        except:
            chars_dict[ord(char)] = ''
    data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))
    
    # Remove Bad Symbols PART 2
    chars = '·' + ''.join([c for c in global_chars_list if (c not in white_list_chars) and (c not in emoji.UNICODE_EMOJI) and (c not in white_list_punct) and (ord(c)>256)])
    chars_dict = {}
    for char in chars:
        try:
            new_char = unicodedata.name(char).split()[-1:][0].lower()
            if len(new_char)==1:
                chars_dict[ord(char)] = new_char
            else:
                chars_dict[ord(char)] = ''
        except:
            chars_dict[ord(char)] = ''
    data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

    # Remove html tags
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if ('<' in word) and ('>' in word):
            for tag in html_tags:
                if ('<'+tag+'>' in word) or ('</'+tag+'>' in word):
                    temp_dict[word] = BeautifulSoup(word, 'html5lib').text  
    data = data.apply(lambda x: ' '.join([temp_dict.get(i, i) for i in x.split()]))

    # Remove links (There is valuable information in links (probably you will find a way to use it)) 
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    url_rule = r'(?P<url>https?://[^\s]+)'
    temp_dict = {k:domain_search(k) for k in temp_vocab if k!= re.compile(url_rule).sub('url', k)}
    for word in temp_dict:
        new_value = temp_dict[word]
        if word.find('http')>2:
            temp_dict[word] =  word[:word.find('http')] + ' ' + place_hold(new_value)
        else:
            temp_dict[word] = new_value  #place_hold(new_value)
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Convert urls part 2
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        url_check = False
        if 'file:' in word:
            url_check = True
        elif ('http' in word) or ('ww.' in word) or ('.htm' in word) or ('ftp' in word) or ('.php' in word) or ('.aspx' in word):
            if 'Aww' not in word:
                for d_zone in url_extensions:
                    if '.' + d_zone in word:
                        url_check = True
                        break            
        elif ('/' in word) and ('.' in word):
            for d_zone in url_extensions:
                if '.' + d_zone + '/' in word:
                    url_check = True
                    break
        if url_check:
            temp_dict[word] =  place_hold(domain_search(word))
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Normalize pictograms
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if len(re.compile('[a-zA-Z0-9]').sub('', word))>2:
            for pict in pictograms_to_emoji:
                if (pict in word) and (len(pict)>2):
                    temp_dict[word] = word.replace(pict, pictograms_to_emoji[pict])
                elif pict==word:  
                    temp_dict[word] = pictograms_to_emoji[pict]
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Isolate emoji
    # Global
    chars = ''.join([c for c in global_chars_list if c in emoji.UNICODE_EMOJI])
    chars_dict = {ord(c):f' {c} ' for c in chars}
    data = data.apply(lambda x: make_cleaning(x,chars_dict))

    # Duplicated dots, question marks and exclamations
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if (Counter(word)['.']>1) or (Counter(word)['!']>1) or (Counter(word)['?']>1) or (Counter(word)[',']>1):
            if (Counter(word)['.']>1):
                new_word = re.sub('\.\.+', ' . . . ', new_word)
            if (Counter(word)['!']>1):
                new_word = re.sub('\!\!+', ' ! ! ! ', new_word)
            if (Counter(word)['?']>1):
                new_word = re.sub('\?\?+', ' ? ? ? ', new_word)
            if (Counter(word)[',']>1):
                new_word = re.sub('\,\,+', ' , , , ', new_word)
            temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Remove underscore for spam words
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if (len(re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word))/len(word) > 0.6) and ('_' in word):
            temp_dict[word] = re.sub('_', '', word)       
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Isolate spam chars repetition
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if (len(re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word))/len(word) > 0.6) and (len(Counter(word))==1) and (len(word)>2):
            temp_dict[word] = ' '.join([' ' + next(iter(Counter(word).keys())) + ' ' for i in range(3)])
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Normalize pictograms part 2
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if len(re.compile('[a-zA-Z0-9]').sub('', word))>1:
            for pict in pictograms_to_emoji:
                if pict==word:  
                    temp_dict[word] = pictograms_to_emoji[pict]
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Isolate brakets and quotes
    chars = '()[]{}<>"'
    chars_dict = {ord(c):f' {c} ' for c in chars}
    data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

    # Break short words
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (len(k)<=20) and ('/' in k)]
    temp_dict = {}
    for word in temp_vocab:
        temp_dict[word] = re.sub('/', ' / ', word)
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Break long words
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (len(k)>20)]
    temp_dict = {}
    for word in temp_vocab:
        if '_' in word:
            temp_dict[word] = re.sub('_', ' ', word)
        elif '/' in word:
            temp_dict[word] = re.sub('/', ' / ', word)
        elif len(' '.join(word.split('-')).split())>2:
            temp_dict[word] = re.sub('-', ' ', word)
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    
    # Remove/Convert usernames and hashtags (add username/hashtag word?????)
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if (len(word) > 3) and (word[1:len(word)-1].isalnum()) and (not re.compile('[#@,.:;]').sub('', word).isnumeric()):
            if word[len(word)-1].isalnum():
                if (word.startswith('@')) or (word.startswith('#')):
                    new_word = place_hold(new_word[0] + ' ' + new_word[1:]) 
            else:
                if (word.startswith('@')) or (word.startswith('#')):
                    new_word = place_hold(new_word[0] + ' ' + new_word[1:len(word)-1]) + ' ' + word[len(word)-1]

        temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Remove ending underscore (or add quotation marks???)
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('_' in k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if word[len(word)-1]=='_':
            for i in range(len(word),0,-1):
                if word[i-1]!='_':
                    new_word = word[:i]
                    temp_dict[word] = new_word   
                    break
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    
    # Remove starting underscore 
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('_' in k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        if word[0]=='_':
            for i in range(len(word)):
                if word[i]!='_':
                    new_word = word[i:]
                    temp_dict[word] = new_word   
                    break
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
        
    # End word punctuations
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (not k[len(k)-1].isalnum())]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        for i in range(len(word),0,-1):
            if word[i-1].isalnum():
                new_word = word[:i] + ' ' + word[i:]
                break
        temp_dict[word] = new_word     
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
       
    # Start word punctuations
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (not k[0].isalnum())]
    temp_dict = {}
    for word in temp_vocab:
        new_word = word
        for i in range(len(word)):
            if word[i].isalnum():
                new_word = word[:i] + ' ' + word[i:]
                break
        temp_dict[word] = new_word     
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Find and replace acronims
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if (Counter(word)['.']>1) and (check_replace(word)):
            if (domain_search(word)!='') and (('www' in word) or (Counter(word)['/']>3)):
                temp_dict[word] = place_hold('url ' + domain_search(word))
            else: 
                if (re.compile('[\.\,]').sub('', word) in local_vocab) and (len(re.compile('[0-9\.\,\-\/\:]').sub('', word))>0):
                    temp_dict[word] =  place_hold(re.compile('[\.\,]').sub('', word))
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Apply spellchecker for contractions
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ("'" in k)]
    temp_dict = {}
    for word in temp_vocab:
        if word in helper_contractions:
            temp_dict[word] = place_hold(helper_contractions[word])
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
        
    # Isolate obscene (or just keep the word????)
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = re.compile('[a-zA-Z0-9\-\.\,\/\']').sub('', word)
        if len(Counter(new_word))>2:
            temp_dict[word] = place_hold('fuck')
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
             
    # Remove 's 
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {k:k[:-2] for k in temp_vocab if (check_replace(k)) and (k.lower()[-2:]=="'s")}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
     
    # Convert backslash
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and ('\\' in k)]    
    temp_dict = {k:re.sub('\\\\+', ' / ', k) for k in temp_vocab}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
     
    # Try remove duplicated chars (not sure about this!!!!!)
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    temp_vocab_dup = []
    for word in temp_vocab:
        temp_vocab_dup.append(''.join(ch for ch, _ in itertools.groupby(word)))
    temp_vocab_dup = set(temp_vocab_dup)
    temp_vocab_dup = temp_vocab_dup.difference(temp_vocab_dup.difference(set(local_vocab)))
    for word in temp_vocab:
        new_word = ''.join(ch for ch, _ in itertools.groupby(word))
        if new_word in temp_vocab_dup:
            temp_dict[word] = new_word
    temp_dict = {k: v for k, v in temp_dict.items() if (k != v) and (v in local_vocab)}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Isolate numbers
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if re.compile('[a-zA-Z]').sub('', word) == word:
            if re.compile('[0-9]').sub('', word) != word:
                temp_dict[word] = word
    global_chars_list = list(set([c for line in temp_dict for c in line]))
    chars = ''.join([c for c in global_chars_list if not c.isdigit()])
    chars_dict = {ord(c):f' {c} ' for c in chars}                
    temp_dict = {k:place_hold(make_cleaning(k,chars_dict)) for k in temp_dict}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    
    # Join dashes
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        temp_dict[word] = re.sub('\-\-+', '-', word)
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    
    # Try join word (Sloooow)
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k)) and (Counter(k)['-']>1)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = ''.join(['' if c in '-' else c for c in word])
        if (new_word in local_vocab) and (len(new_word)>3):
            temp_dict[word] = new_word    
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
     
    # Try Split word
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        if len(re.compile('[a-zA-Z0-9\*]').sub('', word))>0:
            chars = re.compile('[a-zA-Z0-9\*]').sub('', word)
            temp_dict[word] = ''.join([' ' + c + ' ' if c in chars else c for c in word])
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
   
    # L33T vocabulary (SLOW)
    def convert_leet(word):
        # basic conversion 
        word = re.sub('0', 'o', word)
        word = re.sub('1', 'i', word)
        word = re.sub('3', 'e', word)
        word = re.sub('\$', 's', word)
        word = re.sub('\@', 'a', word)
        return word
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if check_replace(k)]
    temp_dict = {}
    for word in temp_vocab:
        new_word = convert_leet(word)
        if (new_word!=word): 
            if (len(word)>2) and (new_word in local_vocab):
                temp_dict[word] = new_word
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    
    # Search "fuck"
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (check_replace(k))]
    temp_dict = {}
    for word in temp_vocab:
        if ('*' in word.lower()):
            if (word.lower()[0]=='n') and ('er' in word.lower()):
                temp_dict[word] = 'nigger'
            elif (('fuck' in word.lower()) or (word.lower()[0]=='f')) and ('k' in word.lower()):
                temp_dict[word] = 'fuck'
            elif (word.lower()[0]=='a') and ('le' in word.lower()):
                temp_dict[word] = 'asshole'
            elif (word.lower()[0]=='s') and (word.lower()[len(word)-1]=='t'):
                temp_dict[word] = 'shit'
            else:
                temp_dict[word] = 'fuck'   

    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
        
    # Open Holded words
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if (not check_replace(k))]
    temp_dict = {}
    for word in temp_vocab:
        temp_dict[word] = re.sub('___', ' ', word[17:-1])
    data = data.apply(lambda x: ' '.join([temp_dict.get(i, i) for i in x.split()]))
    data = data.apply(lambda x: ' '.join([i for i in x.split()]))

    # Search multiple form
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (k[-1:]=='s') and (len(k)>4)]
    temp_dict = {k:k[:-1] for k in temp_vocab if (k[:-1] in local_vocab)}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Convert emoji to text
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (k in emoji.UNICODE_EMOJI)]
    temp_dict = {}
    for word in temp_vocab:
        temp_dict[word] = re.compile('[:_]').sub(' ', emoji.UNICODE_EMOJI.get(word)) 
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Isolate Punctuation
    temp_vocab = check_vocab(data, local_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab]
    temp_dict = {}
    for word in temp_vocab:
        new_word = re.compile('[a-zA-Z0-9]').sub('', word)
        chars_dif = set(word).difference(set(word).difference(set(new_word)))
        if len(chars_dif)>0:
            temp_dict[word] = ''.join([' ' + c + ' ' if c in chars_dif else c for c in word])
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    return data
## -------------------------------------

## -------------------------------------
def mod_classic_preprocess(data):
    
    def load_helper_file(HELPER_PATH, filename):
        with open(HELPER_PATH+filename+'.pickle', 'rb') as f:
            temp_obj = pickle.load(f)
        return temp_obj    
    
    HELPER_PATH                 = '../input/stage-2-general-helpers/'
    white_list_chars            = load_helper_file(HELPER_PATH,'helper_white_list_chars')
    
    glove_vocabulary            = load_helper_file(HELPER_PATH,'helper_glove_vocabulary')
    glove_vocabulary_chars      = ''.join([c for c in glove_vocabulary if len(c) == 1])
    glove_vocabulary_chars_1    = ''.join([c for c in glove_vocabulary_chars if not c in white_list_chars])
    
    current_chars               = ''.join(set([c for line in data for c in line]))
    current_chars_1             = ''.join([c for c in current_chars if not c in white_list_chars])

    symbols_to_delete           = ''.join([c for c in current_chars_1 if not c in glove_vocabulary_chars_1])
    symbols_to_isolate          = ''.join([c for c in current_chars_1 if c in glove_vocabulary_chars_1])

    from nltk.tokenize.treebank import TreebankWordTokenizer
    tokenizer = TreebankWordTokenizer()
    
    isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
    remove_dict = {ord(c):f'' for c in symbols_to_delete}
    
    def handle_punctuation(x):
        x = x.translate(remove_dict)
        x = x.translate(isolate_dict)
        return x
    
    def handle_contractions(x):
        x = tokenizer.tokenize(x)
        return x
    
    def fix_quote(x):
        x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
        x = ' '.join(x)
        return x
    
    def preprocess(x):
        x = handle_punctuation(x)
        x = handle_contractions(x)
        x = fix_quote(x)
        return x
    
    data = data.astype(str).apply(lambda x: preprocess(x))
    return data
## -------------------------------------
 
## -------------------------------------
def mod_exp(data):
    
    def load_helper_file(HELPER_PATH, filename):
        with open(HELPER_PATH+filename+'.pickle', 'rb') as f:
            temp_obj = pickle.load(f)
        return temp_obj

    def make_cleaning(s, c_dict):
        s = s.translate(c_dict)
        return s
      
    def make_dict_cleaning(s, w_dict):
        s = w_dict.get(s, s)
        return s   

    def check_vocab(c_list, vocabulary, response='default'):
        try:
            words = set([w for line in c_list for w in line.split()])
            u_list = words.difference(set(vocabulary))
            k_list = words.difference(u_list)
        
            if response=='default':
                print('Unknown words:', len(u_list), '| Known words:', len(k_list))
            elif response=='unknown_list':
                return list(u_list)
            elif response=='known_list':
                return list(k_list)
        except:
            return []
    HELPER_PATH             = '../input/stage-2-general-helpers/'
            
    normalized_chars            = load_helper_file(HELPER_PATH,'helper_normalized_chars')
    helper_contractions         = load_helper_file(HELPER_PATH,'helper_contractions')
    crawl_vocab                 = load_helper_file(HELPER_PATH,'helper_crawl_vocabulary')
    toxic_misspell_dict         = load_helper_file(HELPER_PATH,'helper_toxic_misspell_dict')

    white_list_chars            = load_helper_file(HELPER_PATH,'helper_white_list_chars')
    
    glove_vocabulary            = load_helper_file(HELPER_PATH,'helper_glove_vocabulary')
    glove_vocabulary_chars      = ''.join([c for c in glove_vocabulary if len(c) == 1])
    glove_vocabulary_chars_1    = ''.join([c for c in glove_vocabulary_chars if not c in white_list_chars])
    
    current_chars               = ''.join(set([c for line in data for c in line]))
    current_chars_1             = ''.join([c for c in current_chars if not c in white_list_chars])

    symbols_to_delete           = ''.join([c for c in current_chars_1 if not c in glove_vocabulary_chars_1])
    symbols_to_isolate          = ''.join([c for c in current_chars_1 if c in glove_vocabulary_chars_1])

        
    data = data.astype(str)

    # Normalize chars and dots - SEE HELPER FOR DETAILS
    data = data.apply(lambda x: ' '.join([make_cleaning(i,normalized_chars) for i in x.split()]))
    data = data.apply(lambda x: re.sub('\(dot\)', '.', x))
    data = data.apply(lambda x: deaccent(x))
    global_chars_list = list(set([c for line in data for c in line]))
    chars_dict = {c:'' for c in global_chars_list if unicodedata.category(c)[0]=='C'}
    data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

    # Remove Bad Symbols
    chars_dict = {c:'' for c in symbols_to_delete}
    data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

    # Isolate brakets and quotes
    chars = '()[]{}<>"'
    chars_dict = {ord(c):f' {c} ' for c in chars}
    data = data.apply(lambda x: ' '.join([make_cleaning(i,chars_dict) for i in x.split()]))

    # Fix bad words misspell
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_dict = {}
    for word in temp_vocab:
        for w in toxic_misspell_dict:
            if w==word:
                temp_dict[word] = word.replace(w, ' ' + toxic_misspell_dict[w] + ' ')
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    
    # End word punctuations
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if not k[-1:].isalnum()]
    temp_dict = {}
    for word in temp_vocab:
        for i in range(len(word),0,-1):
            if word[i-1].isalnum():
                new_word = word[:i] + ' ' + word[i:]
                temp_dict[word] = new_word     
                break
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Start word punctuations
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if not k[0].isalnum()]
    temp_dict = {}
    for word in temp_vocab:
        for i in range(len(word)):
            if word[i].isalnum():
                new_word = word[:i] + ' ' + word[i:]
                temp_dict[word] = new_word     
                break
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Contaractions
    temp_vocab = list(set([c for line in data for c in line.split()]))
    temp_vocab = [k for k in temp_vocab if "'" in k]
    temp_dict = {}
    for word in temp_vocab:
        if word in helper_contractions:
            temp_dict[word] = helper_contractions[word]
        elif word[-2:]=="'s":
            temp_dict[word] = word[:-2]
        else:
            new_word = word
            for w in helper_contractions:
                if w in new_word:
                    new_word  = new_word.replace(w,' ' + helper_contractions[w] + ' ')
            if word!=new_word:         
                temp_dict[word] = new_word
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Try Split word
    temp_vocab = check_vocab(data, crawl_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if ('-' in k) or ('/' in k)]
    chars = '-/'
    for char in chars:
        temp_dict = {}
        for word in temp_vocab:
            if char in word:
                new_word = re.sub(char, ' ', word)
                if len(new_word)>1:
                    new_word_p_hold = re.sub(char, ' '+char+' ', word)
                    for sub_word in new_word.split():
                        if sub_word not in crawl_vocab:
                            new_word_p_hold = word 
                            break
                    temp_dict[word] = new_word_p_hold 
    
        temp_dict = {k: v for k, v in temp_dict.items() if k != v}    
        data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Try lowercase words
    temp_vocab = check_vocab(data, crawl_vocab, response='unknown_list')
    temp_dict = {}
    temp_vocab_cased = set([c.lower() for cc in temp_vocab for c in cc.split()])
    temp_vocab_cased = temp_vocab_cased.difference(temp_vocab_cased.difference(set(crawl_vocab)))
    for word in temp_vocab:
        temp_dict[word] = word.lower() if word.lower() in temp_vocab_cased else word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Try uppercase words
    temp_vocab = set(temp_vocab).difference(set(temp_dict))
    temp_dict = {}
    temp_vocab_cased = set([c.upper() for cc in temp_vocab for c in cc.split()])
    temp_vocab_cased = temp_vocab_cased.difference(temp_vocab_cased.difference(set(crawl_vocab)))
    for word in temp_vocab:
        temp_dict[word] = word.upper() if word.upper() in temp_vocab_cased else word
    temp_dict = {k: v for k, v in temp_dict.items() if k != v}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Search multiple form
    temp_vocab = check_vocab(data, crawl_vocab, response='unknown_list')
    temp_vocab = [k for k in temp_vocab if (k[-1:]=='s') and (len(k)>4)]
    temp_dict = {k:k[:-1] for k in temp_vocab if (k[:-1] in crawl_vocab)}
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))

    # Isolate chars
    chars_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
    temp_vocab = check_vocab(data, crawl_vocab, response='unknown_list')
    temp_dict = {}
    for word in temp_vocab:
        temp_dict[word] = make_cleaning(word,chars_dict)   
    data = data.apply(lambda x: ' '.join([make_dict_cleaning(i,temp_dict) for i in x.split()]))
    
    return data
## -------------------------------------

x_test_bert_cased = df_parallelize_run(test['comment_text'], mod_bert)
x_test_bert_uncased = x_test_bert_cased.str.lower()

x_test_mod_exp = df_parallelize_run(test['comment_text'], mod_exp)
x_test_classic_modified = df_parallelize_run(test['comment_text'], mod_classic_preprocess)

## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 2: Preprocess Data - Used time - {time.time()-start:.2f}s')
print('#'*20)
## ----------------------------------------------------------------------------------------------------



#################################################################################
#################################################################################
########################### LSTM
#################################################################################
#################################################################################
print('1.3. LSTM')

## -------------------------------------
LSTM_UNITS  = 128                   ## LSTM units
DHU         = 4 * LSTM_UNITS        ## LSTM Dense Hidden Units
DHHU        = 8 * LSTM_UNITS        ## Dense Hidden Units
DHHU2       = int(DHHU/16)

## -------------------------------------
LSTM_1_t    = 1
LSTM_1_T_p  = '../input/stage-2-lstm-helpers/tokenizer_lstm_1.pickle'
LSTM_1_M_p  = '../input/stage-2-lstm-helpers/embedding_matrix_lstm_1.pickle'

LSTM_1_F_p  =   [
                '/kaggle/input/s2-lstm1-p1/LSTM_mod_exp_300_fold_0',
                '/kaggle/input/s2-lstm1-p1/LSTM_mod_exp_300_fold_1',
                '/kaggle/input/s2-lstm1-p1/LSTM_mod_exp_300_fold_2',
                '/kaggle/input/s2-lstm1-p2/LSTM_mod_exp_300_fold_3',
                '/kaggle/input/s2-lstm1-p2/LSTM_mod_exp_300_fold_4',
                '/kaggle/input/s2-lstm1-p2/LSTM_mod_exp_300_fold_5',
                '/kaggle/input/s2-lstm1-p3/LSTM_mod_exp_300_fold_6',
                '/kaggle/input/s2-lstm1-p3/LSTM_mod_exp_300_fold_7',
                '/kaggle/input/s2-lstm1-p3/LSTM_mod_exp_300_fold_8',
                '/kaggle/input/s2-lstm1-p4/LSTM_mod_exp_300_fold_9',
                '/kaggle/input/s2-lstm1-p4/LSTM_mod_exp_300_fold_10',
                '/kaggle/input/s2-lstm1-p4/LSTM_mod_exp_300_fold_11',
                ]

LSTM_1_P_p  =   [
                '../input/s2-train-results/lstm1_train.csv',
                ]               
                
## -------------------------------------
LSTM_2_t    = 1
LSTM_2_T_p  = '../input/stage-2-lstm-helpers/tokenizer_lstm_2.pickle'
LSTM_2_M_p  = '../input/stage-2-lstm-helpers/embedding_matrix_lstm_2.pickle'

LSTM_2_F_p  =   [
                '/kaggle/input/s2-lstm2-p1/LSTM_classic_modified_300_fold_0',
                '/kaggle/input/s2-lstm2-p1/LSTM_classic_modified_300_fold_1',
                '/kaggle/input/s2-lstm2-p1/LSTM_classic_modified_300_fold_2',
                '/kaggle/input/s2-lstm2-p2/LSTM_classic_modified_300_fold_3',
                '/kaggle/input/s2-lstm2-p2/LSTM_classic_modified_300_fold_4',
                '/kaggle/input/s2-lstm2-p2/LSTM_classic_modified_300_fold_5',
                ]

LSTM_2_P_p  =   [
                '../input/s2-train-results/lstm2_train.csv',
                ] 
                
## -------------------------------------
LSTM_3_t    = 2
LSTM_3_T_p  = '../input/stage-2-lstm-helpers/tokenizer_lstm_1.pickle'
LSTM_3_M_p  = '../input/stage-2-lstm-helpers/embedding_matrix_lstm_1.pickle'

LSTM_3_F_p  =   [
                '/kaggle/input/s2-lstm3-p1/LSTM_mod_exp_300_fold_0',
                '/kaggle/input/s2-lstm3-p1/LSTM_mod_exp_300_fold_1',
                '/kaggle/input/s2-lstm3-p1/LSTM_mod_exp_300_fold_2',
                '/kaggle/input/s2-lstm3-p2/LSTM_mod_exp_300_fold_3',
                '/kaggle/input/s2-lstm3-p2/LSTM_mod_exp_300_fold_4',
                '/kaggle/input/s2-lstm3-p2/LSTM_mod_exp_300_fold_5',
                ]

LSTM_3_P_p  =   [
                '../input/s2-train-results/lstm3_train.csv',
                ]                 
                
## -------------------------------------

## -------------------------------------
def lstm_dummy_databanch(x_test, bs=1024, dim=9):
    x_train_torch = torch.tensor(x_test[:bs], dtype=torch.long)
    x_test_torch = torch.tensor(x_test, dtype=torch.long)
    y_train_torch = torch.tensor(np.zeros((bs, dim), dtype=np.float32), dtype=torch.float32)
    y_test_torch = torch.tensor(np.zeros((len(test), dim), dtype=np.float32), dtype=torch.float32)
    
    test_dataset = data.TensorDataset(x_test_torch, y_test_torch)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    databunch = DataBunch(train_dl=train_loader, valid_dl=train_loader, test_dl=test_loader)
    return databunch

class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, max_features):
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

class NeuralNet2(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets, max_features):
        super(NeuralNet2, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DHHU, DHHU2)
        self.linear2 = nn.Linear(DHHU, DHHU2)
        
        self.linear_out = nn.Linear(DHHU2, 1)
        self.linear_aux_out = nn.Linear(DHHU2, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, (h1,_) = self.lstm1(h_embedding)
        h_lstm2, h2 = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # print(avg_pool.shape)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        # print(max_pool.shape)
        
        h_conc = torch.cat((max_pool, avg_pool,  h1[0], h1[1], h2[0], h2[1]), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        # h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        # hidden = h_conc + h_conc_linear1 + h_conc_linear2
        hidden = h_conc_linear1
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out
## -------------------------------------

## -------------------------------------
def lstm_model_preds(x_test, token_path, matrix_path, file_names, n_type=1):
    
    with open(token_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open(matrix_path, 'rb') as f:
        embedding_matrix = pickle.load(f)

    MAX_FEATURES    = len(tokenizer.word_index) + 1
    BATCH_SIZE      = 1024
    TOTAL_DIM       = 9
    OUT_DIM         = TOTAL_DIM - 1
    AUX_DIM         = OUT_DIM - 1
    N_FOLDS         = len(file_names)
    MAX_LEN         = 300
    
    x_test_lstm = tokenizer.texts_to_sequences(x_test)
    x_test_lstm = sequence.pad_sequences(x_test_lstm, maxlen=MAX_LEN)

    databunch = lstm_dummy_databanch(x_test_lstm, bs=BATCH_SIZE, dim=TOTAL_DIM)
    test_preds = np.zeros((len(test), OUT_DIM))    

    for m_run, file_name in enumerate(file_names):
        print('Processing fold', m_run)
        seed_everything(SEED)
        
        if n_type==1:
            model = NeuralNet(embedding_matrix, AUX_DIM, MAX_FEATURES)
        else:
            model = NeuralNet2(embedding_matrix, AUX_DIM, MAX_FEATURES)
            
        learner = Learner(databunch, model, loss_func=None)
        learner.load(file_name) 
        for param in learner.model.parameters():
           param.requires_grad = False
        learner.model.eval()
   
        #test_preds += get_preds_as_nparray(learner,databunch,DatasetType.Test).astype(np.float32)
        for i, x_batch in enumerate(databunch.dl(DatasetType.Test)):
            X = x_batch[0].cuda()
            y_pred = sigmoid(learner.model(X).detach().cpu().numpy())
            test_preds[i * BATCH_SIZE:(i+1) * BATCH_SIZE, :] += y_pred
    
        learner.purge()
        del learner, model
        torch.cuda.empty_cache()
        gc.collect()        
    
    return test_preds/N_FOLDS
## -------------------------------------

## -------------------------------------
LSTM_config = [
                [x_test_mod_exp, LSTM_1_T_p, LSTM_1_M_p, LSTM_1_F_p, LSTM_1_t, LSTM_1_P_p],
                [x_test_classic_modified, LSTM_2_T_p, LSTM_2_M_p, LSTM_2_F_p, LSTM_2_t, LSTM_2_P_p],
                [x_test_mod_exp, LSTM_3_T_p, LSTM_3_M_p, LSTM_3_F_p, LSTM_3_t, LSTM_3_P_p],
              ]

for lconfig in LSTM_config: 
    LSTM_test_preds = lstm_model_preds(lconfig[0], lconfig[1], lconfig[2], lconfig[3], n_type=lconfig[4]).astype(np.float16)

    LSTM_train_preds = pd.read_csv(lconfig[5][0])
    LSTM_train_preds = train[['id']].merge(LSTM_train_preds, on='id', how='left').iloc[:,1:].values.astype(np.float16)

    try:
        second_train_df = np.concatenate((second_train_df, LSTM_train_preds), axis=1).astype(np.float16)
        second_test_df = np.concatenate((second_test_df, LSTM_test_preds), axis=1).astype(np.float16)
    except:
        second_train_df = LSTM_train_preds
        second_test_df = LSTM_test_preds
  
try:
    del LSTM_test_preds, LSTM_train_preds, LSTM_config; gc.collect()
except:
    print('ALERT ERROR')
    pass
print('#'*10)
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 3: LSTM - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------




#################################################################################
#################################################################################
########################### BERT
#################################################################################
#################################################################################
print('1.4. BERT')

## -------------------------------------
package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch, BertConfig, BertAdam
  
class FastAiBertTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len] + ["[SEP]"]
    
class BertTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)
    
class BertNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)
    
def get_bert_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    return [BertTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]
    
class BertDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)
        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)

## -------------------------------------

## -------------------------------------
def bert_model_preds(x_test, MAX_LEN=300, BATCH_SIZE=512, file_names=[], label_cols=[], bert_path=''):

    N_FOLDS = len(file_names)
    OUTPUT_DIM=len(label_cols)
    
    bert_tok = BertTokenizer.from_pretrained(bert_path)
    fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=MAX_LEN), pre_rules=[], post_rules=[])
    fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))

    bert_train = pd.DataFrame(columns=label_cols)
    bert_train['comment_text'] = x_test
    bert_train = bert_train.iloc[:BATCH_SIZE,:].fillna(0)
        
    bert_test = pd.DataFrame()
    bert_test['comment_text'] = x_test

    test_preds = np.zeros((len(x_test), OUTPUT_DIM), dtype=np.float32)

    databunch = TextDataBunch.from_df('.', bert_train, bert_train, bert_test,
                      tokenizer=fastai_tokenizer,
                      vocab=fastai_bert_vocab,
                      include_bos=False,
                      include_eos=False,
                      text_cols='comment_text',
                      label_cols=label_cols,
                      bs=BATCH_SIZE,
                      collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
                    )   
                    
    for m_run, file_name in enumerate(file_names):
        print('Processing fold', m_run)
        seed_everything(SEED)
    
        model = BertForSequenceClassification.from_pretrained(bert_path, num_labels=OUTPUT_DIM)

        learner = Learner(databunch, model, loss_func=None)         
        learner.load(file_name) 
        learner.model.eval()

        test_preds += get_preds_as_nparray(learner,databunch,DatasetType.Test).astype(np.float32)
    
        learner.purge()
        del learner, model
        torch.cuda.empty_cache()
        gc.collect()        

    return test_preds/N_FOLDS
## -------------------------------------

########################### BERT1 260 3 epochs 4 folds
## -------------------------------------
BERT_1_MODEL_PATH = '../input/bert-base-uncased'

BERT_1_F_p  =   [
                '/kaggle/input/s2-bert1-p1/bert_bert-base-uncased_mod_bert_260_fold_1',
                '/kaggle/input/s2-bert1-p2/bert_bert-base-uncased_mod_bert_260_fold_2',
                '/kaggle/input/s2-bert1-p3/bert_bert-base-uncased_mod_bert_260_fold_3',
                '/kaggle/input/s2-bert1-p4/bert_bert-base-uncased_mod_bert_260_fold_4',
                '/kaggle/input/s2-bert1-p5/bert_bert-base-uncased_mod_bert_260_fold_5',
                '/kaggle/input/s2-bert1-p6/bert_bert-base-uncased_mod_bert_260_fold_6',
                '/kaggle/input/s2-bert1-p7/bert_bert-base-uncased_mod_bert_260_fold_7',
                '/kaggle/input/s2-bert1-p8/bert_bert-base-uncased_mod_bert_260_fold_8',
                ]

BERT_1_P_p  =   [
                '../input/s2-train-results/bert1_train.csv',
                ]
## -------------------------------------

BERT_1_test_preds = bert_model_preds(x_test_bert_uncased, MAX_LEN=260, BATCH_SIZE=512, 
                                    file_names=BERT_1_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_1_MODEL_PATH)
BERT_1_test_preds = np.delete(BERT_1_test_preds, 1, 1)

BERT_1_train_preds = pd.read_csv(BERT_1_P_p[0])
del BERT_1_train_preds['var_1']

BERT_1_train_preds = train[['id']].merge(BERT_1_train_preds, on='id', how='left').iloc[:,1:].values.astype(np.float16)
BERT_1_train_preds = sigmoid(BERT_1_train_preds)

second_train_df = np.concatenate((second_train_df, BERT_1_train_preds), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, BERT_1_test_preds), axis=1).astype(np.float16)

del BERT_1_train_preds, BERT_1_test_preds
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.1: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------


########################### BERT2 120 3 epochs 4 folds
## -------------------------------------
BERT_2_MODEL_PATH = '../input/bert-base-uncased'

BERT_2_F_p  =   [
                '/kaggle/input/s2-bert2-p1/bert_bert-base-uncased_mod_bert_120_fold_1',
                '/kaggle/input/s2-bert2-p2/bert_bert-base-uncased_mod_bert_120_fold_2',
                '/kaggle/input/s2-bert2-p3/bert_bert-base-uncased_mod_bert_120_fold_3',
                '/kaggle/input/s2-bert2-p4/bert_bert-base-uncased_mod_bert_120_fold_4',
                ]

BERT_2_P_p  =   [
                '../input/s2-train-results/bert2_train.csv',
                ]
## -------------------------------------


bert_uncased_120 = make_split(x_test_bert_uncased, 120)
x_test_bert_uncased_120 = bert_uncased_120['comment_text']

BERT_2_test_preds = bert_model_preds(x_test_bert_uncased_120, MAX_LEN=120, BATCH_SIZE=512, 
                                    file_names=BERT_2_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_2_MODEL_PATH)

BERT_2_test_preds = pd.concat([bert_uncased_120[['id']], pd.DataFrame(BERT_2_test_preds, columns=['var_'+str(k) for k in range(BERT_2_test_preds.shape[-1])])],axis=1)

BERT_2_train_preds = pd.read_csv(BERT_2_P_p[0])
BERT_2_train_preds = train[['id']].merge(BERT_2_train_preds, on='id', how='left')
BERT_2_train_preds.iloc[:,1:] = sigmoid(BERT_2_train_preds.iloc[:,1:])

del BERT_2_train_preds['var_1'], BERT_2_test_preds['var_1']

agg_cols = {k:['max','sum','mean'] for k in list(BERT_2_train_preds.iloc[:,1:])}
temp_df_1 = BERT_2_train_preds.groupby('id').agg(agg_cols).reset_index()
temp_df_2 = BERT_2_test_preds.groupby('id').agg(agg_cols).reset_index()

temp_df_1 = train[['id']].merge(temp_df_1, on='id', how='left').iloc[:,1:].values.astype(np.float16)
temp_df_2 = test[['id']].merge(temp_df_2, on='id', how='left').iloc[:,1:].values.astype(np.float16)

second_train_df = np.concatenate((second_train_df, temp_df_1), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, temp_df_2), axis=1).astype(np.float16)

del temp_df_1, temp_df_2, BERT_2_test_preds, BERT_2_train_preds, bert_uncased_120, x_test_bert_uncased_120
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.2: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------


########################### BERT3 80 4 epochs 4 folds
## -------------------------------------
BERT_3_MODEL_PATH = '../input/bert-base-uncased'

BERT_3_F_p  =   [
                '/kaggle/input/s2-bert3-p1/bert_bert-base-uncased_mod_bert_80_fold_1',
                '/kaggle/input/s2-bert3-p2/bert_bert-base-uncased_mod_bert_80_fold_2',
                '/kaggle/input/s2-bert3-p3/bert_bert-base-uncased_mod_bert_80_fold_3',
                '/kaggle/input/s2-bert3-p4/bert_bert-base-uncased_mod_bert_80_fold_4',
                ]

BERT_3_P_p  =   [
                '../input/s2-train-results/bert3_train.csv',
                ]
## -------------------------------------


bert_uncased_80 = make_split(x_test_bert_uncased, 80)
x_test_bert_uncased_80 = bert_uncased_80['comment_text']

BERT_3_test_preds = bert_model_preds(x_test_bert_uncased_80, MAX_LEN=80, BATCH_SIZE=1024, 
                                    file_names=BERT_3_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_3_MODEL_PATH)

BERT_3_test_preds = pd.concat([bert_uncased_80[['id']], pd.DataFrame(BERT_3_test_preds, columns=['var_'+str(k) for k in range(BERT_3_test_preds.shape[-1])])],axis=1)

BERT_3_train_preds = pd.read_csv(BERT_3_P_p[0])
BERT_3_train_preds = train[['id']].merge(BERT_3_train_preds, on='id', how='left')
BERT_3_train_preds.iloc[:,1:] = sigmoid(BERT_3_train_preds.iloc[:,1:])

del BERT_3_train_preds['var_1'], BERT_3_test_preds['var_1']

agg_cols = {k:['max','sum','mean'] for k in list(BERT_3_train_preds.iloc[:,1:])}
temp_df_1 = BERT_3_train_preds.groupby('id').agg(agg_cols).reset_index()
temp_df_2 = BERT_3_test_preds.groupby('id').agg(agg_cols).reset_index()

temp_df_1 = train[['id']].merge(temp_df_1, on='id', how='left').iloc[:,1:].values.astype(np.float16)
temp_df_2 = test[['id']].merge(temp_df_2, on='id', how='left').iloc[:,1:].values.astype(np.float16)

second_train_df = np.concatenate((second_train_df, temp_df_1), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, temp_df_2), axis=1).astype(np.float16)

del temp_df_1, temp_df_2, BERT_3_test_preds, BERT_3_train_preds, bert_uncased_80, x_test_bert_uncased_80
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.3: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------


"""
########################### BERT4 160 4epochs 4 folds
## -------------------------------------
BERT_4_MODEL_PATH = '../input/bert-base-uncased'

BERT_4_F_p  =   [
                '/kaggle/input/s2-bert4-p1/bert_bert-base-uncased_mod_bert_160_fold_1',
                '/kaggle/input/s2-bert4-p2/bert_bert-base-uncased_mod_bert_160_fold_2',
                '/kaggle/input/s2-bert4-p3/bert_bert-base-uncased_mod_bert_160_fold_3',
                '/kaggle/input/s2-bert4-p4/bert_bert-base-uncased_mod_bert_160_fold_4',
                ]

BERT_4_P_p  =   [
                '../input/s2-train-results/bert4_train.csv',
                ]
## -------------------------------------


bert_uncased_160 = make_split(x_test_bert_uncased, 160)
x_test_bert_uncased_160 = bert_uncased_160['comment_text']

BERT_4_test_preds = bert_model_preds(x_test_bert_uncased_160, MAX_LEN=160, BATCH_SIZE=512, 
                                    file_names=BERT_4_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_4_MODEL_PATH)

BERT_4_test_preds = pd.concat([bert_uncased_160[['id']], pd.DataFrame(BERT_4_test_preds, columns=['var_'+str(k) for k in range(BERT_4_test_preds.shape[-1])])],axis=1)

BERT_4_train_preds = pd.read_csv(BERT_4_P_p[0])
BERT_4_train_preds = train[['id']].merge(BERT_4_train_preds, on='id', how='left')
BERT_4_train_preds.iloc[:,1:] = sigmoid(BERT_4_train_preds.iloc[:,1:])

del BERT_4_train_preds['var_1'], BERT_4_test_preds['var_1']

agg_cols = {k:['max','sum','mean'] for k in list(BERT_4_train_preds.iloc[:,1:])}
temp_df_1 = BERT_4_train_preds.groupby('id').agg(agg_cols).reset_index()
temp_df_2 = BERT_4_test_preds.groupby('id').agg(agg_cols).reset_index()

temp_df_1 = train[['id']].merge(temp_df_1, on='id', how='left').iloc[:,1:].values.astype(np.float16)
temp_df_2 = test[['id']].merge(temp_df_2, on='id', how='left').iloc[:,1:].values.astype(np.float16)

second_train_df = np.concatenate((second_train_df, temp_df_1), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, temp_df_2), axis=1).astype(np.float16)

del temp_df_1, temp_df_2, BERT_4_test_preds, BERT_4_train_preds, bert_uncased_160, x_test_bert_uncased_160
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.4: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------
"""

"""
########################### BERT5 260 4pochs 4 folds folds google cloud
## -------------------------------------
BERT_5_MODEL_PATH = '../input/bert-base-uncased'

BERT_5_F_p  =   [
                '/kaggle/input/s2-bert5-p1/bert_bert-base-uncased_mod_bert_260_fold_1_1',
                '/kaggle/input/s2-bert5-p1/bert_bert-base-uncased_mod_bert_260_fold_2_1',
                '/kaggle/input/s2-bert5-p2/bert_bert-base-uncased_mod_bert_260_fold_3_1',
                '/kaggle/input/s2-bert5-p2/bert_bert-base-uncased_mod_bert_260_fold_4_1',
                ]

BERT_5_P_p  =   [
                '../input/s2-train-results/bert5_train.csv',
                ]
## -------------------------------------

BERT_5_test_preds = bert_model_preds(x_test_bert_uncased, MAX_LEN=260, BATCH_SIZE=512, 
                                    file_names=BERT_5_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_5_MODEL_PATH)
BERT_5_test_preds = np.delete(BERT_5_test_preds, 1, 1)

BERT_5_train_preds = pd.read_csv(BERT_5_P_p[0])
del BERT_5_train_preds['var_1']

BERT_5_train_preds = train[['id']].merge(BERT_5_train_preds, on='id', how='left').iloc[:,1:].values.astype(np.float16)
BERT_5_train_preds = sigmoid(BERT_5_train_preds)

second_train_df = np.concatenate((second_train_df, BERT_5_train_preds), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, BERT_5_test_preds), axis=1).astype(np.float16)

del BERT_5_train_preds, BERT_5_test_preds
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.5: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------
"""

"""
########################### BERT6 260 large 1 epoch 4 folds gpu box
## -------------------------------------
BERT_6_MODEL_PATH = '../input/bert-large-uncased'

BERT_6_F_p  =   [
                '/kaggle/input/s2-bert6-p1/bert_bert-large-uncased_mod_bert_260_fold_1_1',
                '/kaggle/input/s2-bert6-p2/bert_bert-large-uncased_mod_bert_260_fold_2_1',
                '/kaggle/input/s2-bert6-p3/bert_bert-large-uncased_mod_bert_260_fold_3_1',
                #'/kaggle/input/s2-bert6-p2/bert_bert-large-uncased_mod_bert_260_fold_4_1',
                ]

BERT_6_P_p  =   [
                '../input/s2-train-results/bert6_train.csv',
                ]
## -------------------------------------

BERT_6_test_preds = bert_model_preds(x_test_bert_uncased, MAX_LEN=260, BATCH_SIZE=256, 
                                    file_names=BERT_6_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_6_MODEL_PATH)
BERT_6_test_preds = np.delete(BERT_6_test_preds, 1, 1)

BERT_6_train_preds = pd.read_csv(BERT_6_P_p[0])
del BERT_6_train_preds['var_1']

BERT_6_train_preds = train[['id']].merge(BERT_6_train_preds, on='id', how='left').iloc[:,1:].values.astype(np.float16)
BERT_6_train_preds = sigmoid(BERT_6_train_preds)

second_train_df = np.concatenate((second_train_df, BERT_6_train_preds), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, BERT_6_test_preds), axis=1).astype(np.float16)

del BERT_6_train_preds, BERT_6_test_preds
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.6: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------
"""

"""
########################### BERT7 260 Cased 3pochs 4 folds folds google cloud
## -------------------------------------
BERT_7_MODEL_PATH = '../input/bert-base-cased'

BERT_7_F_p  =   [
                '/kaggle/input/s2-bert7-p1/bert_bert-base-cased_mod_bert_260_fold_1_1',
                '/kaggle/input/s2-bert7-p1/bert_bert-base-cased_mod_bert_260_fold_2_1',
                '/kaggle/input/s2-bert7-p2/bert_bert-base-cased_mod_bert_260_fold_3_1',
                '/kaggle/input/s2-bert7-p2/bert_bert-base-cased_mod_bert_260_fold_4_1',
                ]

BERT_7_P_p  =   [
                '../input/s2-train-results/bert7_train.csv',
                ]
## -------------------------------------

BERT_7_test_preds = bert_model_preds(x_test_bert_cased, MAX_LEN=260, BATCH_SIZE=512, 
                                    file_names=BERT_7_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_7_MODEL_PATH)
BERT_7_test_preds = np.delete(BERT_7_test_preds, 1, 1)

BERT_7_train_preds = pd.read_csv(BERT_7_P_p[0])
del BERT_7_train_preds['var_1']

BERT_7_train_preds = train[['id']].merge(BERT_7_train_preds, on='id', how='left').iloc[:,1:].values.astype(np.float16)
BERT_7_train_preds = sigmoid(BERT_7_train_preds)

second_train_df = np.concatenate((second_train_df, BERT_7_train_preds), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, BERT_7_test_preds), axis=1).astype(np.float16)

del BERT_7_train_preds, BERT_7_test_preds
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.7: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------
"""


########################### BERT8 70 large 2 epochs 4 folds
## -------------------------------------
BERT_8_MODEL_PATH = '../input/bert-large-uncased'

BERT_8_F_p  =   [
                '/kaggle/input/s2-bert8-p1/bert_bert-large-uncased_mod_bert_70_fold_1',
                '/kaggle/input/s2-bert8-p2/bert_bert-large-uncased_mod_bert_70_fold_2',
                #'/kaggle/input/s2-bert8-p3/bert_bert-large-uncased_mod_bert_70_fold_3',
                #'/kaggle/input/s2-bert8-p4/bert_bert-large-uncased_mod_bert_70_fold_4',
                ]

BERT_8_P_p  =   [
                '../input/s2-train-results/bert8_train.csv',
                ]
## -------------------------------------


bert_uncased_70 = make_split(x_test_bert_uncased, 70)
x_test_bert_uncased_70 = bert_uncased_70['comment_text']

BERT_8_test_preds = bert_model_preds(x_test_bert_uncased_70, MAX_LEN=70, BATCH_SIZE=512, 
                                    file_names=BERT_8_F_p, label_cols=label_cols_set_1,
                                    bert_path=BERT_8_MODEL_PATH)

BERT_8_test_preds = pd.concat([bert_uncased_70[['id']], pd.DataFrame(BERT_8_test_preds, columns=['var_'+str(k) for k in range(BERT_8_test_preds.shape[-1])])],axis=1)

BERT_8_train_preds = pd.read_csv(BERT_8_P_p[0])
BERT_8_train_preds = train[['id']].merge(BERT_8_train_preds, on='id', how='left')
BERT_8_train_preds.iloc[:,1:] = sigmoid(BERT_8_train_preds.iloc[:,1:])

del BERT_8_train_preds['var_1'], BERT_8_test_preds['var_1']

agg_cols = {k:['max','sum','mean'] for k in list(BERT_8_train_preds.iloc[:,1:])}
temp_df_1 = BERT_8_train_preds.groupby('id').agg(agg_cols).reset_index()
temp_df_2 = BERT_8_test_preds.groupby('id').agg(agg_cols).reset_index()

temp_df_1 = train[['id']].merge(temp_df_1, on='id', how='left').iloc[:,1:].values.astype(np.float16)
temp_df_2 = test[['id']].merge(temp_df_2, on='id', how='left').iloc[:,1:].values.astype(np.float16)

second_train_df = np.concatenate((second_train_df, temp_df_1), axis=1).astype(np.float16)
second_test_df = np.concatenate((second_test_df, temp_df_2), axis=1).astype(np.float16)

del temp_df_1, temp_df_2, BERT_8_test_preds, BERT_8_train_preds, bert_uncased_70, x_test_bert_uncased_70
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4.8: BERT - Used time - {time.time()-start:.2f}s')
print(second_train_df.shape, second_test_df.shape)
print('#'*20)
## ----------------------------------------------------------------------------------------------------






## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 4: BERT - Used time - {time.time()-start:.2f}s')
print('#'*20)
## ----------------------------------------------------------------------------------------------------
   







#################################################################################
#################################################################################
########################### Stat Features
#################################################################################
#################################################################################
print('1.5 Stat Features')

## -------------------------------------
def get_text_stats(data):
    
    features = pd.DataFrame()
    data = data.astype(str)

    for char in '!?*",.@#$€':
        features['c_feat_' + char] = data.apply(lambda x: sum(v for k,v in Counter(x).items() if k==char))
   
    features['text_length'] = data.apply(lambda x: len(x))
    features['words'] = data.apply(lambda x: len(x.split()))
    
    features['super'] = data.apply(lambda x: sum(v for k, v in Counter(x).items() if k.isupper()))
    features['digit'] = data.apply(lambda x: sum(v for k, v in Counter(x).items() if k.isdigit()))

    return features  
## -------------------------------------

try:
    test_stat_features          = df_parallelize_run(test['comment_text'], get_text_stats)
    test_stat_features['year']  = np.where(test['id']>775667,1,0)
    
    train_stat_features         = pd.read_pickle('../input/stage-2-preprocessed-data/mod_stats_x_train.pkl').iloc[:,1:]
    
    second_train_df = np.concatenate((second_train_df, train_stat_features.values), axis=1).astype(np.float16)
    second_test_df = np.concatenate((second_test_df, test_stat_features.values), axis=1).astype(np.float16)
except:
    print('ALERT ERROR')
    pass
  
try:
    del train_stat_features, test_stat_features; gc.collect()
except:
    print('ALERT ERROR')
    pass
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 5: Stat Features - Used time - {time.time()-start:.2f}s')
print('#'*20)
## ----------------------------------------------------------------------------------------------------






#################################################################################
#################################################################################
########################### LGBM
#################################################################################
#################################################################################
print('1.6. LGBM')

## -------------------------------------
weights = np.ones((len(train),)) / 10
weights += (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
weights += (( (train['target'].values>=0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
weights += (( (train['target'].values<0.5).astype(bool).astype(np.int) +
   (train[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
train_y = np.where((train['target']>=0.5),1,0)
## -------------------------------------

## -------------------------------------
import lightgbm as lgb
def lgbm_model():
    
    tr_data = lgb.Dataset(tr_x, label=tr_y, weight=weights[trn_idx])
    vl_data = lgb.Dataset(vl_x, label=vl_y, weight=weights[val_idx])  
    
    lgb_params = {
                    'objective':'binary', 
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'boost_from_average':'true',
                    'n_jobs':-1,
                    'learning_rate':0.05,
                    'num_leaves': 2**6,
                    'min_data_in_leaf': 2**7,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree':0.8,
                    'subsample_freq':1,
                    'subsample':0.8,
                    'min_child_weight':12,
                    'n_estimators':2000,
                    'max_bin':250,
                    'verbose':-1,
                    'reg_lambda':0,
                    'seed':SEED,
                    'early_stopping_rounds':20
    }

    estimator = lgb.train(
        lgb_params,
        tr_data,
        valid_sets = [tr_data, vl_data],
        verbose_eval = 50,
    )        
    return estimator  
## -------------------------------------

## -------------------------------------
train_preds = train[['id']]
train_preds['target'] = train_y

test_preds = test[['id']]

oof = np.zeros(len(train))
predictions = np.zeros(len(test))
    
for fold_, (trn_idx, val_idx) in enumerate(folds.split(second_train_df, np.where((train['target']>=0.5),1,0))):
    print('Fold:',fold_)
    tr_x, tr_y = second_train_df[trn_idx], train_y[trn_idx]
    vl_x, vl_y = second_train_df[val_idx], train_y[val_idx]
    
    estimator = lgbm_model()
    tt_p = estimator.predict(vl_x)
    tt_p = (tt_p - tt_p.mean())/tt_p.std()
            
    pp_p = estimator.predict(second_test_df)
    pp_p = (pp_p - pp_p.mean())/pp_p.std()
            
    oof[val_idx] = tt_p
    predictions += pp_p/NFOLDS

train_preds['prediction']   = oof
test_preds['prediction']    = predictions

## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 6: LGBM - Used time - {time.time()-start:.2f}s')
print('#'*20)
## ----------------------------------------------------------------------------------------------------



#################################################################################
#################################################################################
########################### CV Check
#################################################################################
#################################################################################
print('1.7. CV Check')

## -------------------------------------
def fast_cv_check():
    auc_categories = pd.DataFrame()
    auc_categories['over_all'] = 1
    
    for identity in identity_columns:
        auc_categories['subgroup_'+identity] = np.where(train[identity]>=0.5, 1,0)
        
    for identity in identity_columns:
        auc_categories['bpsn_'+identity] = np.where(((train[identity]>=0.5)&(train['target']<0.5))|
                                                   ((train[identity]<0.5)&(train['target']>=0.5)), 1,0)
    for identity in identity_columns:
        auc_categories['bnsp_'+identity] = np.where(((train[identity]>=0.5)&(train['target']>=0.5))|
                                                   ((train[identity]<0.5)&(train['target']<0.5)), 1,0)
    
    labels = np.where((train['target']>=0.5),1,0)
    preds = train_preds['prediction'].values
    temp_auc_categories = auc_categories.iloc[:,1:].values
    final_auc = metrics.roc_auc_score(labels, preds)*0.25
    
    for i in range(3):
        temp_auc = []
        for col in range(9*i,(i+1)*9):
            mask = temp_auc_categories[:,col]==1
            temp_auc.append(metrics.roc_auc_score(labels[mask], preds[mask]))
        final_auc += np.power(sum(np.power(temp_auc, -5))/9, 1/-5)*0.25
    
    print('CV AUC check:', final_auc)
## -------------------------------------

try:
    fast_cv_check()
except:
    print('ALERT ERROR')
    pass
## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 7: CV Check - Used time - {time.time()-start:.2f}s')
print('#'*20)
## ----------------------------------------------------------------------------------------------------



#################################################################################
#################################################################################
########################### Export
#################################################################################
#################################################################################
print('1.8. Export')

test_preds[['id','prediction']].to_csv('submission.csv', index=False)

## ----------------------------------------------------------------------------------------------------
print(f'DONE: Step 8: Export - Used time - {time.time()-start:.2f}s')
print('#'*20)
## ----------------------------------------------------------------------------------------------------
