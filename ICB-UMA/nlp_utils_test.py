### Module containing auxiliary functions and classes for NLP using BERT


## Load text

import os

def load_text_files(file_names, path):
    """
    It loads the text contained in a set of files into a returned list of strings.
    Code adapted from https://stackoverflow.com/questions/33912773/python-read-txt-files-into-a-dataframe
    """
    output = []
    for f in file_names:
        with open(path + f, "r") as file:
            output.append(file.read())
            
    return output


def load_ss_files(file_names, path):
    """
    It loads the start-end pair of each splitted sentence from a set of files (start + \t + end line format expected) into a 
    returned dictionary, where keys are file names and values a list of tuples conatining the start-end pairs of the 
    splitted sentences.
    """
    output = dict()
    for f in file_names:
        with open(path + f, "r") as file:
            f_key = f.split('.')[0]
            output[f_key] = []
            for sent in file:
                output[f_key].append(tuple(map(int, sent.strip().split('\t'))))
            
    return output


## Keras BERT Tokenizer

# Our aim is to use the same tokenizer the Keras BERT library applies before performing WordPiece sub-tokenization.
# For that reason, the next code is adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py

import unicodedata

def is_punctuation(ch):
    code = ord(ch)
    return 33 <= code <= 47 or \
        58 <= code <= 64 or \
        91 <= code <= 96 or \
        123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')

def is_cjk_character(ch):
        code = ord(ch)
        return 0x4E00 <= code <= 0x9FFF or \
            0x3400 <= code <= 0x4DBF or \
            0x20000 <= code <= 0x2A6DF or \
            0x2A700 <= code <= 0x2B73F or \
            0x2B740 <= code <= 0x2B81F or \
            0x2B820 <= code <= 0x2CEAF or \
            0xF900 <= code <= 0xFAFF or \
            0x2F800 <= code <= 0x2FA1F

def is_space(ch):
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
        unicodedata.category(ch) == 'Zs'

def is_control(ch):
    return unicodedata.category(ch) in ('Cc', 'Cf')


# Improve NER fragment generation using sentence-splitting (SS)

def word_piece_tokenize(word, word_pos, token_dict):
    """
    word_pos: list containing the start position of each of the characters forming the word
    Code taken from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L121
    """
    
    if word in token_dict:
        return [word], [(word_pos[0], word_pos[-1]+1)]
    sub_tokens, start_end_sub = [], []
    start, stop = 0, 0
    while start < len(word):
        stop = len(word)
        while stop > start:
            sub = word[start:stop]
            if start > 0:
                sub = '##' + sub
            if sub in token_dict:
                break
            stop -= 1
        if start == stop:
            # When len(sub) = 1 and sub is not in token_dict (unk sub-token)
            stop += 1
        sub_tokens.append(sub)
        # Following brat standoff format (https://brat.nlplab.org/standoff.html), end position
        # is the first character position after the considered sub-token
        start_end_sub.append((word_pos[start], word_pos[stop-1]+1))
        start = stop
    return sub_tokens, start_end_sub


def start_end_tokenize(text, token_dict, start_i=0, cased=True):
    """
    Our aim is to produce both a list of sub-tokens and a list of tuples containing the start and
    end char positions of each sub-token.
    
    start_i: the start position of the first character in the text.
    
    Code adapted from: https://github.com/CyberZHG/keras-bert/blob/master/keras_bert/tokenizer.py#L101
    """
    
    if not cased:
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        text = text.lower()
    spaced = ''
    # Store the start positions of each considered character (ch), such that
    # sum([len(word) for word in spaced.strip().split()]) = len(start_arr)
    start_arr = [] 
    for ch in text:
        if is_punctuation(ch) or is_cjk_character(ch):
            spaced += ' ' + ch + ' '
            start_arr.append(start_i)
        elif is_space(ch):
            spaced += ' '
        elif not(ord(ch) == 0 or ord(ch) == 0xfffd or is_control(ch)):
            spaced += ch
            start_arr.append(start_i)
        # If it is a control char we skip it but take its offset into account
        start_i += 1
    
    tokens, start_end_arr = [], []
    i = 0
    for word in spaced.strip().split():
        j = i + len(word)
        sub_tokens, start_end_sub = word_piece_tokenize(word, start_arr[i:j], token_dict)
        tokens += sub_tokens
        start_end_arr += start_end_sub
        i = j
        
    return tokens, start_end_arr


import numpy as np
import pandas as pd

def process_ner_labels(df_ann):
    df_res = []
    for i in range(df_ann.shape[0]):
        ann_i = df_ann.iloc[i].values
        # Separate discontinuous locations and split each location into start and end offset
        ann_loc_i = ann_i[4]
        for loc in ann_loc_i.split(';'):
            split_loc = loc.split(' ')
            df_res.append(np.concatenate((ann_i[:4], [int(split_loc[0]), int(split_loc[1])])))

    return pd.DataFrame(np.array(df_res), 
                        columns=list(df_ann.columns[:-1]) + ["start", "end"]).drop_duplicates()


from math import ceil

def process_brat_labels(brat_files):
    """
    brat_files: list containing the path of the annotations files in BRAT format (.ann).
    
    Check the annotations contained in the files are in BRAT format with codes assigned.
    """
    
    df_res = []
    for file in brat_files:
        with open(file) as ann_file:
            doc_name = file.split('/')[-1].split('.')[0]
            i = 0
            for line in ann_file:
                i += 1
                line_split = line.strip().split('\t')
                assert len(line_split) == 3
                if i % 2 > 0:
                    # BRAT annotation
                    assert line_split[0] == "T" + str(ceil(i/2))
                    text_ref = line_split[2]
                    location = ' '.join(line_split[1].split(' ')[1:]).split(';')
                else:
                    # Code assignment
                    assert line_split[0] == "#" + str(ceil(i/2))
                    code = line_split[2]
                    for loc in location:
                        split_loc = loc.split(' ')
                        df_res.append([doc_name, code, text_ref, int(split_loc[0]), int(split_loc[1])])

    return pd.DataFrame(df_res, 
                        columns=["doc_id", "code", "text_ref", "start", "end"])


def convert_token_to_id_segment(token_list, tokenizer, seq_len):
    """
    Given a list of tokens representing a sentence, and a tokenizer, it returns their correponding lists of 
    indices and segments. Padding is added as appropriate.
    
    Code adapted from https://github.com/CyberZHG/keras-bert/tree/master/keras_bert/tokenizer.py#L72
    """
    
    # Add [CLS] and [SEP] tokens (second_len = 0)
    tokens, first_len, second_len = tokenizer._pack(token_list, None)
    # Generate idices and segments
    token_ids = tokenizer._convert_tokens_to_ids(tokens)
    segment_ids = [0] * first_len + [1] * second_len
    
    # Padding
    pad_len = seq_len - first_len - second_len
    token_ids += [tokenizer._pad_index] * pad_len
    segment_ids += [0] * pad_len

    return token_ids, segment_ids


def ss_start_end_tokenize(ss_start_end, max_seq_len, text, token_dict, cased=True):
    """
    ss_start_end: list of tuples, where each tuple contains the start-end character positions pair of 
                  the splitted sentences from the input document text.
    text: document text.
    
    return: two lists of lists, the first for the sub-tokens sentences and the second for the 
            sub-tokens start-end pairs.
    """
    out_sub_token, out_start_end = [], []
    for ss_start, ss_end in ss_start_end:
        sub_token, start_end = start_end_tokenize(text=text[ss_start:ss_end], token_dict=token_dict, 
                                                  start_i=ss_start, cased=cased)
        assert len(sub_token) == len(start_end)
        # Re-split large sub-tokens splitted sentences
        for i in range(0, len(sub_token), max_seq_len):
            out_sub_token.append(sub_token[i:i+max_seq_len])
            out_start_end.append(start_end[i:i+max_seq_len])
    
    return out_sub_token, out_start_end


def ss_fragment_greedy(ss_token, ss_start_end, max_seq_len):
    """
    ss_token and ss_start_end: list of lists of sub-tokenized sentences.
    
    return: list of lists representing the obtained sub-tokens fragments.
    """
    frag_token, frag_start_end = [[]], [[]]
    i = 0
    while i < len(ss_token):
        assert len(ss_token[i]) <= max_seq_len
        if len(frag_token[-1]) + len(ss_token[i]) > max_seq_len:
            # Fragment is full, so create a new empty fragment
            frag_token.append([])
            frag_start_end.append([])
            
        frag_token[-1].extend(ss_token[i])
        frag_start_end[-1].extend(ss_start_end[i])
        i += 1
          
    return frag_token, frag_start_end
        

## Brute force approach to annotate each fragment

from tqdm import tqdm

def ss_brute_force_create_frag_input_data(df_text, text_col, df_ann, doc_list, ss_dict, tokenizer, lab_encoder, seq_len, cased=True):
    """
    ss_dict: dict where keys are file names and values a list of tuples containing the start-end pairs of the 
    splitted sentences in each file.
    
    Temporal complexity: O(n_doc x n_frag x n_ann), where n_frag and n_ann vary for each doc.
    """
    indices, segments, labels, n_fragments, start_end_offsets = [], [], [], [], []
    for doc in tqdm(doc_list):
        # Extract doc annotation
        doc_ann = df_ann[df_ann["doc_id"] == doc]
        # Extract doc text
        doc_text = df_text[df_text["doc_id"] == doc][text_col].values[0]
        # Perform SS of doc text
        doc_ss = ss_dict[doc] # SS start-end pairs of the doc
        doc_ss_token, doc_ss_start_end = ss_start_end_tokenize(ss_start_end=doc_ss, 
                                max_seq_len=seq_len-2, text=doc_text, 
                                token_dict=tokenizer._token_dict, cased=cased)
        assert len(doc_ss_token) == len(doc_ss_start_end)
        # Split the list of sub-tokens sentences into fragments using greedy strategy
        frag_token, frag_start_end = ss_fragment_greedy(ss_token=doc_ss_token, 
                           ss_start_end=doc_ss_start_end, max_seq_len=seq_len-2)
        assert len(frag_token) == len(frag_start_end)
        # Store the start-end char positions of all the fragments
        start_end_offsets.extend(frag_start_end)
        # Store the number of fragments of each doc text
        n_fragments.append(len(frag_token))
        # Assign to each fragment the labels (codes) from the NER-annotations exclusively occurring inside 
        # the fragment
        for f_token, f_start_end in zip(frag_token, frag_start_end):
            # fragment length is assumed to be <= SEQ_LEN-2
            assert len(f_token) == len(f_start_end) <= seq_len-2
            # Indices & Segments
            frag_id, frag_seg = convert_token_to_id_segment(f_token, tokenizer, seq_len)
            indices.append(frag_id)
            segments.append(frag_seg)
            # Labels
            frag_labels = []
            # start-end char positions of the whole fragment, i.e. the start position of the first
            # sub-token and the end position of the last sub-token
            frag_start, frag_end = f_start_end[0][0], f_start_end[-1][1]
            for j in range(doc_ann.shape[0]):
                doc_ann_cur = doc_ann.iloc[j] # current annotation
                # Add the annotations whose text references are contained within the fragment
                if doc_ann_cur['start'] < frag_end and doc_ann_cur['end'] > frag_start:
                    frag_labels.append(doc_ann_cur['code'])
            labels.append(frag_labels)
    
    # start_end_offsets is returned for further sanity checking purposes only
    return np.array(indices), np.array(segments), lab_encoder.transform(labels), np.array(n_fragments), start_end_offsets


# MAP score evaluation

def max_fragment_prediction(y_frag_pred, n_fragments, label_encoder_classes, doc_list, exc_code=None):
    """
    Convert fragment-level to doc-level predictions, usin max criterion.
    """
    y_pred = []
    i_frag = 0
    for n_frag in n_fragments:
        y_pred.append(y_frag_pred[i_frag:i_frag+n_frag].max(axis=0))
        i_frag += n_frag
    return prob_codiesp_prediction_format(np.array(y_pred), label_encoder_classes, doc_list, exc_code)


def prob_codiesp_prediction_format(y_pred, label_encoder_classes, doc_list, exc_code=None):
    """
    Given a matrix of predicted probabilities (m_docs x n_codes), for each document, this procedure stores all the
    codes sorted according to their probability values in descending order. Finally, predictions are saved in a dataframe
    defined following CodiEsp submission format (see https://temu.bsc.es/codiesp/index.php/2020/02/06/submission/).
    
    exc_code can contain a code (string) that is not considered for evaluation.
    """
    
    # Sanity check
    assert y_pred.shape[0] == len(doc_list)
    
    pred_doc, pred_code, pred_rank = [], [], []
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        # Codes are sorted according to their probability values in descending order
        codes_sort = [label_encoder_classes[j] for j in np.argsort(pred)[::-1]]
        pred_code += codes_sort
        pred_doc += [doc_list[i]]*len(codes_sort)
        # For compatibility with format_predictions function
        pred_rank += list(range(1, len(codes_sort)+1))
            
    # Save predictions in CodiEsp submission format
    result = pd.DataFrame({"doc_id": pred_doc, "code": pred_code, "rank": pred_rank})
    if exc_code is not None: 
        result = result[result["code"] != exc_code]
    return result


# Code adapted from: https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_coding.py

from trectools import TrecQrel, TrecRun, TrecEval

    
def format_predictions(pred, output_path, valid_codes, 
                       system_name = 'xx', pred_names = ['query','docid', 'rank']):
    '''
    DESCRIPTION: Add extra columns to Predictions table to match 
    trectools library standards.
        
    INPUT: 
        pred: pd.DataFrame
                Predictions.
        output_path: str
            route to TSV where intermediate file is stored
        valid_codes: set
            set of valid codes of this subtask

    OUTPUT: 
        stores TSV files with columns  with columns ['query', "q0", 'docid', 'rank', 'score', 'system']
    
    Note: Dataframe headers chosen to match library standards.
          More informative INPUT headers would be: 
          ["clinical case","code"]

    https://github.com/joaopalotti/trectools#file-formats
    '''
    # Rename columns
    pred.columns = pred_names
    
    # Not needed to: Check if predictions are empty, as all codes sorted by prob, prob-thr etc., are returned
    
    # Add columns needed for the library to properly import the dataframe
    pred['q0'] = 'Q0'
    pred['score'] = float(10) 
    pred['system'] = system_name 
    
    # Reorder and rename columns
    pred = pred[['query', "q0", 'docid', 'rank', 'score', 'system']]
    
    # Not needed to Lowercase codes
    
    # Not needed to: Remove codes predicted twice in the same clinical case
    
    # Not needed to: Remove codes predicted but not in list of valid codes
    
    # Write dataframe to Run file
    pred.to_csv(output_path, index=False, header=None, sep = '\t')


def compute_map(valid_codes, pred, gs_out_path=None):
    """
    Custom function to compute MAP evaluation metric. 
    Code adapted from https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_coding.py
    """
    
    # Input args default values
    if gs_out_path is None: gs_out_path = './intermediate_gs_file.txt' 
    
    pred_out_path = './intermediate_predictions_file.txt'
    ###### 2. Format predictions as TrecRun format: ######
    format_predictions(pred, pred_out_path, valid_codes)
    
    
    ###### 3. Calculate MAP ######
    # Load GS from qrel file
    qrels = TrecQrel(gs_out_path)

    # Load pred from run file
    run = TrecRun(pred_out_path)

    # Calculate MAP
    te = TrecEval(run, qrels)
    MAP = te.get_map(trec_eval=False) # With this option False, rank order is taken from the given document order
    
    ###### 4. Return results ######
    return MAP


# Code copied from: https://github.com/TeMU-BSC/cantemist-evaluation-library/blob/master/src/cantemist_coding.py

def format_gs(filepath, output_path=None, gs_names = ['qid', 'docno']):
    '''
    DESCRIPTION: Load Gold Standard table.
    
    INPUT: 
        filepath: str
            route to TSV file with Gold Standard.
        output_path: str
            route to TSV where intermediate file is stored
    
    OUTPUT: 
        stores TSV files with columns ["query", "q0", "docid", "rel"].
    
    Note: Dataframe headers chosen to match library standards. 
          More informative headers for the INPUT would be: 
          ["clinical case","label","code","relevance"]
    
    # https://github.com/joaopalotti/trectools#file-formats
    '''
    # Input args default values
    if output_path is None: output_path = './intermediate_gs_file.txt' 
    
    # Check GS format:
    check = pd.read_csv(filepath, sep='\t', header = 0, nrows=1)
    if check.shape[1] != 2:
        raise ImportError('The GS file does not have 2 columns. Then, it was not imported')
    
    # Import GS
    gs = pd.read_csv(filepath, sep='\t', header = 0, names = gs_names)  
        
    # Preprocessing
    gs["q0"] = str(0) # column with all zeros (q0) # Columnn needed for the library to properly import the dataframe
    gs["rel"] = str(1) # column indicating the relevance of the code (in GS, all codes are relevant)
    gs.docno = gs.docno.str.lower() # Lowercase codes
    gs = gs[['qid', 'q0', 'docno', 'rel']]
    
    # Remove codes predicted twice in the same clinical case 
    # (they are present in GS because one code may have several references)
    gs = gs.drop_duplicates(subset=['qid','docno'],  
                            keep='first')  # Keep first of the predictions

    # Write dataframe to Qrel file
    gs.to_csv(output_path, index=False, header=None, sep=' ')


from keras.callbacks import Callback

class EarlyMAP_Frag(Callback):
    """
    Custom callback that performs early-stopping strategy monitoring MAP-prob metric on validation fragment dataset.
    Both train and validation MAP-prob values are reported in each epoch.
    """
    
    def __init__(self, x_train, x_val, frag_train, frag_val, label_encoder_cls, valid_codes, train_doc_list, val_doc_list, 
                 train_gs_file=None, val_gs_file=None, patience=10):
        self.X_train = x_train
        self.X_val = x_val
        self.frag_train = frag_train
        self.frag_val = frag_val
        self.label_encoder_cls = label_encoder_cls
        self.valid_codes = valid_codes
        self.train_doc_list = train_doc_list
        self.val_doc_list = val_doc_list
        self.train_gs_file = train_gs_file
        self.val_gs_file = val_gs_file
        self.patience = patience
    
    
    def on_train_begin(self, logs=None):
        self.best = 0.0
        self.wait = 0
        self.best_weights = None


    def on_epoch_end(self, epoch, logs={}):
        # Metrics reporting
        ## MAP-prob
        ### Train data
        y_pred_train = self.model.predict(self.X_train)
        # Save predictions file in CodiEsp format
        df_pred_train = max_fragment_prediction(y_frag_pred=y_pred_train, n_fragments=self.frag_train, 
                                                label_encoder_classes=self.label_encoder_cls, 
                                                doc_list=self.train_doc_list)
        map_train = compute_map(valid_codes=self.valid_codes, pred=df_pred_train, gs_out_path=self.train_gs_file)
        logs['map'] = map_train
        
        ### Val data
        y_pred_val = self.model.predict(self.X_val)
        # Save predictions file in CodiEsp format
        df_pred_val = max_fragment_prediction(y_frag_pred=y_pred_val, n_fragments=self.frag_val, 
                                              label_encoder_classes=self.label_encoder_cls, 
                                              doc_list=self.val_doc_list)
        map_val = compute_map(valid_codes=self.valid_codes, pred=df_pred_val, gs_out_path=self.val_gs_file)
        logs['val_map'] = map_val
        
        print('\rmap: %s - val_map: %s' % 
              (str(round(map_train,4)),
               str(round(map_val,4))),end=100*' '+'\n')
            
        
        # Early-stopping
        if (map_val > self.best):
            self.best = map_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
    
    
    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
