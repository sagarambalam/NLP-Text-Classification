import pickle
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import wordninja

import pandas as pd


def load_model(model_path):
    '''
    Input: string
    Output: Model object
    '''
    return pickle.load(open(model_path, 'rb'))

def split_joint_text(tokens):
    '''
    This function is for splitting words like 'wickedmachine' to ['wicked', 'machine']
    '''
    lm = wordninja.LanguageModel('text.txt.gz')
    
    for each_token in tokens:
        temp_list = lm.split(each_token)
        temp_list = list(filter(lambda x: len(x)>1,temp_list))
        if len(temp_list)> 1:
            tokens.remove(each_token)
            tokens.extend(temp_list)
    
    return tokens

def preprocess(input_sentence):
    
    tokenizer_obj= TreebankWordTokenizer()
    stemmer_obj = PorterStemmer()
    
    ## Removing punctuations and special characters
    input_sentence = input_sentence.lower()
    clean_text = re.sub(r"[^a-z0-9]", ' ', input_sentence)
    
    ## Tokenizing text
    tokens = tokenizer_obj.tokenize(clean_text)
    tokens = [stemmer_obj.stem(x) for x in tokens]
    
    ## Splitting joint words (For eg. partsfor, forcaptive)
    tokens = split_joint_text(tokens)
    
    ## Removing all tokens of length less than 3
    stop_words = stopwords.words("english")
    clean_tokens = [word for word in tokens if word not in stop_words]
    filtered_tokens = list(filter(lambda x: len(x)>2, clean_tokens))
    processed_sentence = ' '.join(filtered_tokens)
    
    return processed_sentence
    

def predict(input_string):

    response = {'success': False, 'label':None}

    if len(input_string) == 0:
        response = {'success': False, 'label':None}
    else:
        model = load_model('model.pickle')
        clean_string = preprocess(input_string)
        if len(clean_string)>3:
            series_obj = pd.Series(clean_string)
            label = model.predict(series_obj)[0]
            response = {'success': True, 'label':label}
    
    return response

if __name__=='__main__':
    predict(input_string)
    


