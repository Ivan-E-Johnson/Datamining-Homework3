# This is a sample Python script.
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import json
import pickle
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk import tokenize
from collections import Counter
from ast import literal_eval as make_tuple
import csv
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from copy import deepcopy
def read_data():
    #FIXME Change this to Train
    paths = ["restaurants/test/", "restaurants/train/"]
    #paths = ["restaurants/Debug_test/", "restaurants/Debug_train/"]
    data_dict = {}
    all_files = {}
    reviews = {}
    train = {}
    test = {}
    for path in paths:
        for file in os.listdir(path):
            with open(path + file) as project_file:
                data = json.load(project_file)
                data_dict["Reviews"] =  pd.DataFrame.from_dict(data["Reviews"])
                reviews[file] = pd.DataFrame.from_dict(data["Reviews"])
                data_dict["Resturaunt"] = data["RestaurantInfo"]
                if("train" in path):
                    train[file] = deepcopy(data_dict)
                else:
                    test[file] = deepcopy(data_dict)

                all_files[file] = deepcopy(data_dict)
    return all_files, train, test, reviews


def prep_all_data(all_files):
    unigram_tokens = []
    bigram_tokens = []
    token_freq = []
    for file in all_files.keys():
        current_file = all_files[file]
        df = current_file["Reviews"]
        df = prep_data(df)
        unigram_tokens.extend(get_unigrams_list(df))
        bigram_tokens.extend(get_bigrams_list(df))
        all_files[file]["Reviews"] = df
    return all_files, unigram_tokens, bigram_tokens

#This is were changes should be made
def prep_data(df):
    df["Content"] = df["Content"].apply(remove_punct)
    df['Unigrams'] = df["Content"].apply(nltk.wordpunct_tokenize)
    df['Unigrams'] = df['Unigrams'].apply(stemm_tokens)
    df['Unigrams'] = df['Unigrams'].apply(remove_stopwords)
    df['Unigrams'] = df['Unigrams'].apply(remove_nums)
    df['bigrams'] = df['Unigrams'].apply(calc_bigrams)

    return df
#FIXME This may need to be changed to use his methodoligy??
def calc_bigrams(tokens):
    return list(ngrams(tokens, 2))
def stemm_tokens(tokenized):
    ps = PorterStemmer()
    return [ps.stem(w) for w in tokenized]

def remove_stopwords(tokens):
    sw = pd.read_csv("STOPWORDS_txt.csv" , header=None).to_numpy()
    return[word for word in tokens if word not in sw]
def remove_nums(tokens):
    #words = review_content.split()
    #bwords = ["NUM" if word.isnumeric() else word for word in tokens]
    return ["num" if word.isnumeric() else word for word in tokens]
    #return " ".join(bwords)
def remove_punct(review_content):
    return review_content.translate(str.maketrans("", "", string.punctuation))



'''
def flatten_review_tokens(review_tokens):
    token_by_review = [item for sublist in review_tokens for item in sublist]
    raw_tokenized = [item.lower() for sublist in token_by_review for item in sublist]
    sw = pd.read_csv("STOPWORDS_txt.csv", header = None).to_numpy()
    with_out_stopwords = [word for word in raw_tokenized if word not in sw and word not in string.punctuation]
    return with_out_stopwords
'''
def get_bigrams_list(df : pd.DataFrame):
    return [item for sublist in df["bigrams"] for item in sublist]

def get_unigrams_list(df:pd.DataFrame):
    return [item for sublist in df["Unigrams"] for item in sublist]

def get_document_tokens(df: pd.DataFrame):
    freq = get_unigrams_list(df)
    freq.extend(get_bigrams_list(df))
    return Counter(freq).keys()

def get_freq_data( train_data:dict, counts: pd.DataFrame):
    doc_freq = dict.fromkeys(counts["Word"] , 0)
    for cfile in train_data:
        #doc_tokens = get_document_tokens(train_data[cfile]["Reviews"])
        df= train_data[cfile]["Reviews"]
        for row in df.itertuples():
            for token in set(row.bigrams):
                if token in doc_freq:
                    # FIXME Only Add Once per doc
                    doc_freq[token] += 1
            for token in set(row.Unigrams):
                if token in doc_freq:
                    # FIXME Unsure why this must be done but it breaks if you dont
                    doc_freq[token] += 1
        #for token in doc_tokens:

#            else:
#                doc_freq[str(token)] = 1
    return doc_freq


def update_counts(counts: pd.DataFrame, df: pd.DataFrame):
    updated = pd.merge(right = df,left = counts, on = "Word")
    updated.drop("DF_x", axis=1, inplace = True)
    updated.rename(columns={"DF_y" : "Doc_Freq"}, inplace=True)
    updated = updated[updated["Doc_Freq"]] > 50
    return updated

def generate_bigrams(train_data : dict):
    bigrams = []
    for cfile in train_data:
        bigrams.extend(get_bigrams_list(train_data[cfile]["Reviews"]))
    #bi_count = Counter(bigrams)
    return bigrams

'''
# Depreicaited
def token(df):
     df['Unigrams'] = df.apply(lambda row: nltk.sent_tokenize(row['Content']) , axis=1)
     return df

'''

def get_word_frequency(flat_token_list : list):
    counts = Counter(flat_token_list)
    words = pd.DataFrame(counts.items() , columns=["Word" , "TTF"])
    words.sort_values(by="TTF" , inplace=True , ascending=False)
    words["Rank"] = words["TTF"].rank(method='dense' , ascending=False)
    return words
def plot_word_frequency(word_counts : pd.DataFrame, save = False, title = '' ):
    fig , ax = plt.subplots(figsize=(12 , 8))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Ranking")
    plt.ylabel("TTF")
    # Plot horizontal bar graph
    plt.scatter(x = word_counts["Rank"], y = word_counts["TTF"])
    ax.set_title("Common Words Found")
    if save:
        plt.savefig(title+"ZIPF_Law_Curve")
    else:
        plt.show()


def reload_tokens(file_dict : dict):
    uni_tokens = []
    bi_tokens = []

    for file in file_dict:
        file_dict[file]["Reviews"]["bigrams"] = file_dict[file]["Reviews"]["bigrams"].apply(make_tuple)
        file_dict[file]["Reviews"]["Unigrams"] = file_dict[file]["Reviews"]["Unigrams"].apply(make_tuple)

        bi_tokens.extend(get_bigrams_list(file_dict[file]["Reviews"]))
        uni_tokens.extend(get_unigrams_list(file_dict[file]["Reviews"]))
    return  uni_tokens,bi_tokens
def load_data():
    test_path = "Load_Data/Test/"
    train_path ="Load_Data/Train/"
    test = {}
    train = {}
    c_dict = {}
    for file in os.listdir(test_path):
        if "csv" in file:
            c_dict["Reviews"] = pd.read_csv(test_path + file,  lineterminator='\n')
            with open(test_path + "Resturaunt/" + file[:-4], 'r') as fp:
                c_dict["Resturaunt"] = json.load(fp)
            test[file[:-4]] = deepcopy(c_dict)
    for file in os.listdir(train_path):
        if "csv" in file:
            c_dict["Reviews"] = pd.read_csv(train_path + file,  lineterminator='\n')
            with open(train_path + "Resturaunt/" + file[:-4] , 'r') as fp:
                c_dict["Resturaunt"] = json.load(fp)
            train[file[:-4]] = deepcopy(c_dict)
    return train, test

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #nltk.download('punkt') #FIXME This must be donwloaded for everything to work properly
    #nltk.download('stopwords')
    load = True

    if not load:
        all_files, train, test ,reviews = read_data()
        #PARTA DEVLIVERABLES
        #preped_all_data, all_review_tokens = prep_all_data(all_files)
        prep_test_data, test_uni_tokens, test_bi_tokens = prep_all_data(test)
        prep_train_data, train_uni_tokens, train_bi_tokens = prep_all_data(train)
    else:
        prep_train_data , prep_test_data = load_data()
        test_uni_tokens , test_bi_tokens = reload_tokens(prep_test_data)
        train_uni_tokens , train_bi_tokens = reload_tokens(prep_train_data)
    test_tokens = []
    test_tokens.extend(test_uni_tokens)
    test_tokens.extend(test_bi_tokens)
    train_tokens = []
    train_tokens.extend(train_uni_tokens)
    train_tokens.extend(train_bi_tokens)

    c_dict = prep_train_data['9IRdWhDNo2T6vyMLwrQdMw.json']
    df = c_dict["Reviews"]
    all_review_tokens =[]
    all_review_tokens.extend(train_uni_tokens)
    all_review_tokens.extend(test_uni_tokens)
    #PARTB DELIVERABLES
    counts = get_word_frequency(all_review_tokens) #Without bigrams
    plot_word_frequency(counts, save = True, title= "Uni_")
    #plot_word_frequency(counts, save = False)
    all_review_tokens.extend(test_bi_tokens)
    counts = get_word_frequency(all_review_tokens)  # With bigrams
    plot_word_frequency(counts , save=True, title = "All_")



    #PARTC Deliverables
    train_counts = get_word_frequency(train_tokens)
    doc_freqency = pd.DataFrame(get_freq_data(prep_train_data , counts).items() , columns=["Word" , "DF"])
    counts = update_counts(counts, doc_freqency)
    print("Top 50 NGrams")
    print(counts.sort_values(["Doc_Freq", "TTF"], ascending = False).head(50))
    print("Bottom 50 NGrams")
    print(counts.sort_values(["Doc_Freq" , "TTF"]).head(50))
    print("Finished")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
