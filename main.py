# This is a sample Python script.
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import json
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk import tokenize
from collections import Counter
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams

def read_data():
    #FIXME Change this to Train
    paths = ["restaurants/test/", "restaurants/train/"]
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
                    train[file] = data_dict
                else:
                    test[file] = data_dict

                all_files[file] = data_dict
    return all_files, train, test, reviews


def prep_all_data(all_files):
    review_tokens = []
    for file in all_files.keys():
        current_file = all_files[file]
        df = current_file["Reviews"]
        df = prep_data(df)
        review_tokens.append(df['Unigrams'].values)
        current_file["Reviews"] = df
    return all_files, flatten_review_tokens(review_tokens)

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
    return ["NUM" if word.isnumeric() else word for word in tokens]
    #return " ".join(bwords)
def remove_punct(review_content):
    return review_content.translate(str.maketrans("", "", string.punctuation))



def flatten_review_tokens(review_tokens):
    token_by_review = [item for sublist in review_tokens for item in sublist]
    raw_tokenized = [item.lower() for sublist in token_by_review for item in sublist]
    sw = pd.read_csv("STOPWORDS_txt.csv", header = None).to_numpy()
    with_out_stopwords = [word for word in raw_tokenized if word not in sw and word not in string.punctuation]
    return with_out_stopwords

def get_bigrams_list(df : pd.DataFrame):
    return [item for sublist in df["bigrams"] for item in sublist]

def get_unigrams_list(df:pd.DataFrame):
    return [item for sublist in df["Unigrams"] for item in sublist]

def get_document_tokens(df: pd.DataFrame):
    freq = get_unigrams_list(df)
    freq.extend(get_bigrams_list(df))
    return set(freq)

def get_freq_data( train_data:dict, counts: pd.DataFrame):
    doc_freq = dict.fromkeys(counts["Word"] , 0)
    for cfile in train_data:
        doc_tokens = get_document_tokens(train_data[cfile]["Reviews"])
        for token in doc_tokens:
            doc_freq[token] +=1

    return doc_freq

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
def plot_word_frequency(word_counts : pd.DataFrame, save = False):
    fig , ax = plt.subplots(figsize=(12 , 8))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Ranking")
    plt.ylabel("TTF")
    # Plot horizontal bar graph
    plt.scatter(x = word_counts["Rank"], y = word_counts["TTF"])
    ax.set_title("Common Words Found")
    if save:
        plt.savefig("ZIPF_Law_Curve")
    else:
        plt.show()







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #nltk.download('punkt') #FIXME This must be donwloaded for everything to work properly
    #nltk.download('stopwords')

    all_files, train, test ,reviews = read_data()
    #PARTA DEVLIVERABLES
    preped_all_data, all_review_tokens = prep_all_data(all_files)
    preped_train_data, train_tokens = prep_all_data(train)
    c_dict = preped_all_data['3OLZOlqgOXdqY0uwxcOTfw.json']
    df = c_dict["Reviews"]

    #PARTB DELIVERABLES
    counts = get_word_frequency(all_review_tokens)
    #plot_word_frequency(counts, save = True)
    #plot_word_frequency(counts, save = False)

    #PARTC Deliverables
    train_tokens.extend(generate_bigrams(preped_train_data))
    train_counts = get_word_frequency(train_tokens)



    print("Finished")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
