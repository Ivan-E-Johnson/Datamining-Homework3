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
def read_data():
    #FIXME Change this to Train
    path = "restaurants/test/"
    data_dict = {}
    all_files = {}
    reviews = {}
    for file in os.listdir(path):
        with open(path + file) as project_file:
            data = json.load(project_file)
            data_dict["Reviews"] =  pd.DataFrame.from_dict(data["Reviews"])
            reviews[file] = pd.DataFrame.from_dict(data["Reviews"])
            data_dict["Resturaunt"] = data["RestaurantInfo"]
            all_files[file] = data_dict
    return all_files, reviews


def prep_all_data(all_files):
    review_tokens = []
    for file in all_files.keys():
        current_file = all_files[file]
        df = current_file["Reviews"]
        df = prep_data(df)
        review_tokens.append(df['tokenized_review'].values)
        current_file["Reviews"] = df
    return all_files, flatten_review_tokens(review_tokens)

#This is were changes should be made
def prep_data(df):
    df["Content"] = df["Content"].apply(remove_punct)
    df['tokenized_review'] = df["Content"].apply(nltk.wordpunct_tokenize)
    df['tokenized_review'] = df['tokenized_review'].apply(stemm_tokens)
    df['tokenized_review'] = df['tokenized_review'].apply(remove_nums)
    return df

def stemm_tokens(tokenized):
    ps = PorterStemmer()
    return [ps.stem(w) for w in tokenized]
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
    with_out_punct = []
    with_out_stopwords = [word for word in raw_tokenized if word not in sw and word not in string.punctuation]
    return with_out_stopwords

'''
# Depreicaited
def token(df):
     df['tokenized_review'] = df.apply(lambda row: nltk.sent_tokenize(row['Content']) , axis=1)
     return df

'''






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #nltk.download('punkt') #FIXME This must be donwloaded for everything to work properly
    #nltk.download('stopwords')

    all_files, reviews = read_data()
    preped_data, review_tokens = prep_all_data(all_files)
    c_dict = preped_data['3OLZOlqgOXdqY0uwxcOTfw.json']
    df = c_dict["Reviews"]
    counts = Counter(review_tokens)
    # Instantiates a frequency dictionary





    print("Finished")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
