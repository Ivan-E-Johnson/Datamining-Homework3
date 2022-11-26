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
            #data_dict["Reviews"] = pd.DataFrame.from_dict(data["Reviews"])
            all_files[file] = data_dict
        #df = pd.read_json(path + file , lines=True , orient="columns")
        #df = pd.read_json(path + file)
    return all_files, reviews


def prep_data(all_files):
    review_tokens = []
    for file in all_files.keys():
        current_file = all_files[file]
        df = current_file["Reviews"]
        df['tokenized_review'] = df.apply(lambda row: nltk.word_tokenize(row['Content']), axis = 1)
        review_tokens.append(df['tokenized_review'].values)
        current_file["Reviews"] = df
    return all_files, flatten_review_tokens(review_tokens)




def token(df):
     df['tokenized_review'] = df.apply(lambda row: nltk.sent_tokenize(row['Content']) , axis=1)
     return df

#takes a list


def flatten_review_tokens(review_tokens):
    token_by_review = [item for sublist in review_tokens for item in sublist]
    raw_tokenized = [item.lower() for sublist in token_by_review for item in sublist]
    #sw = list(stopwords.words("english"))
    sw = pd.read_csv("STOPWORDS_txt.csv", header = None)
    with_out_stopwords =[word for word in raw_tokenized if word not in sw  & word not in string.punctuation]
    return with_out_stopwords

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    all_files, reviews = read_data()
    #nltk.download('punkt') #FIXME This must be donwloaded for everything to work properly
    #nltk.download('stopwords')
    preped_data, review_tokens = prep_data(all_files)
    c_dict = preped_data['3OLZOlqgOXdqY0uwxcOTfw.json']
    df = c_dict["Reviews"]
    # Instantiates a frequency dictionary
    fdist = FreqDist()
    print("Finished")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
