# This is a sample Python script.
import math
import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk import tokenize
from collections import Counter
from collections import OrderedDict
from ast import literal_eval as make_tuple
from sklearn.preprocessing import OneHotEncoder
import csv
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.util import ngrams
from copy import deepcopy
import numpy as np
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

def load_Query():
	with open("QUERY.JSON") as project_file:
		data = json.load(project_file)
		df =prep_data(pd.DataFrame.from_dict(data["Reviews"]))
		df = vectorize_prep(df)
		tf = []
		for row in df.itertuples():
			tf_dict = {}
			for token in row.Counts.keys():
				tf_dict[token] = row.Counts[token] / row.Doc_len
			tf.append(tf_dict)
		df["TF"] = tf
		return df

def load_data():
	test_path = "Load_Data/Test/"
	train_path ="Load_Data/Train/"
	test = {}
	train = {}
	c_dict = {}
	for file in os.listdir(test_path):
		if "csv" in file:
			c_dict["Reviews"] = pd.read_csv(test_path + file,  lineterminator='\n')
			c_dict["Reviews"].drop("Unnamed: 0" , axis = 1, inplace = True)
			c_dict["Reviews"].dropna(inplace=True)
			if "bigrams\r" in c_dict["Reviews"].columns:
				c_dict["Reviews"].rename(columns={"bigrams\r": "bigrams"} , inplace=True)
			with open(test_path + "Resturaunt/" + file[:-4], 'r') as fp:
				c_dict["Resturaunt"] = json.load(fp)
			test[file[:-4]] = deepcopy(c_dict)
	for file in os.listdir(train_path):
		if "csv" in file:
			c_dict["Reviews"] = pd.read_csv(train_path + file,  lineterminator='\n')
			c_dict["Reviews"].drop("Unnamed: 0" , axis=1 , inplace=True)
			if "bigrams\r" in c_dict["Reviews"].columns:
				c_dict["Reviews"].rename(columns={"bigrams\r": "bigrams"} , inplace=True)

			with open(train_path + "Resturaunt/" + file[:-4] , 'r') as fp:
				c_dict["Resturaunt"] = json.load(fp)
			train[file[:-4]] = deepcopy(c_dict)
	return train, test

def reload_tokens(file_dict : dict):
	uni_tokens = []
	bi_tokens = []

	for file in file_dict:
		file_dict[file]["Reviews"]["bigrams"] = file_dict[file]["Reviews"]["bigrams"].apply(make_tuple)
		file_dict[file]["Reviews"]["Unigrams"] = file_dict[file]["Reviews"]["Unigrams"].apply(make_tuple)

		bi_tokens.extend(get_bigrams_list(file_dict[file]["Reviews"]))
		uni_tokens.extend(get_unigrams_list(file_dict[file]["Reviews"]))
	return  uni_tokens,bi_tokens
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

def calc_doc_len(content:str):
	return(len(content.split()))

def get_bigrams_list(df : pd.DataFrame):
	return [item for sublist in df["bigrams"] for item in sublist]

def get_unigrams_list(df:pd.DataFrame):
	return [item for sublist in df["Unigrams"] for item in sublist]

def get_document_tokens(df: pd.DataFrame):
	freq = get_unigrams_list(df)
	freq.extend(get_bigrams_list(df))
	return Counter(freq).keys()

def get_counts(train_data: dict, filter= 50):
	num_Docs = 0
	doc_freq = {}
	TTF_freq = {}
	for cfile in train_data:
		# doc_tokens = get_document_tokens(train_data[cfile]["Reviews"])
		df = train_data[cfile]["Reviews"]
		num_Docs += len(df)
		for row in df.itertuples():
			b = Counter(row.bigrams)
			u = Counter(row.Unigrams)
			for token in b:
				if token in doc_freq:
					doc_freq[token] +=1
				else:
					doc_freq[token] = 1
				if token in TTF_freq:
					TTF_freq[token] += b.get(token)
				else:
					TTF_freq[token] = b.get(token)
			for token in u:
				if token in doc_freq:
					doc_freq[token] +=1
				else:
					doc_freq[token] = 1
				if token in TTF_freq:
					TTF_freq[token] += u.get(token)
				else:
					TTF_freq[token] = u.get(token)

	df1 = pd.DataFrame(doc_freq.items() , columns=["Word" , "Doc_Freq"])
	df2 = pd.DataFrame(TTF_freq.items() , columns=["Word" , "TTF"])
	df3 = pd.concat([df1,df2], axis=1)
	df3.columns = ["Word" , "Doc_Freq" , "DROP" , "TTF"]
	df3.drop("DROP", axis = 1, inplace = True)
	#df3 = df3[df3["Doc_Freq"] > filter]
	df3.sort_values(by="TTF" , inplace=True , ascending=False)
	df3["Rank"] = df3["TTF"].rank(method='dense' , ascending=False)
	df3["IDF"] = (1 + np.log( num_Docs/ df3["Doc_Freq"] ))
	df3.reset_index(drop=True, inplace= True)
	return df3

def get_word_dict(df: pd.DataFrame):
	return dict((v , k) for k , v in df["Word"].items())

def generate_bigrams(train_data : dict):
	bigrams = []
	for cfile in train_data:
		bigrams.extend(get_bigrams_list(train_data[cfile]["Reviews"]))
	#bi_count = Counter(bigrams)
	return bigrams


#####################################

def get_freq_data( train_data:dict, counts: pd.DataFrame):
	doc_freq = dict.fromkeys(counts["Word"] , 0)
	for cfile in train_data:

		#doc_tokens = get_document_tokens(train_data[cfile]["Reviews"])
		df= train_data[cfile]["Reviews"]
		for row in df.itertuples():
			for token in set(row.bigrams):
				if token in doc_freq:
					# FIXME Only Add Once per doc
					#print(token)
					doc_freq[token] += 1
			for token in set(row.Unigrams):
				if token in doc_freq:
					doc_freq[token] += 1
		#for token in doc_tokens:

	#            else:
	#                doc_freq[str(token)] = 1
	return doc_freq
#DEPRECIATED

def update_counts(counts: pd.DataFrame, df: pd.DataFrame):
	updated = pd.merge(right = df,left = counts, on = "Word")
	if "Doc_Freq_x" in updated.columns:
		updated.drop("Doc_Freq_x", axis=1, inplace = True)
		updated.rename(columns={"Doc_Freq_y" : "Doc_Freq"}, inplace=True)
	updated = updated[updated["Doc_Freq"] > 50]  #This is where we filter out the bad data

	return updated


'''
# Depreicaited
def token(df):
     df['Unigrams'] = df.apply(lambda row: nltk.sent_tokenize(row['Content']) , axis=1)
     return df

'''

##############################
def get_all_data(train_dict:dict,test_dict:dict):
	all_dict = deepcopy(train_dict)
	for cfile in test_dict:
		all_dict[cfile] = test_dict[cfile]
	return train_dict

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

#############################

def calc_tf(test_data: dict, word_counts : pd.DataFrame):
	idf_dict = dict(word_counts[["Word","IDF"]].values)
	ave_idf= np.mean(list(idf_dict.values()))
	updated = {}
	for cfile in test_data:
		df = test_data[cfile]["Reviews"]
		print(cfile)
		df = vectorize_prep(df)
		tf = []
		weight = []
		for row in df.itertuples():
			tf_dict = {}
			weight_dict = {}
			for token in row.Counts.keys():
				try:
					tf_dict[token] = row.Counts[token] / row.Doc_len
					weight_dict[token] = tf_dict[token]*idf_dict[token]
				except:
					tf_dict[token] = row.Counts[token] / row.Doc_len
					idf_dict[token] = ave_idf
					weight_dict[token] =tf_dict[token] / ave_idf
			tf.append(tf_dict)
			weight.append(weight_dict)
		df["TF"] = tf
		df["Weight"] = weight
		updated[cfile] = df
	return updated , idf_dict
def vectorize_prep(df: pd.DataFrame):
	df["Ngram"] = df["Unigrams"] +df["bigrams"]
	df["Counts"] = df["Ngram"].apply(Counter)
	df["Doc_len"] = df["Content"].apply(calc_doc_len)
	return df


def vecotorize(tf_dict:dict, word_dict:dict ):
	for cfile in tf_dict.keys():
		vectors = []
		for weight in tf[cfile]["Weight"]:
			vec = to_vector_space(weight , word_dict)
			vectors.append(vec)
		tf[cfile]["Vectors"] = vectors
		tf[cfile].to_csv("Vectors/"+cfile)
	return tf

def to_vector_space(weights, words:dict):
	vec = np.empty(len(words))
	w = list(words.keys())
	for word in weights.keys():
		try:
			vec[w.index(word)] = weights[word]
		except:
			print(word)
	return vec

def load_vectors():
	vector_path = "Vectors/"
	tf = {}
	for file in os.listdir(vector_path):
		df = pd.read_csv(vector_path + file,  lineterminator='\n')
		df.drop("Unnamed: 0" , axis = 1, inplace = True)
		df.dropna(inplace=True)
		tf[file[:-4]] = df
	return tf

#######################################################
def get_unigram_prob(unigram_counter: Counter):
	unigram_prob = {}
	total = sum(unigram_counter.values())
	for uni in unigram_counter:
		unigram_prob[uni] = unigram_counter[uni]/total
	return unigram_prob
def get_bigram_prob(unigramCounts , bigramCounts):
    listOfProb = {}
    for bigram in bigramCounts.keys():
        word1 = bigram[0]
        word2 = bigram[1]
        listOfProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))
    return listOfProb
def Unigram_model(unigram_prob:dict):
	model = pd.DataFrame.from_dict(unigram_prob , orient="index").reset_index()
	model.rename(columns= {0:"Unigrams", 1: "Unigram_prob"}, inplace= True)

	return(counts["Unigrams"].sample(weights = counts["Unigram_prob"]))
def absolute_discounting_bigram(raw_unigram_prob:dict, raw_bigram_prob:dict,bigram_counts, lam = .9, delta = 0.1 , ):
	discounted_prob = {}
	for bi in raw_bigram_prob.keys():
		pass
def linear_interpolation_bigram(raw_unigram_prob:dict, raw_bigram_prob:dict, l1 = .5, l2 = .5):
	smoothed_prob = {}
	for bi in raw_bigram_prob.keys():
		smoothed_prob[bi] = (l1 * raw_unigram_prob[bi[1]]) + (l2* raw_bigram_prob[bi])
	return smoothed_prob
def get_top_words(data, n=10):
    top = sorted(data.items(), key=lambda x: x[1], reverse=True)[:n]
    return OrderedDict(top)
def Unigram_model_to_dict(unigram_model):
	return unigram_model.set_index("Unigrams").to_dict()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	#nltk.download('punkt') #FIXME This must be donwloaded for everything to work properly
	#nltk.download('stopwords')
	reload_vectors = True 		#CAN SET TO TRUE IF YOU WANT TO SPEED UP THE PROCESS
	partA_deliverables = False  #CAN SET TO FALSE IF YOU WANT TO SPEED UP THE PROCESS
	partB_deliverables = False
	partC_deliverables = False
	partD_deliverables = False
	partE_deliverables = True
	partF_deliverables = False
	if partA_deliverables:
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
	print("Finished with loading Data")

	c_dict = prep_train_data['9IRdWhDNo2T6vyMLwrQdMw.json']
	df = c_dict["Reviews"]

	if partB_deliverables:
		all_review_tokens =[]
		all_review_tokens.extend(train_uni_tokens)
		all_review_tokens.extend(test_uni_tokens)
		#PARTB DELIVERABLES
		counts = get_word_frequency(all_review_tokens) #Without bigrams
		plot_word_frequency(counts, save = True, title= "Uni_")
		#plot_word_frequency(counts, save = False)
		all_review_tokens.extend(test_bi_tokens)
		all_review_tokens.extend(train_bi_tokens)
		counts = get_word_frequency(all_review_tokens)  # With bigrams
		plot_word_frequency(counts , save=True, title = "All_")



	#PARTC Deliverables
	train_counts = get_counts(prep_train_data)
	test_counts = get_counts(prep_test_data)
	#all_data = get_all_data(prep_train_data, prep_test_data)
	#all_counts = get_counts(all_data)

	if partC_deliverables:
		print("Total number of NGrams")
		print(len(train_counts))
		print("Top 50 NGrams")
		print(train_counts[train_counts["Doc_Freq"] > 50].head(50))
		print("Bottom 50 NGrams")
		print(train_counts[train_counts["Doc_Freq"] > 50].tail(50))

	#Part D Deviverabples
	if partD_deliverables:
		tf , idf_dict = calc_tf(prep_test_data , test_counts)
		Q = load_Query()
		if reload_vectors:
			full_vectors = load_vectors()

		else:
			full_vectors = vecotorize(tf, idf_dict)


	if partE_deliverables:
		unigram_prob = get_unigram_prob(Counter(train_uni_tokens))
		bigram_prob = get_bigram_prob(Counter(train_uni_tokens) , Counter(train_bi_tokens))
		linear_smoothed_prob = linear_interpolation_bigram()

		print("Linear Interpolation Bigram Model top 10 words")
		print(get_top_words(linear_smoothed_prob, n = 10))

	if partF_deliverables:
		print("hi")
	print("Finished")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
