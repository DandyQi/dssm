import pandas as pd
import gzip
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
import string
from nltk.util import ngrams
def parse(path):
	g = gzip.open(path, 'rb')
	for l in g:
		yield eval(l)

def getDF(path):
	i = 0
	df = {}
	for d in parse(path):
		df[i] = d
		i += 1
		# if i>4:
		# 	break
	return pd.DataFrame.from_dict(df, orient='index')

def make_user_item_text(df_user,df_item):
	df_user_text_1 = pd.DataFrame({'reviewerID':df_user['reviewerID'],'text':df_user['reviewText'] +' '+ df_user['summary']})
	df_item_text = pd.DataFrame({'asin':df_item['asin'],'text':df_item['description'] +' '+ df_item['title']})
	user_list = df_user_text_1.reviewerID.values
	user_list = list(set(user_list))	#9022
	df_user_text = pd.DataFrame(columns=('reviewerID','totalText'))
	i = 0
	for user in user_list:
		df_temp = df_user_text_1[df_user_text_1['reviewerID']==user]['text'].values
		df_user_text.loc[i] = [user,''.join(df_temp)]
		i += 1
	return df_user_text,df_item_text

def make_text_pair(df_user_text,df_item_text,df_u):
	df_pair = pd.DataFrame(columns=('reviewerID','userText','itemText'))
	user_list = df_u.reviewerID.values
	for i in range(len(user_list)):	#9022
		df_pair.loc[i] = [user_list[i],
						df_user_text[df_user_text['reviewerID']==user_list[i]].iloc[0,1],
						df_item_text[df_item_text['asin']==df_u.iloc[i,1]].iloc[0,1]]
	return df_pair#.insert(1,'asin',df_u['asin'].values)

def deleteDuplicatedElementFromList3(listA):
    return sorted(set(listA), key=listA.index)

def word_to_ngrams(word,ngram_size=3,lower=True):
    """Returns a list of all n-gram possibilities of the given word."""
    if lower:
        word = word.lower()
    word = '<' + word + '>'
    return list(map(lambda x: ''.join(x), list(ngrams(word,ngram_size))))

def letter3gramtable(corpus):
    l3g = []
    i = 0
    for line in corpus:
        if i % 100 ==0:
            print('Generating line:%d' % i)
        remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
        #去除标点符号
        line = line.translate(remove_punct_map)
        for word in line.split():
            #print(word)
            w2n = word_to_ngrams(word)
            l3g.extend(w2n)
        i += 1
    print('Before remove duplication,length:',len(l3g))
    l3g = deleteDuplicatedElementFromList3(l3g)
    print('Removed duplication,length:',len(l3g))
    return l3g

def letter3grams(corpus):
    l3g = letter3gramtable(corpus)
    feature = np.zeros((len(corpus),len(l3g)))
    i = 0
    for line in corpus:
        if i % 100 == 0:
            print('Transforming lines:%d' %i)
        remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
        #去除标点符号
        line = line.translate(remove_punct_map)
        line_l3g_count = np.zeros(len(l3g), dtype='int')
        for word in line.split():
            word_l3g_count = np.zeros(len(l3g), dtype='int')
            w3g = word_to_ngrams(word)
            for j in w3g:
                # if j in l3g:
                    idx = l3g.index(j)
                    word_l3g_count[idx] += 1

            # print (V1)
            line_l3g_count += word_l3g_count
        
        feature[i] = line_l3g_count
        i += 1
    # print('Writing file: feature.h5')
    # file = h5py.File('./Cross-Domain_data/feature.h5','w')
    # file.create_dataset('A',data = feature)
    # file.close()
    return feature

if __name__ == "__main__":
	raw_pair_on = False

	pd.set_option('display.max_columns', None)
	
	if raw_pair_on:
		df_i = getDF('./datasets/meta_Musical_Instruments.json.gz')
		df_u = getDF('./datasets/reviews_Musical_Instruments_5.json.gz')
		df_u = df_u[df_u['overall']>3.0]
		df_i = df_i.where(df_i.notnull(), '')
		df_u = df_u.where(df_u.notnull(), '')
		# print(df_i.head())
		# print(df_u.head())
		df_user_text,df_item_text = make_user_item_text(df_u,df_i)
		df_pair = make_text_pair(df_user_text,df_item_text,df_u)
		df_pair=shuffle(df_pair)
		with open('./user_item_raw_pair.pkl','wb') as f:
			pickle.dump(df_pair.values,f)
	else:
		with open('./user_item_raw_pair.pkl','rb') as f:
			df_pair = pd.DataFrame(pickle.load(f),columns = ('reviewerID','userText','itemText'))
	corpus_u = df_pair['userText'].values
	corpus_i = df_pair['itemText'].values
	#TF-IDF
	# vectorizer_u = TfidfVectorizer()
	# vectorizer_i = TfidfVectorizer()
	# feature_u = vectorizer_u.fit_transform(corpus_u)
	# feature_i = vectorizer_i.fit_transform(corpus_i)
	#letter tri-gram
	feature_u = letter3grams(corpus_u)
	feature_i = letter3grams(corpus_i)
	with open('./feature/train_u.pkl','wb') as f:
		train_u = feature_u[2100:]
		pickle.dump(train_u,f)
	with open('./feature/test_u.pkl','wb') as f:
		test_u = feature_u[:2100]
		pickle.dump(test_u,f)
	with open('./feature/train_i.pkl','wb') as f:
		train_i = feature_i[2100:]
		pickle.dump(train_i,f)
	with open('./feature/test_i.pkl','wb') as f:
		test_i = feature_i[:2100]
		pickle.dump(test_i,f)