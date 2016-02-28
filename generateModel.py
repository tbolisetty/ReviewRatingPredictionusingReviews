# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
#import loadData
import numpy as np
import re
import nltk
import gensim
from nltk.corpus import stopwords
import nltk.data
import pandas as pd
import copy
import gensim.models
import logging
import glob
#nltk.download()

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = review
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)
# Download the punkt tokenizer for sentence splitting




# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    #print raw_sentences
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences
 
def train_model(trainData):
 # Load the punkt tokenizer   
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #form sentences for all reviews
    sentences=[]
    for i in range(len(trainData)):
        #print i
        #if i==11443:
            #print trainData[i]
        sentences+=review_to_sentences(trainData[i],tokenizer)
        #sentences+=review_to_sentences("Had to chance to do great things.Second.\
    
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
        
    # Initialize and train the model (this will take some time)
     
    print "Training model..."
    
    ## Set values for various parameters
    num_features = 7    # Word vector dimensionality                      
    min_word_count = 20   # Minimum word count                        
    num_workers = 2       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-3   # Downsample setting for frequent words
    
    
    model = gensim.models.Word2Vec(sentences, min_count=1,size=100,window=10)
    
    #model = gensim.models.Word2Vec(sentences, workers=num_workers, \
    #            size=num_features, min_count = min_word_count, \
    #            window = context, sample = downsampling)
    ##
    
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "first model"
    model.save(model_name)
    print(model.syn0.shape)
    return model
    
    #trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
    
    #model.most_similar(positive=['how'],negative=['good'],)
    
    ##model = gensim.models.Word2Vec(sentences, min_count=1)

def find_feature_vec(model,review):
    num_features=model.syn0.shape[1]
    #for each word output the vector
    index2word_set = set(model.index2word)
    
    #loop over the reviews and generate the vector for each review
    # featureVec holds the vector for each review
    
    #review=trainData[]
    featureVec = np.zeros((num_features,),dtype="float32")
    review=review_to_wordlist(review, True)
    nwords=0
    if not review:
        return "empty"
        
    for word in review:
            if word in index2word_set: 
                wordFile.write(word +',')
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])
    #print featureVec
    featureVec = np.divide(featureVec,nwords)
    
    #print review
    #print featureVec
    return featureVec


def finding_feature(start,stop):
	print "generating featureVec..."
    for i in range(0,len(reviewData)):
        featureVec=find_feature_vec(model,reviewData[i])
        if not reviewData[i] or len(featureVec<100):
            f=emptyf
            fv=''
        else:
            f=fvec
            fv=','.join(['%.5f' % num for num in featureVec])
            fv=fv+','  
        f.write(trainData['user_id'][i] +',')
        f.write(trainData['business_id'][i]  +',')
        f.write(fv)
        f.write(str(trainData['rating'][i] )+ '\n')
        if i>0 and i%500==0:
            print 'completed ',i, 'iterations'
    
    #featureVecData['review'][i]=featureVec

#featureVecData[:5]

# generates feature vectors for each review and write in output file
def generate_features(model,fvec,emptyf,reviewData):
    print 'start of gene'   
    for i in range(0,len(reviewData)):
        featureVec=find_feature_vec(model,reviewData[i])
        if not reviewData[i] or len(featureVec)<100:
            f=emptyf
            fv=''
        else:
            f=fvec
            fv=','.join(['%.5f' % num for num in featureVec])
            fv=fv+','    
        f.write(trainData['user_id'][i] +',')
        f.write(trainData['business_id'][i]  +',')
        f.write(fv)
        f.write(str(trainData['rating'][i] )+ '\n')
        if i>0 and i%500==0:
            print 'completed ',i, 'iterations'
    del fv
    
   

def generate_models():
#creates a header for storing the header info of the files
	header_name=['user_id','business_id','review','rating','date']
	path =r'/media/tej/Windows8_OS/D/BigData/Project/reviews' # use your path
	allFiles = glob.glob(path + "/*.txt")
	wordFile = open('D:\\BigData\\Project\\reviews\\featureOut1.txt', 'w')

	frame = pd.DataFrame()
	list_ = []
	length=[]
	header_name=['user_id','business_id','review','rating','date']
	#print allFiles
	#create data frame using all review files
	for reviewfile in allFiles:
		df=pd.read_csv(reviewfile,names=header_name,sep='|',na_filter=False)
		length.append(len(df))
		list_.append(df)
	frame = pd.concat(list_,ignore_index=True)
	reviewData=frame['review'] # get all the reviews from data frame
	#featureVecData=copy.deepcopy(trainData)
	model=train_model(reviewData) # train model with different model characteritics
	model_name='D:\\BigData\\Project\\7features20min10context'
	model.save(model_name) # save the model for later use
	model=gensim.models.Word2Vec.load(model_name)
	
	
	
   
def main():   
	
	generate_models() # calls generate_models to generate different models according to different characteritics
	#generate_feature_vec(reviewData)
	fvec = open('D:\\BigData\\Project\\reviews\\featureOut1.txt', 'w')
	emptyf = open('D:\\BigData\\Project\\reviews\\featureEmptyOut1.txt', 'w')

	fvec = open('D:\\BigData\\Project\\reviews\\featureOut1.txt', 'w')
	emptyf = open('D:\\BigData\\Project\\reviews\\featureEmptyOut1.txt', 'w')
	
	
	# generating feature vectors

	#modelPath='/media/tej/Windows8_OS/D/BigData/Project/models'
	modelPath =r'C:\D\BigData\Project\250ModelFeatures'
	models=glob.glob(modelPath + "/*")
	header_name=['user_id','business_id','review','rating','date']
	#trainData=pd.read_csv('D:\\BigData\\Project\\Newoutput-0.txt',header=None,sep='|')
	#path='/media/tej/Windows8_OS/D/BigData/Project/reviews/output-0.txt'
	#path =r'C:/D/BigData/Project/r/output-0.txt' # use your path
	path =r'C:/D/BigData/Project/reviews/output-1.txt' # use your path
	print path
	trainData=pd.read_csv(path,names=header_name,sep='|',na_filter=False)
	reviewData=trainData['review']

	for modelName in models: #generating features according to each model 
		print modelName
		if modelName.endswith('.npy') or modelName.endswith('.txt'):
			continue
		
		fve=str(path) + 'featureOut1.txt'
		empt=str(path) + 'featureEmptyOut.txt'
		fvec = open(fve, 'w')
		emptyf = open(empt, 'w')
		print 'start'
		model=gensim.models.Word2Vec.load(modelName) # loads the model which was generated
		generate_features(model,fvec,emptyf,reviewData) # generates feature vector using the model for the input reviews
		print 'completed model'
		fvec.flush()
		fvec.close()
		emptyf.flush()
		emptyf.close()		

main()