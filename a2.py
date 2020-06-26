import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#chosen training algorithms:
#from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#other additional imports:
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, normalize
from nltk.corpus import stopwords

random.seed(42)


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[1])
    print("Data shape: ", X.shape)
    return X


def extract_features(samples):
    #print(samples[0], sample[1])
    print("Extracting features ...")
    dict_postindex_word = {}
    corpus=[]
    stop_words = set(stopwords.words('english'))
    #articles = os.listdir(folder1)+os.listdir(folder2) 
    #folders=[folder1, folder2]

    for sample in samples:
        words = []
        # all words in sample tokenized and saved in list:
        words += [word.lower() for word in sample.split() if (word.isalpha() and word not in stop_words)]
        ## filter out unique words and their frequency in the sample:
        uniqueWords, wordCount=get_unique(words)
           
        ## save frequent words and their count to dictionary:
        #article_plus_classname = article + '_' + folder
        #sample_index = "Doc_" + str(samples.index(sample))
        #for index, count in enumerate(uniqueFrequentWordCount):
        #    if sample_index in dict_postindex_word:
        #        dict_postindex_word[sample_index][uniqueFrequentWords[index]]=count
        #    else: 
        #        dict_postindex_word[sample_index]={}
        #        dict_postindex_word[sample_index][uniqueFrequentWords[index]]=count
        
        sample_index = "Doc_" + str(samples.index(sample))
        for index, count in enumerate(wordCount):
            if sample_index in dict_postindex_word:
                dict_postindex_word[sample_index][uniqueWords[index]]=count
            else: 
                dict_postindex_word[sample_index]={}
                dict_postindex_word[sample_index][uniqueWords[index]]=count
       
    ## filter out those words which show up less than n times:
    #    uniqueFrequentWords, uniqueFrequentWordCount=select_frequent_words(uniqueWords, wordCount, 0)
    #    corpus += [word for word in uniqueFrequentWords]
                
    # extract class name from the file extention the files were previously given:
    #k = [k.partition("_")[2] for k,v in dict_postindex_word.items()]
    #print("Trying dict_postindex_word: ", dict_postindex_word.get('Doc_1')) 
    
    #fill out NaN cells with 0's:
    df = pd.DataFrame(dict_postindex_word).fillna(0) 
    #transpose the dateframe so that x-axis becomes y-axis and vice versa: 
    df_transposed = df.T
    #turn df into numpy array:
    df_as_nparray = df_transposed.to_numpy()
    
    freq_sums_of_nparray = np.sum(df_as_nparray, axis =0)
    filtered_nparray = freq_sums_of_nparray > 10
    filtered_df_as_nparray = df_as_nparray[:, filtered_nparray]
    
    #df_transposed = df.T
    
    #add a column 'class_name' to the dataframe:  
    #df_transposed.insert(0,'class_name', k, True)
   
    
    return filtered_df_as_nparray

    #pass #Fill this in
    #return features
    

def get_unique(x):
    y, f = np.unique(x, return_counts=True)
    return y, f

def select_frequent_words(words, counts, n):
    more_than_n_times = []
    remaining_counts = []
    for index, count in enumerate(counts):
        if count > n:
                more_than_n_times.append(words[index])
                remaining_counts.append(count)
    return more_than_n_times, remaining_counts

##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    #scaler = MinMaxScaler()
    #scaled_X = scaler.fit_transform(X)
    #normalized_X = normalize(scaled_X, norm='l1', axis=1, copy=True)
    #X_dr = reduce_dim(normalized_X, n=n_dim)
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    svd = TruncatedSVD(n_components=n)
    X_transformed = svd.fit_transform(X)
    return X_transformed

##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = KNeighborsClassifier(n_neighbors=3) # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf = DecisionTreeClassifier() # <--- REPLACE THIS WITH A SKLEARN MODEL
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)

# Fill in this:
def shuffle_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test 

# Fill in this:
def train_classifer(clf, X, y):
    assert is_classifier(clf)
    return clf.fit(X,y)
    
# Fill in this:
def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    predicted_labels_for_X = clf.predict(X)
    accuracy = accuracy_score(y, predicted_labels_for_X)
    precision = precision_score(y, predicted_labels_for_X, average='weighted')
    recall = recall_score(y, predicted_labels_for_X, average='weighted')
    f_measure = f1_score(y, predicted_labels_for_X, average='weighted')
    #print("Accuracy:", accuracy_score(y, y_pred, normalize=True))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F-measure:", f_measure)


######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )
