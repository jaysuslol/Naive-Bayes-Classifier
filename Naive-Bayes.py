import os
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def run(user_options):
    inp = str.split(user_options, sep=' ') ## [0]: dataset | [1]: option
    dataset = os.getcwd() + '\\' + inp[0]

    stop_words = set(stopwords.words('english'))
    documents = []
    classes = []

    if (inp[1]):
        clf = MultinomialNB()
        vectorizer = CountVectorizer(binary=False)
    elif (inp[1] == 2):
        clf = BernoulliNB()
        vectorizer = CountVectorizer(binary=True)
    else:
        print("Unrecognized option specified")
        exit(-1)

    try:
        if (inp[2] is not None):
            K_FOLD_CROSS_VALIDATION = int(inp[2])
    except IndexError:
        K_FOLD_CROSS_VALIDATION = 5 ## K-fold cross validation default
    
    print("\nCross validation: %d" %K_FOLD_CROSS_VALIDATION + "-fold\n\n")
    for root, dirs, files in os.walk(dataset):
        currClass = os.path.split(root)[1]

        for f in files:
            with open(root + '\\' + f) as data:
                datatext = data.read()
                word_tokens = word_tokenize(datatext)
                filtered_sentence = ' '.join(w for w in word_tokens if not w.lower() in stop_words) ## Lower case conversion
                documents.append(filtered_sentence)
                classes.append(currClass)
            data.close()
    
    #print(documents)
    documents = numpy.array(documents)
    classes = numpy.array(classes)
    #print(documents.shape)
    #print(classes.shape)
    vectors = vectorizer.fit_transform(documents)
    clf.fit(vectors, classes)

    score = cross_val_score(clf, vectors, classes, cv=K_FOLD_CROSS_VALIDATION)
    if (inp[1]):
        print("Model used: Multinomial Naive-Bayes (Non-Binary Term Occurences)\n")
    elif (inp[1] == 2):
        print("Model used: Bernoulli Naive-Bayes Binary Term Occurences)\n")

    print("Results:")
    print(score)
    
    ### Uncomment below code to check files individually (input)
    """
    while(True):
        fname = input("Training done. Enter filename to classify: ")
        datatext = ''

        for root, dirs, files in os.walk(dataset):
            for f in files:
                if (f == fname):
                    with open(root + '\\' + f) as file:
                        datatext = file.read()
                    file.close()

        vectors_test = vectorizer.transform([datatext])
        pred = clf.predict(vectors_test)
        print(pred)
    """     

def main():
    # !!! DURING FIRST TIME USAGE, UNCOMMENT THE FOLLOWING LINES, RUN THE SCRIPT ONCE AND RECOMMENT THEM
    #nltk.download('punkt')
    #nltk.download('stopwords')
    #exit(1)

    os.system('cls')
    print("\nUsage: DATASET_FILENAME OPTION K_FOLD_CROSS_VALIDATION (default: 5)\n\n"
          + "OPTIONS:\n1: Multinomial (Term Occurences)\n2: Bernoulli (Binary Term Occurrences)\n\n" 
          + "Make sure that the dataset is in the same folder as the python script.\n")
    userinp = input("Input: ")
    run(userinp)

if __name__ == "__main__":
    main()