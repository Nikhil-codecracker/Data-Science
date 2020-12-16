from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import sys


tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


def getStemmedReview(reviews):
    reviews = reviews.lower()
    reviews = reviews.replace("<br/><br />"," ")
    
    tokens = tokenizer.tokenize(reviews)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review


def getStemmedDocument(inputfile,outputfile):
    
    out = open(outputfile,'w',encoding="utf8")
    
    with open(inputFile,encoding="utf8") as f:
        reviews = f.readlines()
    
    for review in reviews:
        cleaned_review = getStemmedReview(review)
        print((cleaned_review),file=out)
        
    out.close()
    
inputFile = sys.argv[1]
outputFile = sys.argv[2]
getStemmedDocument(inputFile,outputFile)