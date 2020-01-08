import re
import nltk
#nltk.download('stopwords')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

def remove(input, pattern):
    temp = re.findall(pattern, input)
    for i in temp:
        input = re.sub(i, '', input)

    return input

def Transform_Data(data):

    stop_words = stopwords.words('english')
    stop_words.remove('not')
    lemmatiser = WordNetLemmatizer()

        #remove twitter handles
    print('removing twitter handles')
    data['tidy'] = np.vectorize(remove)(data['tweet'], "@[\w]*")
    print(data.head)

    #removing short word
    data['tidy'] = data['tidy'].apply(lambda x: ' '.join([w for w in x.split() if len(w) >3]))
    print(data['tidy'].head)

    #stemming and tokenizing
    Temp = []
    lemmatiser = WordNetLemmatizer()
    stemmer = PorterStemmer()
    for i in data['tidy']:
        i = i.lower().split()
        temp = [lemmatiser.lemmatize(temp) for temp in i]
        Temp.append(temp)

    data['tidy'] = Temp
    print(data.head)

# Comment for wordcloud
    '''
    #Printing wordcloud
    split_words = ' '.join([str(x) for x in data['tidy']])
    wordcloud = WordCloud( width = 1000, height = 1000, background_color = 'black', max_words = 1000, min_font_size = 20).generate(split_words)

    #plot wordcloud
    plt.figure(figsize = (12,12),facecolor = None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

    #word clod for negative tweets
    neg_words = ' '.join([str(x) for x in data['tidy'][data['label'] == 0]])
    wordcloud = WordCloud( width = 1000, height = 1000, background_color = 'black', max_words = 1000, min_font_size = 20).generate(neg_words)

    #plot wordcloud
    plt.figure(figsize = (12,12),facecolor = None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

    #word clod for positive tweets
    neg_words = ' '.join([str(x) for x in data['tidy'][data['label'] == 1]])
    wordcloud = WordCloud( width = 800, height = 800, background_color = 'black', max_words = 1000, min_font_size = 20).generate(neg_words)

    #plot wordcloud
    plt.figure(figsize = (12,12),facecolor = None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()'''

    #remove twitter handles
    print('getting # tags')
    tags = []
    for i in data['tidy']:
        temp = re.findall(r"#[\w]*", str(i))
        tags.append(temp)
    data['tags'] = tags

    Tags = sum(tags, [])
    Tags = nltk.FreqDist(Tags)

    Tag = pd.DataFrame({'tags': list(Tags.keys()), 'value': list(Tags.values())})
    d = Tag.nlargest(columns = 'value', n=50)
    sns.barplot(data=d, x='tags', y='value')

    #Remove stopwords
    temp = ['no', 'not', 'never',"aren't",'ain', 'aren', 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stop_words = list(stop_words)
    sw = [x for x in stop_words if x not in temp]
    stop_words = sw # stop words without negation words to evaluate negative comments

    Temp = []
    for t in data['tweet']:
        tokens= word_tokenize(t)
        temp = [x for x in tokens if x not in stop_words]
        temp = [lemmatiser.lemmatize(word) for word in temp]
        Temp.append(temp)

    print(Temp)
    data['lemmatised'] = Temp
    print(data.columns)
    print(data['lemmatised'])
    return data

#load the data
train_data = pd.read_csv('C:/Rahul/NLP/av/train_2kmZucJ.csv')
train_data.drop('id', axis=1,inplace=True)
print(train_data.columns)
train_data = Transform_Data(train_data)

test_data = pd.read_csv('C:/Rahul/NLP/av/test_oJQbWVk.csv')
test_data = Transform_Data(test_data)
print(train_data['tidy'][1:5])

#train and test data
xtrain = pd.DataFrame(train_data['lemmatised'])
xtest = pd.DataFrame(test_data['lemmatised'])
ytrain = train_data['label']


#Tf-Idf vectorizer
vector = TfidfVectorizer(min_df=10, tokenizer=lambda doc: doc,lowercase=False)
xtraintfidf = vector.fit_transform(train_data['lemmatised'])
xtesttfidf = vector.transform(test_data['lemmatised'])

#logestic regression mode
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtraintfidf,ytrain)
ypred = model.predict(xtesttfidf)
ypred
