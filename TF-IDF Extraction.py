from lxml import etree
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer

xml_file = 'C:/Users/zzeni/PycharmProjects/Key Terms Extraction/Key Terms Extraction/task/news.xml'
root = etree.parse(xml_file).getroot()
lemmatizer = WordNetLemmatizer()
vect = TfidfVectorizer()

def get_head(head_ind):
    return root[0][head_ind][0].text + ':'


def normalize(text_index):
    text = root[0][text_index][1].text.lower()
    token_text = word_tokenize(text)
    lemm_text = [lemmatizer.lemmatize(tok, pos='n') for tok in token_text
                 if tok not in punctuation]
    nonstop = [word for word in lemm_text
               if word not in stopwords.words('english')]
    tag_text = [pos_tag([word])[0][0] for word in nonstop
                if pos_tag([word])[0][1] == 'NN']
    return ' '.join(tag_text)


dataset = []
for i in range(10):
    dataset.append(normalize(i))


tfidf_matrix = vect.fit_transform(dataset).toarray()
terms = vect.get_feature_names_out()
dimension = tfidf_matrix.shape

row = 0
while row < dimension[0]:
    print(get_head(row))
    column = 0
    word_rate = []
    while column < dimension[1]:
           if tfidf_matrix[row][column] != 0:
                 word_rate.append((terms[column], tfidf_matrix[row][column]))
           column += 1
    sorted_words = sorted(word_rate, key=lambda x: (x[1], x[0]), reverse=True)[0:5]
    words = []
    for i in range(5):
        words.append(sorted_words[i][0])
    print(*words)
    row += 1
    print()
    