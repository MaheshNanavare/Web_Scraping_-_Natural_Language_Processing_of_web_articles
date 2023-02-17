# Topic: Data Extraction and NLP
# Objective: To extract textual data of articles from the given URLs and perform text analysis. 

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')


######### DATA EXTRACTION #########

input = pd.read_excel("Data/input.xlsx",index_col=0)
headers = {'User-Agent': '109.0.1518.61'}

for url_id in input.index:
    url = input.URL.loc[url_id]
    print(url_id,url)
    
    # Web Scrapping
    req = requests.get(url,headers=headers)
    soup = BeautifulSoup(req.content, "html.parser")
    
    # Write data in a file
    file = open("TextFiles/"+str(url_id)+".txt","w",encoding="utf-8")
    file.write(soup.title.text)
    print(url_id, soup.title.text)

    for data in soup.find_all("p"):
        file.write(' '+data.get_text())  # Added space to separate 2 paragraphs
    file.close()


######### DATA ANALYSIS #########

output_df = pd.read_excel('Data/Output Data Structure.xlsx', index_col=0)
print(output_df.shape)
output_df.sample()

# Making a list with texts from all the text files (from URLs)
texts = []
for i in output_df.index:
    with open('TextFiles/'+str(i)+'.txt',encoding="utf-8",errors="ignore") as f:
        contents = f.read()
    texts.append(contents)
len(texts)

# Load text data from txt files into dataframe
texts_df = pd.DataFrame(output_df.loc[:,'URL'])
texts_df.loc[:,'texts'] = texts
texts_df.sample()

stop_words_eng = set(stopwords.words('english'))


######### 1.Sentimental Analysis #########

##### 1.1 Cleaning using Stop Words #####
column_names = ['POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 'SUBJECTIVITY SCORE', 
                'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 'FOG INDEX',
                'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
                'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']

# For 'Page not found' URLs
no_of_pages_not_found = 0
pages_not_found = []
for i in texts_df.index:
    
    if ('Page not found' in texts_df.texts.loc[i]) or (len(texts_df.texts.loc[i])==0):
        for column in column_names:
            output_df.at[i,column] = 0
        no_of_pages_not_found += 1
        pages_not_found.append(i)
        continue
    
# Convert text into list of tokens using nltk tokenize module 
    word_tokens = word_tokenize(texts_df.texts.loc[i])
# Remove the stop words
    filtered_words = [w for w in word_tokens if w.lower() not in stop_words_eng]
# Remove the punctuations
    clean_words = [w for w in filtered_words if w not in string.punctuation]
    
##### 1.2 Creating dictionary of Positive and Negative words #####
    p_words_df = pd.read_csv('MasterDictionary/positive-words.txt',header=None,names=['words'])
    n_words_df = pd.read_csv('MasterDictionary/negative-words.txt',header=None,names=['words'])

    p_words = list(p_words_df.words)
    n_words = list(n_words_df.words)
    
##### 1.3 Extracting Derived variables #####
# 1.POSITIVE SCORE
# 2.NEGATIVE SCORE
# 3.POLARITY SCORE
    positive_score, negative_score = 0, 0
    for word in filtered_words:
        if word in p_words:
            positive_score += 1
        elif word in n_words:
            negative_score += 1

    polarity_score = (positive_score-negative_score)/(positive_score+negative_score+0.000001)
    
# 4.SUBJECTIVITY SCORE
    total_words = len(filtered_words)
    subjectivity_score = (positive_score+negative_score)/(total_words+0.000001)


######### Analysis of Readability #########

# 5.WORD COUNT
    no_of_words = len(clean_words)

# 6.AVG SENTENCE LENGTH
    sentence_tokens = sent_tokenize(texts_df.texts.iloc[0])
    no_of_sentences = len(sentence_tokens)
    avg_sentence_length = no_of_words/no_of_sentences
    
# 7.COMPLEX WORD COUNT
    vowels = ['a','e','i','o','u']
    no_of_complex_words = 0
    total_syllables = 0
    for word in clean_words:

        vowel_count = 0
        for v in vowels:
            vowel_count += word.count(v)

        syllable_count = vowel_count
        if word[-2:] in ["es","ed"]:
            syllable_count -= 1

        total_syllables += syllable_count
        if syllable_count > 2:
            no_of_complex_words += 1

# 8.PERCENTAGE OF COMPLEX WORDS
    percentage_of_complex_words = 100*no_of_complex_words/no_of_words

# 9.FOG INDEX
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)

# 10.AVG NUMBER OF WORDS PER SENTENCE
    avg_no_of_words_per_sentence = avg_sentence_length
        
# 11.SYLLABLE PER WORD
    syllables_per_word = total_syllables/no_of_words
    
# 12.PERSONAL PRONOUNS
    no_of_pronouns = 0
    personal_pronouns = ['i','me','mine','you','yours','he','him','his','she','her','hers','it','we','us','ours','they','them','theirs']
    for word in word_tokens:
        if word in personal_pronouns:
            no_of_pronouns += 1
            
# 13.AVG WORD LENGTH
    total_letters = 0
    for word in clean_words:
        total_letters += len(word)

    avg_word_length = total_letters/no_of_words
    
    variable_names = [positive_score, negative_score, polarity_score, 
        subjectivity_score, avg_sentence_length, 
        percentage_of_complex_words, fog_index,
        avg_no_of_words_per_sentence, no_of_complex_words, no_of_words,  
        syllables_per_word, no_of_pronouns, avg_word_length]
    
    for (column,variable) in zip(column_names,variable_names):
        output_df.at[i,column] = variable
        output_df = output_df.round(2)
        
output_df.to_excel('Data/output.xlsx')