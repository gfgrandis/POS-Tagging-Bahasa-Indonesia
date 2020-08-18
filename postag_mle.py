import nltk
import string
import re
import random
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.tag import CRFTagger
from nlp_id.lemmatizer import Lemmatizer

############ ~~~~~~~~Preprocessing~~~~~~~ #############

# Parsing
def parsing(file):
    f1 = open(file, "r", encoding='utf-8')
    caption = f1.read()
    f1.close()
    capt2 = caption.splitlines()
    array_caption = []
    for a in capt2:
        array_caption.append([a])
    return array_caption

#case folding
def case_folding(hasil_parsing):
    result = []
    for i in hasil_parsing:
        for j in i:
            if j != "":
                result.append([j.lower()])
    return result

# Tokenisasi
def tokenisasi(hasil_case_folding):
    result = []
    for a in hasil_case_folding:
        for b in a:
            tokens = word_tokenize(b)
            result.append(tokens)
    
    # for i in result:
    #     print(i)
    return result

# Stemming
def stemming(hasil_token):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stem = []
    result = []
    for kata in hasil_token:
        temp = []
        for x in kata:
            output = stemmer.stem(x)
            if output == '':
                output = x
            temp.append(output)
        result.append(temp)
        
    return result

######################### ~Training~ ##################################
tag_dict={}      #dictionary containing each tag with its count value
word_tag_dict={} #dictionary to keep count of a word and tag occuring together
total_no_of_tags = 0

#Buka file tagged corpus
def openfile_train(file):
    result = []
    f = open(file, "r", encoding="ANSI")
    lines = f.read().split("\n\n")
    f.close()
    for line in lines:
        kalimat = line.split("\n")
        sen_tag = []
        for kata in kalimat:
            word_tag = kata.split("\t")
            word = word_tag[0]
            tag = word_tag[1]
            if word == "ï»¿Sebuah":
                word = "Sebuah"
            sen_tag.append((word, tag))
        result.append(sen_tag)

    return result

#Ganti format menjadi dictionary
def fixing_wordtags(manualisasi):
    result = []
    for line in manualisasi:
        fix_word = []
        for wordtags in line:
            words = wordtags[0]
            tags = wordtags[1]
            word_tag=words+"_"+tags 
            fix_word.append(word_tag)
        result.append(fix_word)
    
    return result

def train(training_wordtags):
    global word_tag_dict
    global tag_dict
    global total_no_of_tags

    for line in training_wordtags:
        for token in line:
            token=token.lower()
            if not token in word_tag_dict.keys():
                word_tag_dict[token]=1
            else:
                word_tag_dict[token]=word_tag_dict[token]+1        
            token_break=token.split("_")
            word=token_break[0]
            tag=token_break[1]
            if not tag in tag_dict.keys():
                tag_dict[tag]=1
            else:
                tag_dict[tag]=tag_dict[tag]+1
        
    total_no_of_tags=sum(tag_dict.values())

#Menghitung Jumlah tagset pada tagged-corpus
def hitung_tag(tagged_corpus):
    result = {}

    for line in tagged_corpus:
        for token in line:
            token=token.lower()      
            token_break=token.split("_")
            word=token_break[0]
            tag=token_break[1]
            if not tag in result.keys():
                result[tag]=1
            else:
                result[tag]=result[tag]+1

    return result

#Menghitung kata-tag di data latih
def hitung_wordtag(corpus_latih):

    word_tag_dict = {}
    for line in corpus_latih:
        for token in line:
            token=token.lower()
            if not token in word_tag_dict.keys():
                word_tag_dict[token]=1
            else:
                word_tag_dict[token]=word_tag_dict[token]+1  
    
    return word_tag_dict

#Menghitung Total jumlah tag di data latih
def total_tag(hitung_tag):

    jumlah_total=sum(hitung_tag.values())

    return jumlah_total

###################### ~~~~~~Testing~~~~ ##################
tag_given_word={}

def testing(hasil_prepro, tag_dict):
    global tag_given_word
    proba = []
    result = []

    for kalimat in hasil_prepro:
        for word_token in kalimat :
            kata=word_token.lower()
            max_prob_of_tag=0
            predicted_tags_with_max_probability=[] #list of tags with max probability.All tags in this list have equal probability which is maximum.
            predicted_tags_with_max_probability.append("NNP") #take default tag as NNP
            for tag in tag_dict.keys():
                word_tag=kata+"_"+tag   
                if word_tag in word_tag_dict.keys():
                    likelihood_of_word_given_tag=word_tag_dict[word_tag]/float(tag_dict[tag])
                    
                    if likelihood_of_word_given_tag>max_prob_of_tag:
                        predicted_tags_with_max_probability=[] #if probabilty is greater create new list of predicted tags
                        predicted_tags_with_max_probability.append(tag)
                    elif likelihood_of_word_given_tag==max_prob_of_tag:  #if probabilty is same add to list of predicted tags
                        predicted_tags_with_max_probability.append(tag)
                        
                    
                    if not word_tag in tag_given_word.keys():
                        tag_given_word[word_tag]=likelihood_of_word_given_tag
                        f3 = ("P("+tag.upper()+"|"+word_token+")"+":"+str(tag_given_word[word_tag])+"\n")
                        # print(f3)
                else :
                    if not word_tag in tag_given_word.keys():
                        tag_given_word[word_tag]=0
                        f3 = ("P("+tag.upper()+"|"+word_token+")"+":"+str(tag_given_word[word_tag])+"\n")
                        
            # f2 = (word_token, random.choice(predicted_tags_with_max_probability).upper()) #randomly choose from list of max & equal probable tags
            f2 = (word_token, predicted_tags_with_max_probability[0].upper())
            result.append(f2)
    return result


def main():
    parse_dokuji = parsing("D://dataset/corpus.txt")
    hasil_casefolding = case_folding(parse_dokuji)
    hasil_tokenisasi = tokenisasi(parse_dokuji)
    hasil_stemming = stemming(hasil_tokenisasi)
    korpus_hmm = openfile_train("D://dataset/corpus_manual.txt")
    coba = fixing_wordtags(korpus_hmm)
    jumlahtag = hitung_tag(coba)
    totaltag = total_tag(jumlahtag)
    word_tag = hitung_wordtag(coba)
    training = train(coba)
    test = testing(hasil_tokenisasi, jumlahtag)
    print(test)

main()