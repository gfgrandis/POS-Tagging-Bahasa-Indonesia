import nltk
import string
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.tag import CRFTagger
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer
from nlp_id.postag import PosTag

    
####################### ~NLP-ID~ ######################
#Tokenisasi
def token_nlpid(hasil_casefold):
    result = []
    tokenizer = Tokenizer()
    for i in hasil_casefold:
        for j in i:
            tokens = tokenizer.tokenize(j)
            result.append(tokens)
    
    return result

#Lematisasi
def lemma_nlpid(hasil_token):
    result = []
    data = []
    lemmatizer = Lemmatizer()
    for i in hasil_token:
        for j in i:
            lemma = lemmatizer.lemmatize(j)
            if j != lemma:
                i.remove(j)
                i.append(lemma)
        result.append(i)
        
    return result

#Pos Tag
def postag_nlpid(hasil_lemma):
    result = []
    postagger = PosTag()
    for i in hasil_lemma:
        for j in i:
            postag = postagger.get_pos_tag(j)
            result.append(postag)

    return result

def ubah_tagset(nlpid):
    data = []
    result = []
    for i in nlpid:
        for j in i:
            if j[1] == 'ADV' or j[1] == 'ADJP':
                temp = (j[0], 'RB')
                del j
                data.append(temp)
            elif j[1] == 'NUM' or j[1] == 'NUMP':
                temp = (j[0], 'CD')
                del j
                data.append(temp)
            elif j[1] =='PR' and (j[0] == 'saya' or j[0] == 'kami' or j[0] == 'kita' or j[0] == 'kamu' or j[0] =='mereka' or j[0] == 'kalian'):
                temp = (j[0], 'PRP')
                del j
                data.append(temp)
            elif j[1] == 'VP':
                temp = (j[0], 'VB')
                del j
                data.append(temp)
            elif j[1] == 'NP' or j[1] == 'DP':
                temp = (j[0], 'NNP')
                del j
                data.append(temp)
            else:
                data.append(j)
        if data not in result:
            result.append(data)
    return result
            

######################### ~CRF TAGGER~ ##########################
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
    return result

# Stemming
def stemming(hasil_token):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stem = []
    for kata in hasil_token:
        for x in kata:
            output = stemmer.stem(x)
            if x != output:
                kata.remove(x)
                kata.append(output)
        stem.append(kata)
    return stem

#CRF Tagger
def crftagger(hasil_stem):
    result = []
    ct = CRFTagger()
    ct.set_model_file('C://Users/Grandis/Downloads/all_indo_man_tag_corpus_model.crf.tagger')
    for i in hasil_stem:
        hasil = ct.tag_sents([i])
        for j in hasil:
            result.append(j)
    return result

########################## ~EVALUASI~ ##################################
#POSTAG manual
def manualisasi(file):
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

#Akurasi
def accuracy(data_postag,data_manual):
    hitung = 0
    benar = 0
    total = 0
    salah = 0
    manual = []
    for i in data_manual:
        for j in i:
            total += 1
            manual.append(j)
    # print(manual)
    for i in data_postag:
        for j in i:
            if j in manual:
                benar += 1
                break
            else:
                salah += 1
                # print(j) """Print word tag yang salah"""
    hitung = ((total-salah)/total)*100 
    return hitung

def accuracy2(data_postag,data_manual):
    hitung = 0
    benar = 0
    total = 0
    salah = 0
    temp = []
    manual = []
    for i in data_manual:
        for j in i:
            total += 1
            temp.append(j)
        if temp not in manual:
            manual.append(temp)
    # print(manual)
    for i in data_postag:
        for j in i:
            for k in manual:
                if j in k:
                    benar += 1
                    break
                else:
                    salah += 1
                    #print(j) #"""Print word tag yang salah"""
    hitung = ((total-salah)/total)*100 
    return hitung

############################# ~MAIN~ ################################
def main():
    parse_dokuji = parsing("D://corpus.txt")
    hasil_casefolding = case_folding(parse_dokuji)
    hasil_tokenisasi = tokenisasi(parse_dokuji)
    hasil_stemming = stemming(hasil_tokenisasi)
    crf_tagger = crftagger(hasil_stemming)
    print('CRF-Tagger')
    print(crf_tagger)

    hasil_token_nlpid = token_nlpid(parse_dokuji)
    hasil_lemma_nlpid = lemma_nlpid(hasil_token_nlpid)
    nlp_id = postag_nlpid(hasil_lemma_nlpid)
    fix_nlpid = ubah_tagset(nlp_id)
    print('\nNLP-ID')
    print(fix_nlpid)

    hasil_manual = manualisasi("D://corpus_manual.txt")
    akurasi = accuracy(crf_tagger,hasil_manual)
    akurasi2 = accuracy2(fix_nlpid,hasil_manual)
    print('Akurasi dari CRF-Tagger = '+ str(akurasi) +'%')
    print('Akurasi dari CRF-Tagger = '+ str(akurasi2) +'%')

main()
