from __future__ import division
import numpy as np
import os
import pickle
import math

#grab files that are spam and regular
spamFiles = os.listdir('Spam')
hamFiles = os.listdir('Ham')

#place them in lists
spam = []
ham = []

#add each spam file into list
for fname in spamFiles:
    f = open("Spam/"+fname)
    mailStr = ""
    for line in f:
        mailStr = mailStr + line
    spam.append(mailStr)

#add each regular file into list
for fname in hamFiles:
    f = open("Ham/"+fname)
    mailStr = ""
    for line in f:
        mailStr = mailStr + line
    ham.append(mailStr)

#dictionary declaration
data = {'spam': spam, 'ham': ham}

#writing files
pickleOut = open('Dataset.pickle', 'wb')
pickle.dump(data, pickleOut)

#function to convert to number
def isNum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Split data into training and testing samples

stopwords = set([" ", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours	ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])

avoidChars = set([".", "+", "-", "|", "\r", "\n", "/", ":", "!", "\x00", "\xff", "=", "%", "'", ",", "*", "_", "__"])

#remove special characaters in avoidChars from spam files 
for idx, document in enumerate(spam):
    spam[idx] = "".join([c if c not in avoidChars else " " for c in spam[idx]])
    spam[idx] = spam[idx].split(" ")

#remove special characters in avoidChars from ham files
for idx, document in enumerate(ham):
    ham[idx] = "".join([c if c not in avoidChars else " " for c in ham[idx]])
    ham[idx] = ham[idx].split(" ")

#for every character in file... if char not "", not in stopwords & not float lower char in regular file
for idx, doc in enumerate(ham):
    ham[idx] = [item.lower() for item in ham[idx] if item != "" and item.lower() not in stopwords and not isNum(item.lower())]

#for every character in file... if char not "", not in stopwords & not float lower
char in spam file
for idx, doc in enumerate(spam):
    spam[idx] = [item.lower() for item in spam[idx] if item != "" and item.lower() not in stopwords  and not isNum(item.lower())]

# Split into train & Test set

#testing 10% of all files
spamtest = spam[int(len(spam)*0.9):]
hamtest = ham[int(len(ham)*0.9):]

#validation set 85%-90%
spamval = spam[int(len(spam)*0.85): int(len(spam)*0.9)]
hamval = ham[int(len(ham)*0.85):int(len(ham)*0.9)]

#training set 85% of files
spam = spam[:int(len(spam)*0.85)]
ham = ham[:int(len(ham)*0.85)]

print len(spamval), len(spam)

# We calculate the prior before we split into training and testing samples

#probability file is spam & regular file
Pspam = len(spam)*1.0/(len(spam)+len(ham))
Pham = 1-Pspam

Pspam, Pham

# We also need the conditional probability of each word given it is spam and non spam.

# Calculate set of all words.
wordList = set()
for lst in spam:
    for word in lst:
        wordList.add(word)

for lst in ham:
    for word in lst:
        wordList.add(word)

# P_wc = Probability of finding word w given document class c.
# P_wc = (count(wi, c)+alpha) / sum(count(w, c)+alpha)

totalSpamWords = 0
totalHamWords = 0

countSpam = {}
countHam = {}

#for every word in each spam files place it in spam dictionary (countSpam)
for lst in spam:
    for word in lst:
        if word not in countSpam:
            countSpam[word] = 0
        countSpam[word] += 1
        totalSpamWords += 1

for lst in ham:
    for word in lst:
        if word not in countHam:
            countHam[word] = 0
        countHam[word] += 1
        totalHamWords += 1

alpha = 1
 
#if passed in word on file is in spam dict return log> 1/totalSpamWords+alpha*len(wordList)
def P_wc(word, spam):
    #if spam (word_pos_in_dict+alpha) / (totalSpamWords+alpha*len(wordList))
    if(spam):
        occ = 0
        if word in countSpam:
            occ = countSpam[word]
        return np.log((occ + alpha)*1.0 / (totalSpamWords + alpha*len(wordList)))
    #if not spam (word_pos_in_dict_nonSpam+alpha) / (totalHamWords+alpha*len(wordList))
    else:
        occ = 0
        if word in countHam:
            occ = countHam[word]
        return np.log((occ + alpha)*1.0 / (totalHamWords + alpha*len(wordList)))

thresh = 0.495

def isSpam(email):
    #log of probability spam
    ps, ph = np.log(Pspam), np.log(Pham)
    #find probability email is spam
    for word in email:
        ps += P_wc(word, 1)
    #find probability email is ham
    for word in email:
        ph += P_wc(word, 0)
    return  float(ps)*1.0/(float(ps+ph)) <= thresh

#get Accuracy of predictions number of 
def getAcc(spam, ham):
    acc = 0
    for item in ham:
        if not isSpam(item):
            acc += 1    
    for item in spam:
        if isSpam(item):
            acc += 1
    return float(acc)*1.0/(float(len(spam)+len(ham)))

def getFP(spam, ham):
    joinLst = spam + ham
    fp = 0
    for item in ham:
        if isSpam(item):
            fp += 1
    print float(fp)*1.0/float(len(ham))

# Hyperparameter Selection
# Validation Data

print "Validation accuracy = ", getAcc(spamval, hamval)
print "Validation false positive rate = ", getFP(spamval, hamval)

print "Test accuracy = ", getAcc(spamtest, hamtest)
print "Test false positive rate = ", getFP(spamtest, hamtest)

