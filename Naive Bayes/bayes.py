from numpy import *

def textParse1(vec):    
    return 1 if vec[0] == 'spam' else 0,vec[1:];
    
def textParse2(vec): 
    return vec[0],vec[1:];
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def setOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def createVocabList(dataSet):
    vocabSet = set([])  
    for document in dataSet:
        vocabSet = vocabSet | set(document) 
    return list(vocabSet)

def tfIdf(trainMatrix,setMatrix):
    n = len(trainMatrix)
    m = len(trainMatrix[0])
    d = [n]*n;
    tb = sum(trainMatrix,axis=0)
    tc = sum(setMatrix,axis=0)
    b = array(tb,dtype='float')
    c = array(tc,dtype='float')
    weight = []
    for i in range(m):
        a = trainMatrix[:,i]
        tf = a/b[i]
        weight.append(tf * log(d/(c[i])))
    returnVec = array(weight).transpose()
    return returnVec
    

def trainNB0(trainMatrix,trainCategory,weight):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)     
    p0Denom = 2.0; p1Denom = 2.0      
    a = 0;b = 0    
    a += trainMatrix[0];b += sum(trainMatrix[0])
   # print a,b        
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]*weight[i]
            p1Denom += sum(trainMatrix[i]*weight[i])
        else:
            p0Num += trainMatrix[i]*weight[i]
            p0Denom += sum(trainMatrix[i]*weight[i])
    p1Vect = log(p1Num/p1Denom)     
    p0Vect = log(p0Num/p0Denom)    
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def spamTest():
    trainFile = './train.csv'
    testFile = './test.csv'
    import csv
    docList=[]; classList = []; fullText =[]
    in1 = open(trainFile);in1.readline()
    fr1 = csv.reader(in1)
    trainData = [row for row in fr1]
    
   # prepare trainData
    
    n = 0
    for i in trainData:
        label,wordList = textParse1(i)
        docList.append(wordList)
    #    fullText.extend(wordList)
        classList.append(label)
        n += 1
        
    vocabList = createVocabList(docList)
    
    trainMat = [];trainClasses = []
    setMat = []
    
    for docIndex in range(n):
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        setMat.append(setOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    
    #traiing by bayes
    weight = tfIdf(array(trainMat),array(setMat))
    
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses),array(weight))
    
    
    #prepare testData
    
    in2 = open(testFile);in2.readline()
    fr2 = csv.reader(in2)
    fw = csv.writer(open('predict.csv', 'w'))
    name = ['SmsId','Label']
    fw.writerow(name)
    testData = [row for row in fr2]
    
    #get testData
    
    for i in testData:
        id,wordList = textParse2(i)
        wordVector = bagOfWords2VecMN(vocabList, wordList)
        fw.writerow([id,'spam' if classifyNB(array(wordVector),p0V,p1V,pSpam) else 'ham'])
    
    print 'fianl point'
