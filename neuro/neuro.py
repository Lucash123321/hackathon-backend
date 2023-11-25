from nltk.corpus import stopwords
import nltk

import gensim
from gensim.utils import simple_preprocess
import keras
from keras.models import load_model
import pickle
import re
import numpy as np
import shap
from deeppavlov import build_model
from django.conf import settings
import os


class MessageResults():
    def __init__(self):
        base_dir = settings.BASE_DIR
        dir = os.path.join(base_dir, 'neuro/')
        self.CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});\n')
        self.Model_11 = load_model(dir + "Model_11.h5")
        self.Model_27 = load_model(dir + "Model_27.h5")
        with open(dir + "TFIDF_Vectorizers.pkl", "rb") as f:
            self.tfidf_Centroid, self.tfidf_NN = pickle.load(f)
        with open(dir + "Cif.pkl", "rb") as f:
            self.clf = pickle.load(f)
        with open(dir + "Labels.pkl", "rb") as f:
            self.LabelsText1, self.LabelsText2, self.LabelsText3 = pickle.load(f)
        self.LabelsText1 = np.array(self.LabelsText1)
        self.LabelsText2 = np.array(self.LabelsText2)
        self.LabelsText3 = np.array(self.LabelsText3)

        # self.LabelsText1=[]
        # for i in range(11):self.LabelsText1.append(str(i))

        nltk.download('stopwords')
        self.stop_words = stopwords.words('russian')
        self.shap_masker = shap.maskers.Text(r"\W")  # this will create a basic whitespace tokenizer
        self.shap_Explainer1 = shap.Explainer(self.PredictClass1, self.shap_masker, output_names=self.LabelsText1)
        self.shap_Explainer2 = shap.Explainer(self.PredictClass2, self.shap_masker, output_names=self.LabelsText2)
        self.shap_Explainer3 = shap.Explainer(self.PredictClass3, self.shap_masker, output_names=self.LabelsText3)
        self.ner_model = build_model('ner_collection3_bert', download=True, install=True)

    def cleanhtml(self, raw_html):
        '''Очистка текста от html'''
        cleantext = re.sub(self.CLEANR, '', raw_html)
        return cleantext

    def strip_newline(self, series):
        '''Разделитель строк'''
        return [review.replace('\n', '') for review in series]

    def remove_stopwords(self, texts):
        '''Очистка от стопслов'''
        out = [[word for word in simple_preprocess(str(doc))
                if word not in self.stop_words]
               for doc in texts]
        return out

    def ProcessText2(self, Text):
        Text1 = self.cleanhtml(Text)
        Text1 = Text1.replace("\n", "").replace("№", "").replace("&", "").replace("nbsp", "").replace("   ",
                                                                                                      " ").replace("  ",
                                                                                                                   " ")
        Text1 = self.remove_stopwords(self.strip_newline([Text1]))
        # print(" ".join(Text1[0]))
        return " ".join(Text1[0])

    def PredictClass1(self, Texts):
        ProcessedTexts = []
        for Text in Texts:
            ProcessedTexts.append(self.ProcessText2(Text))
        NNInputs = self.tfidf_NN.transform(ProcessedTexts)
        return self.Model_11.predict(NNInputs.toarray())

    def PredictClass2(self, Texts):
        ProcessedTexts = []
        for Text in Texts:
            ProcessedTexts.append(self.ProcessText2(Text))
        NNInputs = self.tfidf_NN.transform(ProcessedTexts)
        return self.Model_27.predict(NNInputs.toarray())

    def PredictClass3(self, Texts):
        ProcessedTexts = []
        for Text in Texts:
            ProcessedTexts.append(self.ProcessText2(Text))
        Prd = self.clf.predict(self.tfidf_Centroid.transform(ProcessedTexts))
        return np.eye(len(self.LabelsText3))[Prd]

    def Class1ToLabels(self, Prediction):
        return np.array(self.LabelsText1)[np.argmax(Prediction, axis=1)]

    def Class2ToLabels(self, Prediction):
        return np.array(self.LabelsText2)[np.argmax(Prediction, axis=1)]

    def Class3ToLabels(self, Prediction):
        return np.array(self.LabelsText3)[np.argmax(Prediction, axis=1)]

    def GetColorDataClass1(self, Texts):
        LenOk = True
        for Text in Texts:
            if (self.ProcessText2(Text).find(" ") == -1): LenOk = False
        if (LenOk):
            return self.shap_Explainer1(Texts)
        else:
            return -1

    def GetColorDataClass2(self, Texts):
        LenOk = True
        for Text in Texts:
            if (self.ProcessText2(Text).find(" ") == -1): LenOk = False
        if (LenOk):
            return self.shap_Explainer2(Texts)
        else:
            return -1

    def GetColorDataClass3(self, Texts):
        LenOk = True
        for Text in Texts:
            if (self.ProcessText2(Text).find(" ") == -1): LenOk = False
        if (LenOk):
            return self.shap_Explainer3(Texts)
        else:
            return -1

    def DizassembleNER(self, Results):
        Orgs = []
        Persons = []
        Locations = []

        for k in range(len(Results[1])):
            Result = Results[1][k]
            Orgs.append([])
            Persons.append([])
            Locations.append([])
            PassITo = 0
            # print(k,Results[0][k])
            for i in range(len(Result)):
                if (i > PassITo):
                    if (Result[i] == 'S-LOC'):
                        # print(k,i,Results[0][k][i])
                        Locations[-1].append(Results[0][k][i])
                    elif (Result[i] == 'S-PER'):
                        # print(k,i,Results[0][k][i])
                        Persons[-1].append(Results[0][k][i])
                    elif (Result[i] == 'S-ORG'):
                        # print(k,i,Results[0][k][i])
                        Orgs[-1].append(Results[0][k][i])
                    elif (Result[i] == 'B-LOC'):
                        Begin = i
                        End = i
                        for j in range(i + 1, len(Result)):
                            if (Result[j][2:] == 'LOC'):
                                End = j
                            else:
                                break
                        PassITo = End + 1
                        Locations[-1].append(" ".join(Results[0][k][Begin:End + 1]))
                    elif (Result[i] == 'B-PER'):
                        Begin = i
                        End = i
                        for j in range(i + 1, len(Result)):
                            if (Result[j][2:] == 'PER'):
                                End = j
                            else:
                                break
                        PassITo = End + 1
                        Persons[-1].append(" ".join(Results[0][k][Begin:End + 1]))
                    elif (Result[i] == 'B-ORG'):
                        Begin = i
                        End = i
                        for j in range(i + 1, len(Result)):
                            if (Result[j][2:] == 'ORG'):
                                End = j
                            else:
                                break
                        PassITo = End + 1
                        Orgs[-1].append(" ".join(Results[0][k][Begin:End + 1]))
        return Orgs, Persons, Locations

    def GetNerInstances(self, Texts):
        Results = self.ner_model(Texts)
        return self.DizassembleNER(Results)


if __name__ == "__main__":
    Classifier = MessageResults()
    Result = Classifier.GetColorDataClass1(["экземпляры по классам"])
    print(np.argmax(Result, axis=1))
    print(Classifier.Class1ToLabels(Result))

    Result = Classifier.PredictClass2(["экземпляры по классам", "Тренировка", "Здравствуйте"])
    print(np.argmax(Result, axis=1))
    print(Classifier.Class2ToLabels(Result))

    Result = Classifier.PredictClass3(["экземпляры по классам", "Тренировка", "Здравствуйте"])
    print(np.argmax(Result, axis=1))
    print(Classifier.Class3ToLabels(Result))
