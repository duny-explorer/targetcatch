from keras.models import load_model
import gensim.models.keyedvectors as word2vec
import numpy as np
import pymorphy2
import tensorflow as tf
import os
import re
import sys
import datetime


class TargetCatch:
    def __init__(self):
        self.graph = tf.get_default_graph()
        self.__model = load_model('targetcatch/models/model')
        self.__morph = pymorphy2.MorphAnalyzer()
        self.__embed_map = word2vec.KeyedVectors.load_word2vec_format('targetcatch/models/model.bin', binary=True)
        self.__important_words = ['нти', '3d', 'unity3d', 'вшэ', 'аэронет', 'атс', 'python', 'skills', 'design', 'operate',
                                  'product', 'project', 'factory', 'java', 'c++', 'c1', 'c#', 'phd', 'digital', 'english',
                                  'upper', 'kaggle', 'deadline', 'manager', 'professional', 'hr', 'adobe', 'frontend',
                                  'scientist', 'iot', 'backend', 'pmanagement', 'скорочтение', 'mba']
        self.__time_markers = ['сегодня_ADV', 'завтра_ADV', 'год_NOUN', 'месяц_NOUN', 'полгода_NOUN', 'послезавтра_ADV',
                               'полдень_NOUN', 'день_NOUN', 'понедельник_NOUN', 'вторник_NOUN', 'среда_NOUN',
                               'четверг_NOUN', 'пятница_NOUN', 'суббота_NOUN', 'воскресенье_NOUN', 'январь_NOUN',
                               'февраль_NOUN', 'март_NOUN', 'апрель_NOUN', 'май_NOUN', 'июнь_NOUN', 'июль_NOUN',
                               'август_NOUN', 'сентябрь_NOUN', 'октябрь_NOUN', 'ноябрь_NOUN', 'декабрь_NOUN',
                               'полмесяца_NOUN', 'январский_ADJ', 'февральский_ADJ', 'мартовский_ADJ', 'апрельский_ADJ',
                               'майский_ADJ', 'июньский_ADJ', 'июльский_ADJ', 'августовский_ADJ', 'сентябрьский_ADJ',
                               'октябрьский_ADJ', 'ноябрьский_ADJ', 'декабрьский_ADJ', 'неделя_NOUN']

    def predict(self, data):
        text, time = self.__preprocessing_text(data)
        data = self.__vector_words([text])

        with self.graph.as_default():
            predict = self.__model.predict(data)

        return self.__correct_format(predict, time)

    def __correct_format(self, data, time):
        labels = ['label_attainable', 'label_education', 'label_unambiguity',
                  'label_attractor_knowledge', 'label_attractor_hard_skill',
                  'label_attractor_soft_skill', 'label_attractor_tool',
                  'label_attractor_community', 'label_attractor_subjectivity',
                  'label_attractor_habits', 'label_attractor_career',
                  'label_attractor_fixing', 'label_attractor_art',
                  'label_attractor_health', 'label_abstraction_level']

        true_format = dict()
        true_format['label_time_bound'] = 'Да' if time else 'Нет'

        for i in range(len(labels)):
            if i == 14:
                if max(data[0][14:]) == data[0][14]:
                    true_format['label_abstraction_level'] = 'Абстрактный'
                elif max(data[0][14:]) == data[0][15]:
                    true_format['label_abstraction_level'] = 'В предметной области'
                elif max(data[0][14:]) == data[0][16]:
                    true_format['label_abstraction_level'] = 'Конкретный'
            else:
                true_format[labels[i]] = 'Да' if data[0][i] > 0.5 else 'Нет'

        return true_format

    def __preproccessing_word(self, i, normal):
        i = i.lower()

        if self.__morph.parse(normal)[0].tag.POS not in ["INTJ", "PRCL", "CONJ", "PREP", "NPRO", 'PRTF', 'PRED'] and str(
                self.__morph.parse(normal)[0].tag.POS) != 'None':

            tag = str(self.__morph.parse(normal)[0].tag.POS)

            if str(self.__morph.parse(normal)[0].tag.POS) == "ADJF":
                tag = "ADJ"

            if str(self.__morph.parse(normal)[0].tag.POS) == "INFN":
                tag = 'VERB'

            if str(self.__morph.parse(normal)[0].tag.POS) == "ADVB":
                tag = 'ADV'

            if str(self.__morph.parse(normal)[0].tag.POS) == "NUMR":
                tag = 'NUM'

            while 'ё' in normal:
                normal = normal.replace('ё', 'е')

            if i in ['новое', 'новый', 'нового', "новые"] or normal in ['новое', 'новый', 'нового', "новые"]:
                normal = 'новый'
                tag = 'ADJ'

            return normal + "_" + tag

        elif i + "_X" in self.__embed_map and not self.__morph.parse(i)[0].tag.POS:
            return i + "_X"

    def __preprocessing_text(self, text):
        res_text = str(text)

        while 'ё' in res_text:
            res_text = res_text.replace('ё', 'е')
        res_text = re.sub('[^a-zA-Zа-яА-Я1-9-]+', ' ', res_text)
        res_text = re.sub(' +', ' ', res_text)
        res_text = res_text.strip()
        time = 0

        res = []

        for i in res_text.split():
            i = i.lower()

            try:
                num = int(i)

                if i >= datetime.datetime.now().year:
                    time = 1

                continue
            except:
                pass

            if i in self.__important_words:
                res.append(i + '_other')
                continue

            if i[-1] == '-':
                i = i[:-1]

            normal = self.__morph.parse(i)[0].normal_form if self.__morph.parse(i)[0].score > 0.5 else self.__morph.parse(i)[1].normal_form

            i = i.lower()

            word = self.__preproccessing_word(i, normal)

            if word in self.__time_markers:
                time = 1
                continue

            if word in self.__embed_map:
                res.append(word)
            else:
                word = self.__preproccessing_word(i, self.__morph.parse(i)[0].normal_form)

                if word in self.__time_markers:
                    time = 1
                    continue

                if word in self.__embed_map:
                    res.append(word)

        return " ".join(res), time

    def __vector_words(self, data):
        matrix = []
        word_out = []
        word = 0
        vector = []

        for j in data:
            for i in j.split():
                print(i)
                if i in self.__embed_map:
                    vector.append(self.__embed_map[i])
                elif 'COMP' in i:
                    vector.append(self.__embed_map[i.split('_')[0] + '_ADJ'])
                elif 'other' in i:
                    word = 1

            word_out.append(word)
            word = 0

            if vector:
                matrix.append(np.mean(vector, axis=0))
                vector = []
            else:
                matrix.append(np.zeros(300))

        return np.column_stack((np.array(matrix), word_out))

