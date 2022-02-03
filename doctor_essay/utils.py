# coding: utf-8

import re
import numpy as np
from conf import SEPS_V1, SEPS_V2

#============= 提供 文本 / para_lis 维度的接口


def get_word_num(content):
    return len(content) / 3

def print_lis(lis):
    for _ in lis:
        print _

def print_dict(dic):
    for key, val in dic.items():
        print key, val

class GetPara(object):
    def __init__(self, merge_word_num, other_seps=SEPS_V1+SEPS_V2, once_seps=[]):
        self.once_seps = once_seps
        self.other_seps = other_seps
        self.seps = self.once_seps + self.other_seps
        self.merge_word_num = merge_word_num

    def get_para_lis(self, content):
        if not content or type(content) != str:
            return []

        para_lis = []
        while True:
            sep = self.find_next_sep(content)
            if not sep:
                break
            para, content = content.split(sep, 1)
            para_lis.append(para)
        
        if len(para_lis) == 0:
            para_lis = [content]

        return para_lis

    def merge_para(self, para_lis):
        '''
        合并段落，保证每个段落字数 > merge_word_num
        '''
        final_lis = []

        tmp = ""
        for _ in para_lis:
            tmp = tmp + _
            if get_word_num(tmp) < self.merge_word_num:
                continue
            final_lis.append(tmp)
            tmp = ""

        if len(tmp) > 0: 
            final_lis.append(tmp)
        
        return final_lis

    def find_next_sep(self, content):
        '''
        用于找下一个分割符，可以判断哪些内容能被这种方式分割。
        '''
        exist_seps = []
        indexs = []

        for sep in self.seps:
            index = content.find(sep)
            if index == -1:
                continue
            else:
                exist_seps.append(sep)
                indexs.append(index)

        if not exist_seps:
            return None
        else:
            return exist_seps[np.argmin(indexs)]

    def _delete_sep(self, sep):
        # modify self.seps
        seps = [_ for _ in self.seps if _!= sep ]
        self.seps = seps
        return

class GetCategoryWeight(object):
    def __init__(self, category_to_keywords):
        '''
        category_to_keywords: 整体非空, 每个category不含空字符串, 均为合法字符(str)
        '''
        if self._is_category_to_keywords_invalid(category_to_keywords):
            raise Exception("wrong init category_to_keywords")

        self._category_to_keywords = category_to_keywords
        self._categories = category_to_keywords.keys()


    def get_category_weight(self, para_lis):
        '''
        para_lis: 
        '''
        category_to_weight = self._init_category_to_weight()

        if len(para_lis) == 0:
            return category_to_weight

        one_para_weight = self._get_para_weight(para_lis)
        for one_para in para_lis:
            if len(one_para) == 0:
                continue
            
            category_to_keyword_cnt = self._get_category_to_keyword_cnt(one_para)
            one_para_category_to_weight = self._get_category_weight_from_keyword_cnt(category_to_keyword_cnt)
            for category in self._categories:
                category_to_weight[category] += one_para_category_to_weight[category] * one_para_weight
            
        return category_to_weight

    def debug_one_para_category_weight(self, one_para, category='', only_case=False):
        '''
        one_para = para_lis[0]
        '''
        
        one_para_category_to_keyword_cnt = self._get_category_to_keyword_cnt(one_para)
        one_para_category_to_weight = self._get_category_weight_from_keyword_cnt(one_para_category_to_keyword_cnt)

        if not only_case:
            print_dict(one_para_category_to_keyword_cnt)
            print '======================='
            print_dict(one_para_category_to_weight)

        if category not in self._category_to_keywords:
            return 
        
        for keyword in self._category_to_keywords[category]:
            cnt = self._get_word_cnt(one_para, keyword)
            if cnt == 0:
                continue
            print_lis(re.findall('.{6}' + keyword + '.{6}', one_para))

    def _init_category_to_weight(self):
        category_to_weight = {}
        for category in self._categories:
            category_to_weight[category] = 0.0
        return category_to_weight
    
    def _is_category_to_keywords_invalid(self, category_to_keywords):
        category_to_keywords_invalid = (type(category_to_keywords) != dict or len(category_to_keywords) == 0)
        return category_to_keywords_invalid

    def _get_category_weight_from_keyword_cnt(self, category_to_keyword_cnt):

        all_keyword_cnt = sum(np.array(category_to_keyword_cnt.values()))
        if all_keyword_cnt == 0:
            return {category: 0.0 for category in self._categories}
        else:
            return {
                category: category_to_keyword_cnt[category] * 1.0 / all_keyword_cnt
                for category in self._categories
            }

    def _get_category_to_keyword_cnt(self, content):
        category_to_keyword_cnt = {}

        for category in self._categories:
            cnt = 0
            for keyword in self._category_to_keywords[category]:
                cnt += self._get_word_cnt(content, keyword)
            category_to_keyword_cnt[category] = cnt
        
        return category_to_keyword_cnt


    def _get_para_weight(self, para_lis):
        '''
        要求 para_lis 非空
        '''

        return 1.0 / len(para_lis)

    def _get_word_cnt(self, content, word):
        return len(re.findall(word, content))