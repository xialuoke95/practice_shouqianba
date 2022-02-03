#encoding: utf-8


from conf import SEPS
from utils import get_word_num, GetPara, GetCategoryWeight, print_lis, print_dict
import utils
import conf
reload(conf)
reload(utils)


if __name__=="__main__":

    import pandas as pd
    data = pd.read_csv('./politics_doc.csv')[0:10]
    data = data.fillna('', inplace=False)
    rule_csv = pd.read_csv(
        './politics_doc_category_rule.csv'
    )
    data.columns = [
        'a1', 'a2', 'title', 'a3', 'topic', 'publisher', 'finish_date', 
        'id', 'publish_date', 'keyword', 'url', 'content', 'word_num', 'has_attachment', 
        'a5', 'is_han', 'a7'   
    ]
    rule_csv.columns = ['category_root', 'category', 'keywords']
    category_to_keywords = {}
    for idx, row in rule_csv.iterrows():
        category_to_keywords.update({
            row['category']:filter(None, row['keywords'].strip().split(' '))
        })

    content_paragraph = []
    para_getter = GetPara(merge_word_num=40)
    para_list = []
    para_word_num = []
    para_num = []

    for content in data['content']:
        para_s_ = para_getter.get_para_lis(content)
        para_s = para_getter.merge_para(para_s_)

        word_num = [get_word_num(para) for para in para_s]
        
        para_list.append(para_s)
        para_num.append(len(para_s))
        
        para_word_num.append(word_num)
        
    weight_getter = GetCategoryWeight(category_to_keywords=category_to_keywords)
    one_para = para_list[1][0]
    weight_getter.debug_one_para_category_weight(one_para, category='环境', only_case=False)
    weight_getter.debug_one_para_category_weight(one_para, category='其他产业支持', only_case=False)


    print '=================='
    print one_para