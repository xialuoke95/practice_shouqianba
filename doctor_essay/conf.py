# encoding: utf-8

SEPS = ['一', '二','三','四','五','六','七','八','九','十']

SEPS_V1 = ['。' + sep + '、' for sep in SEPS] + ['：' + sep + '、' for sep in SEPS]
SEPS_V2 = ['（' + sep + '）' for sep in SEPS] +  ['(' + sep + ')' for sep in SEPS]