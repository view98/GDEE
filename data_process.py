import logging
import json
import pickle
import numpy as np
logger = logging.getLogger(__name__)

event_roles_short = {
    'EquityOverweight': ['EH','TS', 'LHS', 'SD', 'ED', 'AP'],
    'EquityUnderweight':['EH','TS', 'LHS', 'SD', 'ED', 'AP'],
    'EquityRepurchase':['CN','HTP', 'LTP', 'RS', 'CD', 'RA'],
    'EquityFreeze':  ['EH','LI','FS', 'THS', 'THR', 'SD', 'ED','UD'],
    'EquityPledge': ['PR','PE', 'SD', 'ED', 'PS', 'TPS', 'THS', 'THR', 'RD']
    }
short_roles={'EH':'EquityHolder','LI':'LegalInstitution','FS':'FrozeShares','THS':'TotalHoldingShares',
             'THR':'TotalHoldingRatio','SD':'StartDate','ED': 'EndDate', 'UD': 'UnfrozeDate','CN':'CompanyName',
             'CD':'ClosingDate','RA':'RepurchaseAmount', 'HTP': 'HighestTradingPrice','LTP':'LowestTradingPrice',
             'RS': 'RepurchasedShares','AP':'AveragePrice',  'TS':'TradedShares','LHS':'LaterHoldingShares',
             'RD': 'ReleasedDate', 'PE':'Pledgee', 'TPS':'TotalPledgedShares', 'PR':'Pledger',  'PS': 'PledgedShares'}

na_roles = {
    'EquityOverweight': {'LaterHoldingShares': 1, 'StartDate': 2, 'EndDate': 3, 'AveragePrice': 4},
    'EquityUnderweight': {'LaterHoldingShares': 5, 'StartDate': 6, 'EndDate': 7, 'AveragePrice': 8},
    'EquityRepurchase': {'HighestTradingPrice': 9, 'LowestTradingPrice': 10, 'ClosingDate': 11,
                         'RepurchaseAmount': 12, 'RepurchasedShares': 13},
    'EquityFreeze': {'LegalInstitution': 14, 'TotalHoldingShares': 15, 'TotalHoldingRatio': 16, 'StartDate': 17,
                     'EndDate': 18, 'UnfrozeDate': 19},
    'EquityPledge': {'StartDate': 20, 'EndDate': 21, 'TotalPledgedShares': 22, 'TotalHoldingShares': 23,
                     'TotalHoldingRatio': 24, 'ReleasedDate': 25}}

def create_pairs():
    role_pairs = {'None':{'O':0},'Trace':{'1':1},'EquityOverweight': {}, 'EquityUnderweight': {}, 'EquityRepurchase': {}, 'EquityFreeze': {},
                  'EquityPledge': {}}  # 程序生成
    role_pairs2id = {}
    event_roles = { 'EquityOverweight': [], 'EquityUnderweight': [],'EquityRepurchase': [],'EquityFreeze': [],   'EquityPledge': [] }

    for t, rs in event_roles_short.items():
        for j in range(len(rs)):
            event_roles[t].append(short_roles[rs[j]])

    index=1
    for type,roles in event_roles.items():
        for i in range(len(roles)-1):
            role_pairs[type]['B-'+roles[i],'B-'+roles[i+1]]= index+1
            role_pairs2id[index+1] = 'B-'+roles[i],'B-'+roles[i+1]

            role_pairs[type]['B-' + roles[i], 'I-' + roles[i + 1]] = index + 2
            role_pairs2id[index + 2] = 'B-' + roles[i], 'I-' + roles[i + 1]

            role_pairs[type]['I-' + roles[i], 'B-' + roles[i + 1]] = index + 3
            role_pairs2id[index + 3] = 'I-' + roles[i], 'B-' + roles[i + 1]

            role_pairs[type]['I-' + roles[i], 'I-' + roles[i + 1]] = index + 4
            role_pairs2id[index + 4] = 'I-' + roles[i], 'I-' + roles[i + 1]
            index=index + 4

    label_id_list = []
    for _,roles in role_pairs.items():
        label_id_list += list(roles.values())
    print(role_pairs)
    print(event_roles)
    return event_roles,role_pairs,role_pairs2id,label_id_list

event_roles,role_pairs,role_pairs2id,label_id_list = create_pairs()

def get_token_label(word_pos_list, arg_pos_list):
    for pos in word_pos_list:
        for arg_position in arg_pos_list:
            if pos[0] == arg_position[0] and pos[1] == arg_position[1] and pos[2] <= arg_position[2]:
                return  'B-'
            elif pos[0] == arg_position[0] and pos[1] > arg_position[1] and pos[2] <=  arg_position[2]:
                    return 'I-'

def if_throw(sent_idx, sentence,last_flag):
    if last_flag:
        return True,True

    if sent_idx <=1:
        return True,False
    elif sent_idx<=3 and ('虚假记载'in sentence or '信息披露的内容' in sentence or '误导性陈述' in sentence):   #第2、3句看情况过滤
        return True,False
    elif sentence.startswith('特此公告') or sentence.startswith('特此说明'):
        last_flag = True
        return True,True

    return False,last_flag

def filter_words(i,word,words):
    filter_idxs = []
    if word == '公司' and len(words)> i+2 and words[i + 1] == '控股' and words[i + 2] == '股东':
        filter_idxs.append(i)
        filter_idxs.append(i+1)
        filter_idxs.append(i+2)
    elif word == '证券' and len(words)> i+2 and words[i + 1] == '交易' and words[i + 2] == '系统':
        filter_idxs.append(i)
        filter_idxs.append(i+1)
        filter_idxs.append(i+2)
    elif word == '本' and len(words)> i+2 and words[i + 1] == '公司' and words[i + 2] == '接到':
        filter_idxs.append(i)
        filter_idxs.append(i+1)
        filter_idxs.append(i+2)
    elif word == '公司' and len(words)> i+1 and (words[i + 1] == '股票' or words[i+1]=='高管'or words[i+1]=='股份' or words[i+1]=='股东'):
        filter_idxs.append(i)
        filter_idxs.append(i+1)
    elif word == '限售' and len(words)> i+1 and (words[i + 1] == '流通' ):
        filter_idxs.append(i)
        filter_idxs.append(i+1)
    elif word == '流通' and len(words)> i+1 and (words[i + 1] == '股份' ):
        filter_idxs.append(i)
        filter_idxs.append(i+1)
    elif word == '控股' and len(words)> i+1 and (words[i + 1] == '股东' ):
        filter_idxs.append(i)
        filter_idxs.append(i+1)
    return filter_idxs
