from collections import Counter, defaultdict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch
import os
import joblib
import gensim
from torch.utils.data import Dataset
from data_process import *
from bert_serving.client import BertClient
logger = logging.getLogger(__name__)
torch.set_printoptions(profile="full")

max_args = 30
max_same_tokens = 8

def load_datasets_and_vocabs(args):
    train_example_file = os.path.join(args.cache_dir, 'train_example.pkl')
    dev_example_file = os.path.join(args.cache_dir, 'dev_example.pkl')
    test_example_file = os.path.join(args.cache_dir, 'test_example.pkl')

    train_weight_file = os.path.join(args.cache_dir, 'train_weight_catch.txt')
    dev_weight_file = os.path.join(args.cache_dir, 'dev_weight_catch.txt')
    test_weight_file = os.path.join(args.cache_dir, 'test_weight_catch.txt')

    if os.path.exists(train_example_file) and os.path.exists(dev_example_file) and os.path.exists(test_example_file):
        logger.info('Loading train_example from %s', train_example_file)
        with open(train_example_file, 'rb') as f:
            train_examples = joblib.load(f)

        logger.info('Loading dev_example from %s', dev_example_file)
        with open(dev_example_file, 'rb') as f:
            dev_examples = joblib.load(f)

        logger.info('Loading test_example from %s', test_example_file)
        with open(test_example_file, 'rb') as f:
            test_examples = joblib.load(f)

        with open(train_weight_file, 'rb') as f:
            train_label_weight = json.load(f)
        with open(dev_weight_file, 'rb') as f:
            dev_label_weight = json.load(f)
        with open(test_weight_file, 'rb') as f:
            test_label_weight = json.load(f)

    else:
        # get examples of data
        train_file = os.path.join(args.dataset_path, 'train.json')
        train_examples,train_label_weight = create_example(args,train_file,'train')
        logger.info('Creating train examples')
        with open(train_example_file, 'wb') as f:
            joblib.dump(train_examples, f)
        with open(train_weight_file, 'w') as wf:
            json.dump(train_label_weight, wf)

        dev_file = os.path.join(args.dataset_path, 'dev.json')
        dev_examples,dev_label_weight = create_example(args,dev_file,'dev')
        logger.info('Creating dev examples')
        with open(dev_example_file, 'wb') as f:
            joblib.dump(dev_examples, f)
        with open(dev_weight_file, 'w') as wf:
            json.dump(dev_label_weight, wf)

        test_file = os.path.join(args.dataset_path, 'test.json')
        test_examples,test_label_weight = create_example(args,test_file,'test')
        logger.info('Creating test examples')
        with open(test_example_file, 'wb') as f:
            joblib.dump(test_examples, f)
        logger.info('Creating test_weight_cache')
        with open(test_weight_file, 'w') as wf:
            json.dump(test_label_weight, wf)

    logger.info('Train set size: %s', len(train_examples))
    logger.info('dev set size: %s', len(dev_examples))
    logger.info('Test set size: %s,', len(test_examples))

    # Build word vocabulary(dep_tag, part of speech) and save pickles.
    word_vecs, word_vocab,wType_tag_vocab = load_and_cache_vocabs(train_examples+dev_examples+test_examples, args)
    embedding = torch.from_numpy(np.asarray(word_vecs, dtype=np.float32)).squeeze(1)
    args.embedding = embedding

    train_dataset = EE_Dataset(train_examples, args, word_vocab,wType_tag_vocab)
    dev_dataset = EE_Dataset(dev_examples, args, word_vocab, wType_tag_vocab)
    test_dataset = EE_Dataset(test_examples, args, word_vocab,wType_tag_vocab)

    return train_dataset,dev_dataset, test_dataset,train_label_weight,dev_label_weight,test_label_weight, word_vocab,wType_tag_vocab,test_examples

def create_example(args,file,file_type):
    with open(file, 'r', encoding='utf-8-sig') as fp:
        datas = json.load(fp)
    if file_type == 'train':
        hanlp_examples_file = os.path.join(args.cache_dir, 'hanlp_train_examples.pkl')
    else:
        hanlp_examples_file = os.path.join(args.cache_dir, 'hanlp_test_examples.pkl')

    if os.path.exists(hanlp_examples_file):
        logger.info('Loading hanlp_examples_file from %s', hanlp_examples_file)
        with open(hanlp_examples_file, 'rb') as f:
            hanlp_examples = joblib.load(f)
        filter_name = os.path.join(args.cache_dir, file_type + '_hanlpfilters.txt')
        with open(filter_name, 'r', encoding='utf-8-sig') as ft:
            filter_doc = json.load(ft)
    else:
        # get hanlp_examples of data
        hanlp_examples,filter_doc = create_hanlp(datas)
        logger.info('Creating  hanlp_examples')
        with open(hanlp_examples_file, 'wb') as f:
            joblib.dump(hanlp_examples, f)
        filter_name = os.path.join(args.cache_dir, file_type + '_hanlpfilters.txt')
        with open(filter_name, 'w') as fp:
            json.dump(filter_doc, fp)

    examples = []
    count = Counter()
    unfinish = []
    for d,doc in enumerate(datas):
        docid = hanlp_examples[d]['docid']
        if docid in filter_doc:
            continue
        word_list = hanlp_examples[d]['word_list']
        word_locs = hanlp_examples[d]['word_locs']
        word_type_list = hanlp_examples[d]['word_type_list']
        #word_position_dict = hanlp_examples[d]['word_position_dict']
        doc_id = doc[0]
        events = doc[1]['recguid_eventname_eventdict_list']
        #arguments = doc[1]['ann_mspan2dranges']
        words_len = len(word_list)
        word_role_list = []
        for _ in range(words_len):
            word_role_list.append('NULL')

        words_table_label = np.zeros((words_len, words_len), dtype=int)
        event_dict_list = []
        finish_roles = {}
        for _ in range(len(events)):
            event_dict_list.append({})
            finish_roles[_]=[]
        current = {'EquityOverweight':{},'EquityUnderweight':{},'EquityRepurchase':{},'EquityFreeze':{},'EquityPledge':{}}
        exist_use = {}
        for i, word in enumerate(word_list):
            if i in exist_use.keys():
                continue
            event_flag = False
            for j, event in enumerate(events):
                if event_flag:
                    break
                type = event[1]
                if event_dict_list[j]=={}:
                    roles_order = event_roles[type]
                    for role in roles_order:
                        event_dict_list[j][role] = {}

                for role, arg in event[2].items():
                    if not arg:
                        arg='NA'
                    if i in exist_use.keys():
                        event_flag = True
                        break
                    if role in finish_roles[j]:
                        continue
                    if role not in current[type].keys():
                        current[type][role] = {}

                    if arg in current[type][role].keys():
                        loc = current[type][role][arg][1]
                        next_flag = False
                        if len(loc)==1:
                            loc = current[type][role][arg][1][0]
                            loc_length = len(word_locs[arg])
                            for m in range(1, loc_length):
                                if loc == word_locs[arg][-1]:
                                    next_loc = word_locs[arg][0]
                                else:
                                    next_loc = word_locs[arg][word_locs[arg].index(loc) + 1]
                                if next_loc not in exist_use.keys() or exist_use[next_loc] == role:
                                    event_dict_list[j][role][arg] = (next_loc, 'B-')
                                    finish_roles[j].append(role)
                                    current[type][role][arg][0] += 1
                                    current[type][role][arg][1] = [next_loc]
                                    exist_use[next_loc] = role
                                    next_flag = True
                                    break
                                loc = next_loc
                            if not next_flag:
                                event_dict_list[j][role][arg] = (loc, 'B-')
                                finish_roles[j].append(role)
                                current[type][role][arg][0] += 1
                        else:
                            dict_list = []
                            for k, event in enumerate(events):
                                for rrole, aarg in event[2].items():
                                    if aarg == arg and event_dict_list[k][rrole] != {}:
                                        dict_list.append(event_dict_list[k][rrole])
                            last_dict = dict_list[-1]
                            event_dict_list[j][role] = last_dict
                            finish_roles[j].append(role)
                            current[type][role][arg][0] += 1
                    elif word==arg:
                        event_dict_list[j][role][arg] = (i, 'B-')
                        finish_roles[j].append(role)
                        current[type][role][arg] = [1, [i]]
                        exist_use[i] = role
                        event_flag = True
                        break
                    elif arg!='NA' and arg.startswith(word):
                        curr = i
                        combines = []
                        combine_locs = []
                        event_flag = False
                        while (len(combines) < len(arg)):
                            if curr < len(word_list):
                                combines.append(word_list[curr])
                            else:
                                break
                            if len(combines) > len(arg):
                                break
                            combine_locs.append(curr)
                            if ''.join(combines) == arg and ''.join(combines) not in event_dict_list[j][role].keys():
                                for n, token in enumerate(combines):
                                    if combine_locs[n] == i:
                                        event_dict_list[j][role][token] = (i, 'B-')
                                        exist_use[i] = role
                                    else:
                                        if token not in event_dict_list[j][role].keys():
                                            event_dict_list[j][role][token] = (combine_locs[n], 'I-')
                                        else:
                                            event_dict_list[j][role][token + '-repeat'] = (combine_locs[n], 'I-')
                                        exist_use[combine_locs[n]] = role
                                finish_roles[j].append(role)
                                current[type][role][arg]=[1,combine_locs]
                                event_flag = True
                                break
                            while (curr < words_len -16  and word_list[curr] == word_list[curr + 1]):
                                curr += 1
                            curr += 1
                            while (curr < words_len - 16 and curr in exist_use.keys()):
                                if word_list[curr] == word_list[curr + 1]:
                                    curr += 1
                                else:
                                    break
                        if event_flag:
                            break

        filter_flag = False
        for j, event in enumerate(events):
            type = event[1]
            for role, arg in event[2].items():
                if not arg:
                    arg = 'NA'
                if role in finish_roles[j]:
                    continue
                if arg not in current[type][role].keys():
                    dict_list = []
                    for k, event in enumerate(events):
                        for rrole, aarg in event[2].items():
                            if aarg == arg and event_dict_list[k][rrole] != {}:
                                dict_list.append(event_dict_list[k][rrole])
                    try:
                        last_dict = dict_list[-1]
                    except Exception as e:
                        print(docid,e)
                        filter_doc.append(docid)
                        filter_flag = True
                        break
                    event_dict_list[j][role] = last_dict
                    finish_roles[j].append(role)
                elif arg == 'NA':
                    loc = current[type][role]['NA'][1][0]
                    loc_length = len(word_locs['NA'])
                    for m in range(0, loc_length):
                        if loc == word_locs['NA'][-1]:
                            next_loc = word_locs['NA'][0]
                        else:
                            next_loc = word_locs['NA'][word_locs['NA'].index(loc) + 1]
                        if exist_use[next_loc] == role:
                            event_dict_list[j][role]['NA'] = (next_loc, 'B-')
                            finish_roles[j].append(role)
                            current[type][role]['NA'][0] += 1
                            current[type][role]['NA'][1] = [next_loc]
                            break
                        loc = next_loc
            if filter_flag:
                break
        if filter_flag:
            continue

        for event_idx, event in enumerate(events):
            for role, arg in event[2].items():
                if arg and role not in finish_roles[event_idx]:
                    unfinish.append(arg)
        for _,dict in enumerate(event_dict_list):
            event_type = events[_][1]
            roles_order = event_roles[event_type]
            history = []
            for k, role_head in enumerate(roles_order):
                head = []
                tail = []
                if k == len(roles_order) - 1:
                    break
                role_tail = roles_order[k + 1]
                head_dict = dict[role_head]
                tail_dict = dict[role_tail]
                for head_item in head_dict.values():
                    pos_head = head_item[0]
                    pref_head = head_item[1]
                    if pos_head not in head:
                        head.append(pos_head)
                    for tail_item in tail_dict.values():
                        pos_tail = tail_item[0]
                        pref_tail = tail_item[1]
                        if pos_tail not in tail:
                            tail.append(pos_tail)
                        label = role_pairs[event_type][pref_head + role_head, pref_tail + role_tail]
                        t = words_table_label[pos_head][pos_tail]
                        words_table_label[pos_head][pos_tail] = label
                if k == 0:
                    history.append(head)
                history.append(tail)
            for m,ends in enumerate(history):
                if m == 0:
                    continue
                for n,starts in enumerate(history):
                    if n >= m:
                        break
                    for end in ends:
                        for start in starts:
                            if words_table_label[end][start] > 1:
                                a=1
                            else:
                                words_table_label[end][start] = 1

        example = {}
        example['id'] = d
        example['docid'] = docid
        example['tokens'] = word_list
        example['word_types'] = word_type_list
        example['words_table_label'] = words_table_label
        if docid not in filter_doc:
            examples.append(example)
        count.update(list(words_table_label.flatten()))
    label_weight = get_labels_weight(count)
    filter_name = os.path.join(args.cache_dir, file_type+'_examplefilters.txt')
    with open(filter_name, 'w') as fp:
        json.dump(filter_doc, fp)
    return examples,label_weight

def get_position(arg,role,word_list,word_role_list):
    for i,word in enumerate(word_list):
        if arg not in word:
            continue
        if word_role_list[i] == role:
            return i
        elif word_role_list[i] == 'NULL':
            word_role_list[i] = role
            return i

def same_role_one_event(event_arguments):
    if len(set(event_arguments.values())) != len(event_arguments):
        return True
    else:
        return False

def same_role_multi_events(events):
    len_roles =0
    args =[]
    for e in events:
        if same_role_one_event(e[2]):
            return True
        len_roles = len_roles + len(e[2])
        for v in e[2].values():
            args.append(v)
    if len_roles != len(set(args)):
        return True
    else:
        return False

def count_multi_roles_(datas):
    type_dict = { 'EquityHolder':'Person', 'LegalInstitution':'LInstit', 'FrozeShares':'Shares', 'TotalHoldingShares':'Shares',
                   'TotalHoldingRatio':'Ratio',  'StartDate':'Date', 'EndDate':'Date', 'UnfrozeDate':'Date',
                   'CompanyName': 'Person',  'ClosingDate':'Date', 'RepurchaseAmount':'Money', 'HighestTradingPrice':'Money',
                   'LowestTradingPrice':'Money', 'RepurchasedShares':'Shares', 'AveragePrice':'Money', 'TradedShares':'Shares',
                   'LaterHoldingShares':'Shares','ReleasedDate':'Date', 'Pledgee':'Person', 'TotalPledgedShares':'Shares',
                   'Pledger':'Person', 'PledgedShares':'Shares'}
    # Person/Company, LInstit, Shares, Ratio, Date, Money
    all_dict = {}
    for d,doc in enumerate(datas):
        mutil_sum = 0
        events = doc[1]['recguid_eventname_eventdict_list']
        new_events=[]
        types = []
        for event in events:
            dict={}
            list=[]
            event_type = event[1]
            event_arguments = event[2]
            roles_order = event_roles[event_type]
            new_event_arguments = []
            for r in roles_order:
                new_event_arguments.append((r, event_arguments[r]))
            for tup in new_event_arguments:
                role = tup[0]
                arg = tup[1]
                dict[arg] = type_dict[role]
                list.append(arg)
            new_events.append(list)
            types.append(dict)
        for i, event_list in enumerate(new_events):
            for m,argg in enumerate(event_list):
                if m == len(event_list)-1:
                    break
                next_argg = event_list[m+1]
                for j, oevent_list in enumerate(new_events):
                    if j == i:
                        continue
                    for n,arggg in enumerate(oevent_list):
                        if n == len(oevent_list)-1:
                            break
                        next_arggg = oevent_list[n+1]
                        if argg == 'NA' and arggg != 'NA':
                            mutil_sum = mutil_sum + 1
                        elif argg!='NA' and arggg == 'NA':
                            mutil_sum = mutil_sum + 1
                        elif argg == arggg and next_argg!=next_arggg and types[i][next_argg]==types[j][next_arggg]:
                            mutil_sum = mutil_sum+1
        all_dict[d] = mutil_sum

    sort_list = sorted(all_dict.items(), key=lambda x: x[1])
    print(sort_list)
    divid = int(len(sort_list) / 4)
    I = sort_list[0:divid]
    II = sort_list[divid:2 * divid]
    III = sort_list[2 * divid:3 * divid]
    IV = sort_list[3 * divid:len(sort_list)]
    print(I, '\n', II, '\n', III, '\n', IV)
    I_multi_roles_ids = get_docId(I)
    II_multi_roles_ids = get_docId(II)
    III_multi_roles_ids = get_docId(III)
    IV_multi_roles_ids = get_docId(IV)
    return (I_multi_roles_ids,II_multi_roles_ids,III_multi_roles_ids,IV_multi_roles_ids)

def count_sentences(datas):
    dict = {}
    for i,doc in enumerate(datas):
        events = doc[1]['recguid_eventname_eventdict_list']
        arguments = doc[1]['ann_mspan2dranges']
        sids = []
        for event in events:
            event_arguments = event[2]
            for arg in event_arguments.values():
                if not arg:
                    continue
                for pos in arguments[arg]:
                    if pos[0] not in sids:
                        sids.append(pos[0])

        dict[i] = int(len(set(sids))/len(events))
    sort_list = sorted(dict.items(), key=lambda x: x[1])
    print(sort_list)
    divid = int(len(sort_list) / 4)
    I = sort_list[0:divid]
    II = sort_list[divid:2 * divid]
    III = sort_list[2 * divid:3 * divid]
    IV = sort_list[3 * divid:len(sort_list)]
    print(I, '\n', II, '\n', III, '\n', IV)
    I_doc_ids = get_docId(I)
    II_doc_ids = get_docId(II)
    III_doc_ids = get_docId(III)
    IV_doc_ids = get_docId(IV)
    return (I_doc_ids,II_doc_ids,III_doc_ids,IV_doc_ids)

def get_docId(items):
    doc_ids = []
    for item in items:
        doc_ids.append(item[0])
    return doc_ids

def create_hanlp(datas):
    TOK = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
    TOK.dict_force = {'/股'}
    NER = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)

    filterwords = []
    with open('./data/filterwords.txt', 'r', encoding='utf-8') as f_stopword:
        filterword_datas = f_stopword.readlines()
        for filterword in filterword_datas:
            filterwords.append(filterword.strip())

    hanlp_examples = []
    filter_doc = []
    for doc in datas:
        repeat_num = {}
        exist_repeat_num = {}
        repeat_loc = {}
        word_list = []
        word_locs = {}
        word_type_list = []
        word_type_dict = {}
        word_position_dict = {}
        sentences = doc[1]['sentences']
        arguments = doc[1]['ann_mspan2dranges']
        mspan2guess_field = doc[1]['ann_mspan2guess_field']
        flag = False
        last_flag = False
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip('')
            if len(sentence) == 0 or '○○'in sentence:
                continue
            try:
                words = TOK(sentence.strip())
            except Exception as e:
                print(doc[0],e)
                filter_doc.append(doc[0])
                break
            isthrow,last_flag = if_throw(sent_idx, sentence,last_flag)
            if isthrow or last_flag:
                continue
            ners = NER(words)
            word_loc = 0
            new_words = []
            for w_idx,w in enumerate(words):
                if w=='股份' and w_idx!=0 and words[w_idx-1].isdigit():
                    new_words.append('股')
                    new_words.append('份')
                else:
                    new_words.append(w)
            words = new_words
            filter_idxs = []
            for i, word in enumerate(words):
                filter_idxs += filter_words(i, word, words)
                if i in filter_idxs or word in filterwords:
                    word_loc = word_loc + len(word)
                    continue
                word_type = 'Other'
                for arg, dranges in arguments.items():
                    for sent, ch_s, ch_e in dranges:
                        if sent_idx == sent and word_loc >= ch_s and word_loc + len(word) <= ch_e:
                            word_type = mspan2guess_field[arg]
                            break
                    if word_type != 'Other':
                        break
                index = len(word_list)
                word_list.append(word)
                if word not in word_locs.keys():
                    word_locs[word]=[]
                word_locs[word].append(index)
                word_type_list.append(word_type)
                word_type_dict[word] = word_type
                word_position_dict[word] = [[sent_idx, word_loc, word_loc + len(word)]]

                ner_type = get_ner(word,ners)
                repeatNums = repeat_nums(word,ner_type)
                if word in exist_repeat_num.keys():
                    exist_repeat_num[word] += 1
                    loc = len(word_list)-1
                    repeat_loc[loc] = word
                else:
                    if repeatNums > 0:
                        exist_repeat_num[word] = 1
                        loc = len(word_list) - 1
                        repeat_loc[loc] = word
                if repeatNums > 0:
                    if word not in repeat_num.keys() or (word in repeat_num.keys() and repeatNums > repeat_num[word]):
                        repeat_num[word] = repeatNums
                word_loc = word_loc + len(word)
        need_repeat = {}
        for word,num in repeat_num.items():
            exist = exist_repeat_num[word]
            if exist < num:
                need_repeat[word] = num-exist
        loc_list = list(repeat_loc.keys())
        loc_list.sort()
        con=[]
        current = -1
        for i,loc in enumerate(loc_list):
            if i <= current:
                continue
            current = i
            while(current+1<len(loc_list) and loc+1 == loc_list[current+1]):
                current += 1
            if current > i:
                con.append((loc,loc_list[current]))
        for pair in con:
            start = pair[0]
            end = pair[1]
            for i in range(start,end+1):
                word = repeat_loc[i]
                if word in need_repeat.keys():
                    num = need_repeat[word]
                else:
                    num = 0
                for j in range(num):
                    index = len(word_list)
                    word_list.append(word)
                    if word not in word_locs.keys():
                        word_locs[word] = []
                    word_locs[word].append(index)
                    word_type_list.append(word_type_dict[word])
                if word in need_repeat.keys():
                    del need_repeat[word]
        for word,num in need_repeat.items():
            for i in range(num):
                index = len(word_list)
                word_list.append(word)
                if word not in word_locs.keys():
                    word_locs[word] = []
                word_locs[word].append(index)
                word_type_list.append(word_type_dict[word])
        na_num = 15
        for j in range(na_num):
            index = len(word_list)
            word_list.append('NA')
            if 'NA' not in word_locs.keys():
                word_locs['NA'] = []
            word_locs['NA'].append(index)
            word_type_list.append('Other')

        hanlp_example = {}
        hanlp_example['docid'] = doc[0]
        hanlp_example['word_list'] = word_list
        hanlp_example['word_locs'] = word_locs
        hanlp_example['word_position_dict'] = word_position_dict
        hanlp_example['word_type_list'] = word_type_list
        hanlp_examples.append(hanlp_example)
    return hanlp_examples,filter_doc

def table_label_pos(list_head,list_tail,words_table_label,label):
    for head in list_head:
        for tail in list_tail:
            if words_table_label[head[0]][tail[0]] == label:
                return (head[0],tail[0])

def update_table_label(list_head,list_tail,words_table_label,label0,label1):
    for head in list_head:
        for tail in list_tail:
            if tail[0] == head[0]:
                continue
            if words_table_label[head[0]][tail[0]] == 0:
                words_table_label[head[0]][tail[0]] = label0
                return (head[0], tail[0])
    for head in list_head:
        for tail in list_tail:
            if words_table_label[head[0]][tail[0]] == 0:
                words_table_label[head[0]][tail[0]] = label0
                return (head[0], tail[0])
            elif words_table_label[head[0]][tail[0]] == 1:
                words_table_label[head[0]][tail[0]] = label1
                return (head[0], tail[0])

def repeat_nums(word,ner_type):
    if word == '元':
        ner_type = 'MONEY'
    if word == '股':
        ner_type = 'INTEGER'
    if word == '日':
        ner_type = 'DATE'
    if word.endswith('%'):
        ner_type = 'INTEGER'
    if not ner_type:
        return 0
    if 'DATE' in ner_type:
        return 3
    elif 'INTEGER' in ner_type or 'DECIMAL' in ner_type:
        return 3
    elif 'ORGANIZATION' in ner_type:
        return 3
    elif 'MONEY' in ner_type:
        return 3
    elif 'PERSON' in ner_type:
        return 3
    return 0

def repeat(nums,word_list,word_type_list,n_index):
    for i in range(nums):
        word_list.append(word_list[n_index])
        word_type_list.append(word_type_list[n_index])

def get_ner(word,ners):
    for ner in ners:
        if word == ner[0]:
            return ner[1]

def get_labels_weight(count):
    label_ids = dict(count)
    nums_labels = [(l, k) for k, l in sorted([(j, i) for i, j in count.items()], reverse=True)]
    size = len(nums_labels)
    if size % 2 == 0:
        median = (nums_labels[size // 2][1] + nums_labels[size // 2 - 1][1]) / 2
    else:
        median = nums_labels[(size - 1) // 2][1]

    weight_list = []
    for value_id in label_id_list:
        if value_id not in label_ids:
            weight_list.append(0)
        else:
            for label in nums_labels:
                if label[0] == value_id:
                    weight_list.append(median / label[1])
                    break
    return weight_list

def remove_repetion(llist):
    new_list = []
    for li in llist:
        if li not in new_list:
            new_list.append(li)
    return new_list


def load_and_cache_vocabs(examples, args):
    embedding_cache_path = os.path.join(args.cache_dir, 'embedding')
    if not os.path.exists(embedding_cache_path):
        os.makedirs(embedding_cache_path)

    # Build or load word vocab and embeddings.
    cached_word_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_word_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vocab_file):
        logger.info('Loading word vocab from %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'rb') as f:
            word_vocab = pickle.load(f)
    else:
        logger.info('Creating word vocab from dataset %s', args.dataset_name)
        word_vocab = build_text_vocab(examples)
        logger.info('Word vocab size: %s', word_vocab['len'])
        logging.info('Saving word vocab to %s', cached_word_vocab_file)
        with open(cached_word_vocab_file, 'wb') as f:
            pickle.dump(word_vocab, f, -1)

    cached_word_vecs_file = os.path.join(embedding_cache_path, 'cached_{}_word_vecs.pkl'.format(args.dataset_name))
    if os.path.exists(cached_word_vecs_file):
        logger.info('Loading word vecs from %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'rb') as f:
            word_vecs = pickle.load(f)
    else:
        logger.info('Creating bert vecs')
        word_vecs = load_bert_embedding(word_vocab['itos'])
        logger.info('Saving bert vecs to %s', cached_word_vecs_file)
        with open(cached_word_vecs_file, 'wb') as f:
            pickle.dump(word_vecs, f, -1)

    # Build vocab of word type tags.
    cached_wType_tag_vocab_file = os.path.join(
        embedding_cache_path, 'cached_{}_wType_tag_vocab.pkl'.format(args.dataset_name))
    if os.path.exists(cached_wType_tag_vocab_file):
        logger.info('Loading vocab of word type tags from %s', cached_wType_tag_vocab_file)
        with open(cached_wType_tag_vocab_file, 'rb') as f:
            wType_tag_vocab = pickle.load(f)
    else:
        logger.info('Creating vocab of word type tags.')
        wType_tag_vocab = build_wType_tag_vocab(examples, min_freq=0)
        logger.info('Saving word type tags  vocab, size: %s, to file %s', wType_tag_vocab['len'], cached_wType_tag_vocab_file)
        with open(cached_wType_tag_vocab_file, 'wb') as f:
            pickle.dump(wType_tag_vocab, f, -1)

    return word_vecs, word_vocab,wType_tag_vocab

def load_bert_embedding(word_list):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')
    word_vectors = []
    for word in word_list:
        if word == 'NA':
            word_vectors.append(torch.from_numpy(np.zeros(768, dtype=np.longlong)))
            continue
        tokenized_text = tokenizer.tokenize(word)
        if not tokenized_text:
            continue
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        token_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            last_hidden_states = model(token_tensor)[0]
        tokens_embedding = []
        for token_i in range(len(tokenized_text)):
            hidden_layers = []
            for layer_i in range(len(last_hidden_states)):
                vec = last_hidden_states[layer_i][0][token_i] #如果输入是单句不分块中间是0，因为只有一个维度，如果分块还要再遍历一次
                hidden_layers.append(vec)
            tokens_embedding.append(hidden_layers)
        summed_all_layers = [torch.sum(torch.stack(token_embedding),0) for token_embedding in tokens_embedding]
        word_vectors.append(summed_all_layers[0])
    return word_vectors

def _default_unk_index():
    return 1

def build_wType_tag_vocab(examples, vocab_size=1000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['word_types'])

    itos = []
    min_freq = max(min_freq, 0)

    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word in itos:
            continue
        itos.append(word)
    stoi = defaultdict()
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

def build_text_vocab(examples, vocab_size=1000000, min_freq=0):
    counter = Counter()
    for example in examples:
        counter.update(example['tokens'])

    itos = []
    min_freq = max(min_freq, 0)

    # sort by frequency, then alphabetically
    words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

    for word, freq in words_and_frequencies:
        if freq < min_freq or len(itos) == vocab_size:
            break
        if word in itos:
            continue
        itos.append(word)
    stoi = defaultdict(_default_unk_index)
    stoi.update({tok: i for i, tok in enumerate(itos)})

    return {'itos': itos, 'stoi': stoi, 'len': len(itos)}

class EE_Dataset(Dataset):
    def __init__(self, examples, args, word_vocab,wType_tag_vocab):
        self.examples = examples
        self.args = args
        self.word_vocab = word_vocab
        self.wType_tag_vocab = wType_tag_vocab
        self.convert_features()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        items = e['id'],e['token_ids'],e['wType_ids'], e['words_table_label']
        #print(idx)

        # items_tensor = tuple(torch.tensor(t) for t in items)
        # return items_tensor
        return items

    def convert_features(self):
        '''
        Convert tokens pos_tags, dependency_tags to ids.
        '''
        for i in range(len(self.examples)):
            self.examples[i]['token_ids'] = [self.word_vocab['stoi'][w] for w in self.examples[i]['tokens']]
            self.examples[i]['wType_ids'] = [self.wType_tag_vocab['stoi'][t] for t in self.examples[i]['word_types']]


def my_collate(batch):
    # from Dataset.__getitem__()
    id,token_ids,wType_ids,words_table_label = zip(
        *batch)  # from Dataset.__getitem__()

    # 全部转化为tensor类型，forward中只能处理tensor
    id = torch.tensor(id[0])
    token_ids = torch.tensor(token_ids[0])
    wType_ids = torch.tensor(wType_ids[0])
    words_table_label = torch.LongTensor(words_table_label[0])
    # event_args = event_args[0]
    # event_args = padding_3dim(event_args)
    # arg_labels = arg_labels[0]
    # arg_labels = padding_3dim_args(arg_labels)
    # for line in words_table_label:
    #     for ele in line:
    #         if ele > 0:
    #             print(11)

    return id,token_ids,wType_ids, words_table_label


def padding_3dim(source):
    events_tensor_list = []
    for i, event in enumerate(source):
        args_tensor_list = []
        for j, arg in enumerate(event):
            for k in range(max_same_tokens - len(arg)):
                arg.append(0)
            args_tensor_list.append(torch.Tensor(arg[0:max_same_tokens]))
        # 每个事件扩充到固定候选论元个数
        for m in range(max_args - len(args_tensor_list)):
            args_tensor_list.append(torch.zeros(max_same_tokens, dtype=torch.int32))
        # 只取固定的个数，多于固定个数的部分没有取
        args_tensor = torch.stack(args_tensor_list[0:max_args], dim=0)
        events_tensor_list.append(args_tensor)
    tgt = torch.stack(events_tensor_list, dim=0)
    return tgt

def padding_3dim_args(source):
    events_tensor_list = []
    for i, event in enumerate(source):
        # 每个事件扩充到固定候选论元个数
        for m in range(max_args - len(event)):
            event.append(0)
        # 只取固定的个数，多于固定个数的部分没有取
        args_tensor = torch.LongTensor(event[0:max_args])
        events_tensor_list.append(args_tensor)
    label = torch.stack(events_tensor_list, dim=0)

    return label