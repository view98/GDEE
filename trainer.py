import logging
import random
import numpy as np
import torch,copy
import gc,math,itertools
import torch.nn.functional as F
from sklearn.metrics import f1_score,precision_score,recall_score,precision_recall_fscore_support
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange

from datasets import *
torch.set_printoptions(profile="full")

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_input_from_batch(batch):
    inputs = {
              'token_ids':batch[1],
              'wType_ids':batch[2]}

    words_table_label = batch[3].view(-1)

    return  batch[0],inputs,words_table_label

def get_collate_fn():
    return my_collate

def train(args,model,train_dataset,dev_dataset,train_label_weight,dev_label_weight):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn()
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    train_label_weight = torch.tensor(train_label_weight).to(args.device)
    max_f1 = 0
    max_epoch = 0
    for epoch,_ in enumerate(train_iterator):
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            id,inputs,words_table_label = get_input_from_batch(batch)
            logit = model(**inputs)
            loss = F.cross_entropy(logit, words_table_label,weight=train_label_weight)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info("  train_loss: %s", str((tr_loss - logging_loss) / args.logging_steps))
                    logging_loss = tr_loss

        #f1 = evaluate1(args,dev_dataset,model,dev_label_weight,dev_examples,epoch)
        eval_loss,f1 = evaluate_token(args,dev_dataset,model,dev_label_weight)

        if f1 > max_f1:
            max_f1 = f1
            max_epoch = epoch
            torch.save(model, './output/model_' + str(epoch))

    tb_writer.close()
    logger.info("***** Best result is at epoch %s *****",str(max_epoch))
    return max_epoch

def evaluate_argument(args, eval_dataset, model,test_label_weight,test_examples,best_epoch):
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn()
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,collate_fn=collate_fn)
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    out_label_ids = []
    final_preds = []

    test_label_weight = torch.tensor(test_label_weight).to(args.device)
    all_records = {}
    doc_id = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            id,inputs,labels = get_input_from_batch(batch)
            id = id.detach().cpu().tolist()

            logit = model(**inputs)
            loss = F.cross_entropy(logit, labels,weight=test_label_weight)

            eval_loss += loss.mean().item()
            nb_eval_steps += 1
            preds = np.argmax(logit.detach().cpu().numpy(),axis=1)
            final_preds += preds.tolist()
            out_label_ids += labels.detach().cpu().tolist()

            tokens = test_examples[doc_id]['tokens']
            event_records = decode_event(labels.detach().cpu().tolist(), preds.tolist(), tokens)

            all_records[id] = event_records
            doc_id += 1

    with open('./output/result'+str(best_epoch)+'.json', 'w') as f2:
        json.dump(all_records, f2)
    get_gold_event_records(best_epoch)

def decode_event(labels,preds,tokens):
    n = int(math.sqrt(len(labels)))
    labels = torch.tensor(labels).reshape(n, n).tolist()
    preds = torch.tensor(preds).reshape(n, n).tolist()

    event_records = {}
    for event_type in role_pairs.keys():
        if event_type in ['None','Trace']:
            continue
        event_records[event_type] = { 'roles': {}, 'locs': {},'events': {}}
        start_id = list(role_pairs[event_type].values())[0]
        end_id = list(role_pairs[event_type].values())[-1]
        for role_id in range(int((end_id-start_id+1)/4)):
            roles = []
            role_tokens = []
            locs = []
            for i in range(n):
                for j in range(n):
                    if labels[i][j] < 1:
                        continue
                    if preds[i][j] >= (start_id+role_id*4) and preds[i][j] <= (start_id+role_id*4+3):
                        role1, role2 = role_pairs2id[preds[i][j]]
                        roles.append([[role1],[role2]])
                        role_tokens.append([[tokens[i]],[tokens[j]]])
                        locs.append([[i],[j]])
            del_ids = []
            if len(locs)==0:
               break
            while(existCombine(roles,locs)):
                combine_flag = False
                for k,loc in enumerate(locs):
                    for m,loc_1 in enumerate(locs):
                        if k>=m:
                            continue
                        if loc[0] == loc_1[0] and (roles[k][1][0].startswith('B-') and roles[m][1][0].startswith('I-')):
                            last = loc[1][-1]
                            while (last < n-1 and tokens[last] == tokens[last + 1]):
                                last += 1
                            if (last + 1) == loc_1[1][-1]:
                                locs[k][1] += locs[m][1]
                                role_tokens[k][1] += role_tokens[m][1]
                                roles[k][1] += roles[m][1]
                                if m not in del_ids:
                                    del_ids.append(m)
                                if k in del_ids:
                                    del_ids.remove(k)
                                combine_flag = True
                        elif loc[1] == loc_1[1] and (roles[k][0][0].startswith('B-') and roles[m][0][0].startswith('I-')):
                            first = loc[0][-1]
                            while (first < n-1 and tokens[first] == tokens[first + 1]):
                                first += 1
                            if (first + 1) == loc_1[0][-1]:
                                locs[k][0] += locs[m][0]
                                role_tokens[k][0] += role_tokens[m][0]
                                roles[k][0] += roles[m][0]
                                if m not in del_ids:
                                    del_ids.append(m)
                                if k in del_ids:
                                    del_ids.remove(k)
                                combine_flag = True
                if not combine_flag:
                    break
                cur_num = 0
                del_ids.sort()
                for del_id in del_ids:
                    locs.remove(locs[del_id-cur_num])
                    role_tokens.remove(role_tokens[del_id-cur_num])
                    roles.remove(roles[del_id-cur_num])
                    cur_num += 1
                del_ids = []
                if not combine_flag:
                    break

            event_num = len(event_records[event_type]['events'])
            if event_num == 0:
                for i, role_token in enumerate(role_tokens):
                    event_records[event_type]['events'][event_num] = [role_tokens[i]]
                    event_records[event_type]['roles'][event_num] = [roles[i]]
                    event_records[event_type]['locs'][event_num] = [locs[i]]
                    event_num += 1
            else:
                for event_id in range(event_num):
                    loc_last = event_records[event_type]['locs'][event_id][-1]
                    flag = False
                    exist_locs = event_records[event_type]['locs'][event_id][:]
                    for i, loc in enumerate(locs):
                        if loc[0] == loc_last[1]:
                            trace_flag = False
                            for exist_loc in exist_locs:
                                if preds[loc[1][0]][exist_loc[0][0]] == 0 or preds[loc[1][0]][exist_loc[1][0]] == 0:
                                    trace_flag = True
                                    break
                            if len(exist_locs) > 0 and not trace_flag:
                                if not flag:
                                    event_records[event_type]['locs'][event_id].append(locs[i])
                                    event_records[event_type]['events'][event_id].append(role_tokens[i])
                                    event_records[event_type]['roles'][event_id].append(roles[i])
                                    flag = True
                                elif flag:
                                    num = len(event_records[event_type]['events'])
                                    event_records[event_type]['locs'][num] = copy.deepcopy(
                                        event_records[event_type]['locs'][event_id])
                                    event_records[event_type]['events'][num] = copy.deepcopy(
                                        event_records[event_type]['events'][event_id])
                                    event_records[event_type]['roles'][num] = copy.deepcopy(
                                        event_records[event_type]['roles'][event_id])
                                    event_records[event_type]['locs'][num][-1] = locs[i]
                                    event_records[event_type]['events'][num][-1] = role_tokens[i]
                                    event_records[event_type]['roles'][num][-1] = roles[i]
    return event_records

def existCombine(roles,locs):
    for k, loc in enumerate(locs):
        for m, loc_1 in enumerate(locs):
            if k >= m:
                continue
            if loc[0] == loc_1[0] and not (roles[k][1][0].startswith('B-') and roles[m][1][0].startswith('B-')):
                return True
            elif loc[1] == loc_1[1] and not (roles[k][0][0].startswith('B-') and roles[m][0][0].startswith('B-')):
                return True
    return False

def evaluate_token(args, eval_dataset, model,test_label_weight):
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn()
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,collate_fn=collate_fn)
    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    out_label_ids = []
    final_preds = []

    test_label_weight = torch.tensor(test_label_weight).to(args.device)
    with torch.no_grad():
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            id,inputs,labels = get_input_from_batch(batch)

            logit = model(**inputs)
            loss = F.cross_entropy(logit, labels,weight=test_label_weight)

            eval_loss += loss.mean().item()
            nb_eval_steps += 1
            preds = np.argmax(logit.detach().cpu().numpy(),axis=1)
            final_preds += preds.tolist()
            out_label_ids += labels.detach().cpu().tolist()

    eval_loss = eval_loss / nb_eval_steps

    logger.info(" eval loss: %s", str(eval_loss))
    f1 = prf_compute(final_preds,out_label_ids)
    return eval_loss,f1

def prf_compute(preds, labels):
    results = {}
    preds_type = {}
    labels_type ={}
    for event_type in role_pairs.keys():
        if event_type in ['None','Trace']:
            continue
        results[event_type] = {}
        preds_type[event_type] = []
        labels_type[event_type] = []
    for i,label in enumerate(labels):
        if label >=2 and label <= 21:
            preds_type['EquityOverweight'].append(preds[i])
            labels_type['EquityOverweight'].append(label)
        elif label >= 22 and label <= 41:
            preds_type['EquityUnderweight'].append(preds[i])
            labels_type['EquityUnderweight'].append(label)
        elif label >= 42 and label <= 61:
            preds_type['EquityRepurchase'].append(preds[i])
            labels_type['EquityRepurchase'].append(label)
        elif label >= 62 and label <= 89:
            preds_type['EquityFreeze'].append(preds[i])
            labels_type['EquityFreeze'].append(label)
        elif label >= 90 and label <= 121:
            preds_type['EquityPledge'].append(preds[i])
            labels_type['EquityPledge'].append(label)

    aver_f1 = 0
    for event_type in role_pairs.keys():
        if event_type in ['None','Trace']:
            continue
        pre,recall,f1,support = precision_recall_fscore_support(labels_type[event_type],preds_type[event_type],average='micro')
        logger.info("************%s************", event_type)
        logger.info("  pre = %s", str(pre))
        logger.info("  recall = %s", str(recall))
        logger.info("  f1 = %s", str(f1))
        aver_f1 += f1

    logger.info("  aver_f1 = %s", str(aver_f1 / (len(role_pairs)-2)))
    return aver_f1 / (len(role_pairs)-2)

def get_gold_event_records(epoch):
    doc_gold = {}
    result = {}
    for event_type in role_pairs.keys():
        if event_type in ['None', 'Trace']:
            continue
        if event_type not in result.keys():
            result[event_type] = {'pre': 0, 'recall': 0, 'f1': 0, 'correct_num': 0, 'pred_num': 0, 'gold_num': 0}


    fp1 = open('./output/gold.json', 'r', encoding='utf-8-sig')
    datas = json.load(fp1)
    with open('./output/result'+str(epoch)+'.json','r') as fp:
        doc_preds = json.load(fp)
        for i,doc_pred in doc_preds.items():
            doc_id = int(i)
            doc_gold[doc_id] = datas[i]
            for gold_event_type,event_info in doc_gold[doc_id].items():
                if len(event_info['events']) == 0:
                    continue
                for gold_event_id,event in event_info['events'].items():
                    for arg in event:
                        result[gold_event_type]['gold_num'] += 1

            for event_type in role_pairs.keys():
                if event_type in ['None','Trace']:
                    continue
                if len(doc_pred[event_type]['events']) == 0:
                    continue
                match_gold_event_list = []
                match_pred_event_list = []
                event_match_info = {}
                for event_id,event in doc_pred[event_type]['events'].items():
                    if event_id not in event_match_info.keys():
                        event_match_info[event_id] = {}

                    arg_list = []
                    for arg_id,arg in enumerate(event):
                        arg1_list,arg2_list = arg[0],arg[1]
                        role1_list = doc_pred[event_type]['roles'][event_id][arg_id][0]
                        role2_list = doc_pred[event_type]['roles'][event_id][arg_id][1]
                        process_entity(role1_list,arg1_list,arg_list)
                        process_entity(role2_list,arg2_list,arg_list)

                    for (role_pred,arg_pred) in arg_list:
                        for gold_etype,gold_events in doc_gold[doc_id].items():
                            if len(gold_events['events']) == 0 or not (event_type == gold_etype):
                                continue
                            for gold_eid, gold_args in gold_events['events'].items():
                                for arg_id,gold_arg_pair in enumerate(gold_args):
                                    if gold_eid not in event_match_info[event_id].keys():
                                        event_match_info[event_id][gold_eid] = 0
                                    gold_arg = gold_arg_pair[0][0]
                                    gold_role = gold_events['roles'][gold_eid][arg_id][0][0][2:]
                                    if role_pred == gold_role and ''.join(arg_pred) == gold_arg:
                                        event_match_info[event_id][gold_eid] += 1
                                        break

                if len(event_match_info) == 0:
                    continue

                gold_events_info = doc_gold[doc_id]
                event_num = 5
                for _ in range(event_num):
                    max_match_num = 0
                    max_gold_event_id = 0
                    max_pred_event_id = 0
                    for event_id, gold_info in event_match_info.items():
                        if event_id in match_pred_event_list:
                            continue
                        for gold_event_id,match_num in gold_info.items():
                            if gold_event_id in match_gold_event_list:
                                continue
                            if match_num > max_match_num:
                                max_match_num = match_num
                                max_gold_event_id = gold_event_id
                                max_pred_event_id = event_id
                    if max_match_num > 0:
                        result[event_type]['correct_num'] += max_match_num
                        match_gold_event_list.append(max_gold_event_id)
                        match_pred_event_list.append(max_pred_event_id)
                    else:
                        for gold_etype,gold_events in gold_events_info.items():
                            if len(gold_events['events']) == 0:
                                continue
                            if max_match_num == 0 and gold_etype == event_type and len(gold_events['events']) <= event_num and len(match_pred_event_list) <= event_num:
                                for event_id, event in doc_pred[event_type]['events'].items():
                                    if event_id not in match_pred_event_list:
                                        match_pred_event_list.append(max_pred_event_id)

                match_pred_event_list = list(set(match_pred_event_list))
                for event_id,event in doc_pred[event_type]['events'].items():
                    if event_id in match_pred_event_list:
                        arg_list = []
                        for arg_id,arg in enumerate(event):
                            arg1_list,arg2_list = arg[0],arg[1]
                            role1_list = doc_pred[event_type]['roles'][event_id][arg_id][0]
                            role2_list = doc_pred[event_type]['roles'][event_id][arg_id][1]
                            process_entity(role1_list,arg1_list,arg_list)
                            process_entity(role2_list,arg2_list,arg_list)
                        result[event_type]['pred_num'] += len(arg_list)
    fp1.close()
    aver_f1 = 0
    for event_type in role_pairs.keys():
        if event_type in ['None', 'Trace']:
            continue
        pre = result[event_type]['correct_num'] / result[event_type]['pred_num']
        recall = result[event_type]['correct_num'] / result[event_type]['gold_num']
        f1 = 2*pre*recall/(pre+recall)
        aver_f1 += f1
        logger.info("************%s************", event_type)
        logger.info("  pre = %s", str(pre))
        logger.info("  recall = %s", str(recall))
        logger.info("  f1 = %s", str(f1))
        result[event_type]['pre'] = pre
        result[event_type]['recall'] = recall
        result[event_type]['f1'] = f1

    logger.info("*******aver f1: %s************", str(aver_f1 / 5))
    with open('./output/result_final.json', 'w') as fp:
        json.dump(result,fp)

def get_roles(role_list,args,true_role_list):
    num = min(len(args),len(role_list))
    for i in range(num):
        true_role = role_list[i][2:]
        if (true_role,args[i]) not in true_role_list:
            true_role_list.append((true_role,args[i]))

def process_entity(role_list,args,arg_list):
    for i,role in enumerate(role_list):
        true_role = role[2:]
        if (true_role,args) not in arg_list:
            arg_list.append((true_role,args))
