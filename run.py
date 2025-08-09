# coding=utf-8
import argparse
import logging
import os
import random
import numpy as np
import torch
from datasets import load_datasets_and_vocabs
from models import GDEE
from trainer import *
from data_process import *

logger = logging.getLogger()

for h in logger.handlers:
    logger.removeHandler(h)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_path', type=str, default='./data', help='Dataset path.')
    parser.add_argument('--dataset_name', type=str, default='ChFinAnn',help='Choose ChFinAnn dataset.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to store output data.')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory to store cache data.')
    parser.add_argument('--classify_nums', type=int, default=122, help='Number of classification labels in adjacency matrix')
    parser.add_argument('--argument_nums', type=int, default=29, help='Number of classes of argument type.')
    parser.add_argument('--seed', type=int, default=2022, help='random seed for initialization')

    # Model parameters
    parser.add_argument('--token_embedding_dim', type=int, default=768, help='Dimension of token embedding')
    parser.add_argument('--word_type_embedding_dim', type=int, default=20, help='Dimension of word_type embeddings')

    # MLP
    parser.add_argument('--hidden_size', type=int, default=200,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm.')
    parser.add_argument('--num_mlps', type=int, default=2, help='Number of mlps in the last of model.')
    parser.add_argument('--final_hidden_size', type=int, default=100, help='Hidden size of mlps.')

    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for embedding.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=15.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20,
                        help="Log every X updates steps.")

    return parser.parse_args()


def check_args(args):
    logger.info(vars(args))


def main():
    # Setup logging
    for h in logger.handlers:
        logger.removeHandler(h)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Load datasets and vocabs
    train_dataset,dev_dataset,test_dataset,train_label_weight,dev_label_weight,test_label_weight,word_vocab, wType_tag_vocab,test_examples = load_datasets_and_vocabs(args)

    # Build Model
    model = GDEE(args,wType_tag_vocab['len'])
    model.to(args.device)

    # Train
    best_epoch = train(args,model,train_dataset,dev_dataset,train_label_weight,dev_label_weight)
    #  Evaluate
    model = torch.load('./output/model_' + str(best_epoch))
    evaluate_argument(args, test_dataset, model, test_label_weight, test_examples,best_epoch)

if __name__ == "__main__":
    main()

