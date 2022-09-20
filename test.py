import os
import time

import torch
import numpy as np

from multi_model import SAEM
from utils.evaluation import encode_data, get_multilabel_rank
from scipy.spatial.distance import cdist

import logging

import tensorboard_logger as tb_logger
from utils.preprocessing import convert_to_unicode, get_val_or_test
import pandas as pd
from utils.preprocessing import set_exp_params
import argparse


def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_name', default='',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=.0001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=10, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', default=True, action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--final_dims', default=256, type=int,
                        help='dimension of final codes.')
    parser.add_argument('--max_words', default=32, type=int,
                        help='maximum number of words in a sentence.')

    parser.add_argument("--trans_cfg", default='utils/t_cfg.json',
                        help="config file for bert for image transformer")

    parser.add_argument('--grad_acc_steps', default=1, type=int,
                        help='maximum number of words in a sentence.')

    parser.add_argument("--conf", default='t_cfg.json',
                        help="config for experiments")

    parser.add_argument("--visual_encoder",
                        default='fast',
                        type=str,
                        help="name of visual encoder")

    parser.add_argument("--text_encoder",
                        default='bert',
                        type=str,
                        help="name of visual encoder")

    parser.add_argument("--frozen_bert_layers",
                        default=12,
                        type=int,
                        help="number of frozen_bert_layers")

    parser.add_argument("--frozen_detector_blocks",
                        default=12,
                        type=int,
                        help="number of frozen_detector_layers")

    parser.add_argument("--exp_type",
                        default="",
                        type=str,
                        help="name of path to save experiment")

    parser.add_argument("--test_data",
                        default=None,
                        type=str,
                        help="define another test set")

    # images_path

    parser.add_argument("--images_path",
                        default=None,
                        type=str,
                        help="images_path")

    parser.add_argument("--combine_vec",
                        default="concat",
                        type=str,
                        help="how to combine sources: [concat, average, weighted_average]")

    parser.add_argument("--root",
                        default=None,
                        type=str,
                        help="root path")

    parser.add_argument("--test_labels",
                        default=None,
                        type=str,
                        help="define another new test labels")

    parser.add_argument("--precomp_target",
                        default=None,
                        type=str,
                        help="precomputed target labels")

    parser.add_argument('--no_labels', default=False, action="store_true", help="skip labels")

    return parser.parse_args()


def main():
    opt = parse_args()
    print(opt)

    exp_config = set_exp_params(opt)

    if opt.test_labels is not None:
        exp_config['tgt_txt_path'] = opt.test_labels

    if opt.test_data is not None:
        print(exp_config['src_txt_paths'])
        exp_config['src_txt_paths'] = [opt.test_data]
        print(exp_config['src_txt_paths'])

    if opt.images_path is not None:
        exp_config['images_path'] = opt.images_path

    if opt.root is not None:
        exp_config['root'] = opt.root

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    f = open(opt.logger_name + "opt.txt", 'w')
    f.write(opt.__str__())
    f.close()

    # Construct the model
    model = SAEM(opt)

    opt.resume = opt.model_name + '/model_best.pth.tar'
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        start_epoch = checkpoint['epoch']
        best_rsum = checkpoint['best_rsum']
        model.load_state_dict(checkpoint['model'])
        # Eiters is used to show logs as the continuation of another
        # training
        model.Eiters = checkpoint['Eiters']
        print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
              .format(opt.resume, start_epoch, best_rsum))
        test(opt, exp_config, model)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def test(opt, exp_config, model):
    test_set, tgt_set = get_val_or_test(exp_config, opt.batch_size, opt,
                                        txt_tokenizer=model.txt_enc.tokenize,
                                        im_transform=model.img_enc.transform if model.img_enc is not None else None,
                                        split="test")

    start = time.time()
    src_embs, _, _ = encode_data(model, test_set, opt, opt.log_step, logging.info)
    _, tgt_embs, _ = encode_data(model, tgt_set, opt, opt.log_step, logging.info)

    end = time.time()

    # print(tgt_embs.shape)

    print("calculate backbone time:", end - start)

    start = time.time()

    sims = 1 - cdist(src_embs, tgt_embs, metric='cosine')
    sorted_sim = np.argsort(sims, axis=1)[:, ::-1]
    end = time.time()
    print("calculate similarity time:", end - start)

    result_df = pd.DataFrame(sorted_sim[:, :10], columns=[f"top_{i + 1}" for i in range(10)])
    result_df = result_df.applymap(lambda x: {tgt_set.dataset.codes[x]: tgt_set.dataset.tgt[x]})
    if not opt.no_labels:

        best_ind = []
        for item in test_set.dataset.tgt_txt:
            uni_item = convert_to_unicode(item)
            uni_item = uni_item.split('[CODE]')
            uni_item = list(map(lambda x: tgt_set.dataset.txt2ind[x], uni_item))
            best_ind.append(uni_item)

        r1, r5, r10, medr, mean, ranks = get_multilabel_rank(sorted_sim=sorted_sim, best_ind=best_ind, best_object=True)
        print("ONE OBJECT", r1, r5, r10, medr, mean)
        r1, r5, r10, medr, mean, ranks = get_multilabel_rank(sorted_sim=sorted_sim, best_ind=best_ind)
        print(r1, r5, r10, medr, mean)
        result_df['label'] = [convert_to_unicode(item) for item in test_set.dataset.tgt_txt]
        result_df['ranks'] = ranks

    result_df.to_csv(opt.logger_name + 'results.csv')


if __name__ == '__main__':
    main()
