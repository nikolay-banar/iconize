import os
import time
import shutil

import torch
import numpy as np
from utils.preprocessing import get_train, get_val_or_test, convert_to_unicode, set_exp_params
from multi_model import SAEM
from utils.evaluation import AverageMeter, LogCollector, encode_data, get_rank
from scipy.spatial.distance import cdist

import logging
import tensorboard_logger as tb_logger


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

    parser.add_argument("--conf", default='',
                        help="config for experiments")

    parser.add_argument("--visual_encoder",
                        default='fast',
                        type=str,
                        help="name of visual encoder")

    parser.add_argument("--text_encoder",
                        default='bert',
                        type=str,
                        help="name of text encoder")

    parser.add_argument("--frozen_bert_layers",
                        default=12,
                        type=int,
                        help="number of frozen_bert_layers")

    parser.add_argument("--frozen_detector_blocks",
                        default=33,
                        type=int,
                        help="number of frozen_vusual_layers")

    parser.add_argument("--combine_vec",
                        default="concat",
                        type=str,
                        help="how to combine sources: [concat, average, weighted_average]")

    parser.add_argument("--exp_type",
                        default="",
                        type=str,
                        help="name of path to save experiment")

    parser.add_argument("--precomp_target",
                        default=None,
                        type=str,
                        help="precomputed target labels")

    parser.add_argument('--no_labels', default=False, action="store_true", help="skip labels")

    return parser.parse_args()


def main():
    opt = parse_args()

    exp_config = set_exp_params(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    f = open(opt.logger_name + "opt.txt", 'w')
    f.write(opt.__str__())
    f.close()

    print(opt)
    # Construct the model
    model = SAEM(opt)

    # change to train
    print("BATCH", opt.batch_size)
    train_loader = get_train(exp_config, opt.batch_size, opt,
                             im_transform=model.img_enc.transform if model.img_enc is not None else None,
                             txt_tokenizer=model.txt_enc.tokenize)

    val_loaders = get_val_or_test(exp_config, opt.batch_size, opt,
                                  im_transform=model.img_enc.transform if model.img_enc is not None else None,
                                  txt_tokenizer=model.txt_enc.tokenize, split="dev")

    # optionally resume from a checkpoint
    if opt.resume:
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

            validate(opt, val_loaders, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    else:
        start_epoch = 0
        best_rsum = 0

    # Train the Model
    # best_rsum = 0
    best_values = ()
    for epoch in range(start_epoch, opt.num_epochs):
        print(opt.logger_name)
        print(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loaders)

        r1, r5, r10, medr, mean, rsum = validate(opt, val_loaders, model)
        print("DATASET", opt.data_name)

        is_best = rsum > best_rsum

        if is_best:
            best_rsum = rsum
            best_values = (r1, r5, r10, medr, mean)

        logging.info("src to text best values: %.1f, %.1f, %.1f, %.1f, %.1f" % best_values)

        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint_{}.pth.tar'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loaders):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(epoch, train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loaders, model, return_ranks=False):
    # compute the encoding for all the validation images and captions
    test_set, tgt_set = val_loaders

    best_ind = [tgt_set.dataset.txt2ind[convert_to_unicode(item)]
                for item in test_set.dataset.tgt_txt]

    start = time.time()
    src_embs, _, _ = encode_data(model, test_set, opt, opt.log_step, logging.info)
    _, tgt_embs, _ = encode_data(model, tgt_set, opt, opt.log_step, logging.info)

    end = time.time()

    print("calculate backbone time:", end - start)

    start = time.time()

    sims = 1 - cdist(src_embs, tgt_embs, metric='cosine')
    end = time.time()
    print("calculate similarity time:", end - start)

    best_ind = np.array(best_ind)
    best_ind = best_ind.reshape((best_ind.shape[0], 1))

    if return_ranks:
        (r1, r5, r10, medr, mean), (ranks, sorted_sim) = get_rank(sims, return_ranks, best_ind=best_ind)
    else:
        (r1, r5, r10, medr, mean) = get_rank(sims, return_ranks, best_ind=best_ind)

    logging.info("src to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, mean))

    currscore = r1 + r5 + r10

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    if return_ranks:
        return r1, r5, r10, medr, mean, currscore, ranks, sorted_sim
    else:
        return r1, r5, r10, medr, mean, currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""

    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
