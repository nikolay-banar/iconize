from __future__ import print_function
import numpy as np
import torch
from collections import OrderedDict
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, opt, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    src_final_embs = None
    tgt_final_embs = None
    tgt_lens = None

    # max_n_word = 0
    # for i, (images, input_ids, attention_mask, token_type_ids, lengths, ids) in enumerate(data_loader):
    #     max_n_word = max(max_n_word, max(lengths))

    for i, batch_data in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        # img_emb, cap_emb, cap_len, ids = model.forward_emb(20, batch_data, volatile=True)
        # img_emb, cap_emb = model.forward_emb(20, batch_data, volatile=True)
        with torch.no_grad():
            src_emb = model.infer_src(20, batch_data, volatile=True)
            # print(batch_data)
            tgt_emb = model.infer_tgt(20, batch_data, volatile=True)

        if src_final_embs is None and src_emb is not None:
            src_len = len(data_loader.dataset)
            if src_emb.dim() == 3:
                src_final_embs = np.zeros((src_len, src_emb.size(1), src_emb.size(2)))
            else:
                src_final_embs = np.zeros((src_len, src_emb.size(1)))

        if tgt_final_embs is None and tgt_emb is not None:
            tgt_len = len(data_loader.dataset)
            tgt_final_embs = np.zeros((tgt_len, tgt_emb.size(1)))
            tgt_lens = [0] * tgt_len
        # cache embeddings
        # print(img_emb.shape, len(data_loader.dataset), img_embs.shape)
        ids = batch_data['ids']
        if src_emb is not None:
            src_final_embs[ids] = src_emb.data.cpu().numpy().copy()

        if tgt_emb is not None:
            tgt_final_embs[ids] = tgt_emb.data.cpu().numpy().copy()
        # for j, nid in enumerate(ids):
        #     cap_lens[nid] = cap_len[j]

        # measure accuracy and record loss
        if src_emb is not None and tgt_emb is not None:
            model.forward_loss(0, src_emb, tgt_emb)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            .format(
                i, len(data_loader), batch_time=batch_time,
                e_log=str(model.logger)))
        # del images, input_ids, attention_mask, token_type_ids
    return src_final_embs, tgt_final_embs, tgt_lens


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p


def object_recall_k(actual, predicted, k):
    act_set = set(actual)
    # print(act_set)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result


def best_object_recall_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) > 0
    return float(result)


def get_object_ranks(actual, predicted):
    rank = np.isin(predicted, actual).nonzero()[0]
    return rank


def get_multilabel_rank(sorted_sim, best_ind=None, best_object=False):
    objects = sorted_sim.shape[0]

    r = {1: [], 5: [], 10: []}

    object_mean_ranks = []
    ranks = []
    for i in range(objects):
        r_i = get_object_ranks(actual=best_ind[i], predicted=sorted_sim[i])
        ranks.append(r_i)
        object_mean_ranks.append(r_i.mean())

    for key in r.keys():
        for i in range(objects):
            if best_object:
                r[key].append(best_object_recall_k(actual=best_ind[i], predicted=sorted_sim[i], k=key))
            else:
                r[key].append(object_recall_k(actual=best_ind[i], predicted=sorted_sim[i], k=key))

        r[key] = 100.0 * sum(r[key]) / len(r[key])

    medr = np.floor(np.median(object_mean_ranks)) + 1
    meanr = np.mean(object_mean_ranks) + 1

    # ranks = None

    return r[1], r[5], r[10], medr, meanr, ranks


def get_rank(sims, return_ranks=False, best_ind=None):
    best_ind = np.array(best_ind)
    best_ind = best_ind.reshape((best_ind.shape[0], 1))

    objects = sims.shape[0]
    print(objects)
    sorted_sim = np.argsort(sims, axis=1)[:, ::-1]
    print(sorted_sim)

    #     best_ind = np.concatenate(objects * [np.arange(objects).reshape((objects, 1))], axis=1)
    if best_ind is None:
        best_ind = np.arange(objects).reshape((objects, 1))
    print(best_ind)
    print("SHAPES ", sorted_sim.shape, best_ind.shape)
    ranks = np.argsort(sorted_sim == best_ind, axis=1)[:, -1]
    print(ranks)

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, sorted_sim[:, :10])
    else:
        return (r1, r5, r10, medr, meanr)


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
