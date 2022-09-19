import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import torch.nn.functional as F
from utils import loss_utils
from image_branch import FasterRCNNExtractor
from text_branch import TextEncoder
from image_branch import FeaturesMapper


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class WeightedCombiner(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(opt.n_vec))
        self.linear_layer = nn.Linear(opt.final_dims, opt.final_dims)

    def forward(self, x):
        norm_weights = F.softmax(self.weights)
        # print("norm_weights", norm_weights)
        x = norm_weights * x
        x = torch.sum(x, dim=2)
        x = self.linear_layer(x)
        return x


class SAEM(object):
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.txt_enc = TextEncoder(opt)
        self.gradient_accumulation_steps = opt.grad_acc_steps

        if opt.visual_encoder == "skip":
            self.img_enc = None
        elif opt.visual_encoder == "fast":
            self.img_enc = FasterRCNNExtractor(opt)
        elif opt.visual_encoder == "mapper":
            self.img_enc = FeaturesMapper(opt)
        else:
            raise Exception('Not Implemented')

        if opt.combine_vec == "concat":
            self.combiner = nn.Linear(opt.n_vec * opt.final_dims, opt.final_dims)
        elif opt.combine_vec == "weighted_average":
            self.combiner = WeightedCombiner(opt)
        elif opt.combine_vec == "average":
            self.combiner = nn.Linear(opt.final_dims, opt.final_dims)

        print("DEVICE COUNT", torch.cuda.device_count())

        if torch.cuda.device_count() > 1:
            self.txt_enc.device = torch.device("cuda:1")
            self.txt_enc = self.txt_enc.cuda(self.txt_enc.device)

            if self.img_enc is not None:
                self.img_enc.device = torch.device("cuda:0")
                self.img_enc = self.img_enc.cuda(self.img_enc.device)

            if self.combiner is not None:
                self.combiner = self.combiner.cuda(self.txt_enc.device)

            cudnn.benchmark = True
        elif torch.cuda.device_count() == 1:

            self.txt_enc.device = torch.device("cuda:0")
            self.txt_enc = self.txt_enc.cuda(self.txt_enc.device)

            if self.img_enc is not None:
                self.img_enc.device = torch.device("cuda:0")
                self.img_enc = self.img_enc.cuda(self.img_enc.device)

            if self.combiner is not None:
                self.combiner = self.combiner.cuda(self.txt_enc.device)
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = loss_utils.ContrastiveLoss(opt=opt,
                                                    margin=opt.margin,
                                                    max_violation=opt.max_violation)
        self.criterion2 = loss_utils.AngularLoss()
        self.criterion.device = self.txt_enc.device

        params = list(self.txt_enc.parameters())

        for name, param in self.txt_enc.named_parameters():
            if param.requires_grad:
                print("txt", name, param.shape)

        if self.img_enc is not None:
            params += list(self.img_enc.parameters())
            for name, param in self.img_enc.named_parameters():
                if param.requires_grad:
                    print("img", name, param.shape)

        if self.combiner is not None:
            params += list(self.combiner.parameters())
            for name, param in self.combiner.named_parameters():
                if param.requires_grad:
                    print("comb", name, param.shape)

        params = list(filter(lambda p: p.requires_grad, params))

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        state_dict = []

        if self.img_enc is not None:
            state_dict += [self.img_enc.state_dict()]

        state_dict += [self.txt_enc.state_dict()]

        if self.combiner is not None:
            state_dict += [self.combiner.state_dict()]

        return state_dict

    def load_state_dict(self, state_dict):
        print("STATE DICT", len(state_dict))
        if self.img_enc is not None:
            self.img_enc.load_state_dict(state_dict[0])
            self.txt_enc.load_state_dict(state_dict[1])
        else:
            self.txt_enc.load_state_dict(state_dict[0])

        if self.combiner is not None:
            self.combiner.load_state_dict(state_dict[-1])

    def train_start(self):
        """switch to train mode
        """
        if self.img_enc is not None:
            self.img_enc.train()
        self.txt_enc.train()
        if self.combiner is not None:
            self.combiner.train()

    def val_start(self):
        """switch to evaluate mode
        """
        if self.img_enc is not None:
            self.img_enc.eval()
        self.txt_enc.eval()
        if self.combiner is not None:
            self.combiner.eval()

    def infer_src(self, epoch, batch_data, volatile=False):

        images = batch_data["images"]
        src_txt = batch_data["txt"]
        boxes = batch_data["boxes"]
        output_code = []

        if src_txt is not None:
            for encoded in src_txt:
                output_code.append(self.txt_enc(**encoded))

        if images is not None:
            im = self.img_enc(images, boxes)

            if self.txt_enc.device is not None:
                im = im.cuda(self.txt_enc.device)

            output_code.append(im)

        if len(output_code) == 0:
            return None

        if self.opt.combine_vec == "concat":
            output_code = torch.cat(output_code, 1)  # lin part

        elif self.opt.combine_vec == "weighted_average":
            output_code = torch.stack(output_code, dim=2)  # lin part
        elif self.opt.combine_vec == "average":
            output_code = torch.stack(output_code, dim=2)
            output_code = torch.mean(output_code, dim=2)

        output_code = self.combiner(output_code)
        output_code = F.normalize(output_code, p=2, dim=1)
        return output_code

    def infer_tgt(self, epoch, batch_data, volatile=False):
        encoded = batch_data["tgt"]

        if encoded is not None:
            return self.txt_enc(**encoded)

        else:
            return None

    def forward_emb(self, epoch, batch_data, volatile=False):
        """Compute the image and caption embeddings
        """

        src_code = self.infer_src(0, batch_data)
        tgt_code = self.infer_tgt(0, batch_data)

        return src_code, tgt_code

    def forward_loss(self, epoch, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # alpha = 1
        if epoch > 20:
            alpha = 0
        else:
            alpha = 0.5 * (0.1 ** (epoch // 5))
        # alpha = 0
        loss1 = self.criterion(img_emb, cap_emb)
        loss2 = self.criterion2(img_emb, cap_emb)
        self.logger.update('Loss1', loss1.item(), img_emb.size(0))
        self.logger.update('Loss2', loss2.item(), img_emb.size(0))

        l2_reg = torch.tensor(0., dtype=torch.float)
        if self.img_enc is not None:

            if self.img_enc.device is not None:
                l2_reg = l2_reg.cuda(self.img_enc.device)

            no_decay = ['bias', 'gamma', 'beta']
            for n, p in self.img_enc.mappper.named_parameters():

                en = n.split('.')[-1]
                if en not in no_decay:
                    l2_reg += torch.norm(p)

        if self.txt_enc.device is not None:
            l2_reg = l2_reg.cuda(self.txt_enc.device)

        reg_loss = 0.01 * l2_reg

        return loss1 + reg_loss + alpha * loss2

    def train_emb(self, epoch, batch_data, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(epoch, batch_data)

        # measure accuracy and record loss
        # self.optimizer.zero_grad()
        loss = self.forward_loss(epoch, img_emb, cap_emb) / self.gradient_accumulation_steps
        loss.backward()
        # compute gradient and do SGD step
        if self.Eiters % self.gradient_accumulation_steps == 0:
            if self.grad_clip > 0:
                clip_grad_norm(self.params, self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
