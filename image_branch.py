import torch
from torch import nn
import torch.nn.functional as F
from utils.bert import BertConfig
from utils import bert
import os
from model.faster_rcnn.resnet import resnet


class FeaturesMapper(nn.Module):
    def __init__(self, opt):
        super(FeaturesMapper, self).__init__()
        self.opt = opt
        self.mappper = FasterRCCNMapper(opt)
        self.transform = self.mappper.transform

        self.device = None

    def forward(self, im_data, gt_boxes=None):
        if self.device is not None:
            im_data = im_data.cuda(self.device)
            self.mappper.device = self.device

        output = self.mappper(im_data)
        return output


class FasterRCCNMapper(nn.Module):
    """ Self-attention layer for image branch
    """

    def __init__(self, opt):
        super(FasterRCCNMapper, self).__init__()
        self.opt = opt
        bert_config = BertConfig.from_json_file(opt.trans_cfg)
        self.layer = bert.BERTLayer(bert_config)
        self.mapping = nn.Linear(opt.img_dim, opt.final_dims)

        self.transform = None
        self.device = None

    def forward(self, x, y=None):
        # x: (batch_size, patch_num, img_dim)

        attention_mask = torch.ones(x.size(0), x.size(1))

        if self.device is not None:
            x = x.cuda(self.device)
            attention_mask = attention_mask.cuda(self.device)

        x = self.mapping(x)  # x: (batch_size, patch_num, final_dims)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        hidden_states = self.layer(x, extended_attention_mask)
        embed = torch.mean(hidden_states, 1)  # (batch_size, final_dims)
        codes = F.normalize(embed, p=2, dim=1)  # (N, C)
        return codes


class FasterRCNNExtractor(nn.Module):
    def __init__(self, opt):
        super(FasterRCNNExtractor, self).__init__()
        self.opt = opt
        classes = ['__background__']
        with open(os.path.join('Faster-R-CNN-with-model-pretrained-on-Visual-Genome/data/genome/1600-400-20',
                               'objects_vocab.txt')) as f:
            for object in f.readlines():
                classes.append(object.split(',')[0].lower().strip())

        print("classes", len(classes))

        self.detector = FasterRCNN(classes=classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.detector.create_architecture()
        # to check
        self.detector.RCNN_rpn = None

        load_name = os.path.join('Faster-R-CNN-with-model-pretrained-on-Visual-Genome',
                                 'load_dir/faster_rcnn_res101_vg.pth')
        checkpoint = torch.load(load_name)
        self.detector.load_state_dict(checkpoint['model'], strict=False)
        self._freeze_layers(n=opt.frozen_detector_blocks)

        print("number of layers", len([param for param in self.detector.parameters()]))

        self.mappper = FasterRCCNMapper(opt)
        self.transform = self.mappper.transform

        self.device = None

    def _freeze_layers(self, n=6):
        for param in self.detector.parameters():
            param.requires_grad = False

        all_params = list(self.detector.RCNN_base[4]) + \
                     list(self.detector.RCNN_base[5]) + \
                     list(self.detector.RCNN_base[6]) + \
                     list(self.detector.RCNN_top[0])

        if len(all_params) > n:
            for child in all_params[n:]:
                for param in child.parameters():
                    param.requires_grad = True

        if n == 0:
            for p in self.detector.RCNN_base[0].parameters(): p.requires_grad = True
            for p in self.detector.RCNN_base[1].parameters(): p.requires_grad = True

    def forward(self, im_data, gt_boxes):

        if self.device is not None:
            gt_boxes = gt_boxes.cuda(self.device)
            im_data = im_data.cuda(self.device)
            self.mappper.device = self.device

        x = self.detector(im_data=im_data, im_info=None, gt_boxes=gt_boxes, num_boxes=None, pool_feat=True)
        output = self.mappper(x)
        return output


class FasterRCNN(resnet):
    """ faster RCNN """

    def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False):
        resnet.__init__(self, classes, num_layers, pretrained, class_agnostic)

    def forward(self, im_data, im_info, gt_boxes, num_boxes=None, pool_feat=False):
        gt_boxes = gt_boxes.data

        rois = gt_boxes
        base_feat = self.RCNN_base(im_data)
        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        pooled_feat = self._head_to_tail(pooled_feat)

        batch_size = im_data.size(0)

        output = pooled_feat.view(batch_size, gt_boxes.shape[1], -1)

        return output
