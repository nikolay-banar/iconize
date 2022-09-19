import torch
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pandas as pd
from os.path import join as p_join
import torch.nn.functional as F
import json

from datetime import datetime

def parse_json(opt):
    with open(opt.conf, "r") as reader:
        text = reader.read()
        exp_config = json.loads(text)

    if "exp_type" in exp_config:
        opt.exp_type = exp_config["exp_type"]

    if "visual_encoder" in exp_config:
        opt.visual_encoder = exp_config["visual_encoder"]

    if "combine_vec" in exp_config:
        opt.combine_vec = exp_config["combine_vec"]

    if "text_encoder" in exp_config:
        opt.text_encoder = exp_config["text_encoder"]

    if "frozen_bert_layers" in exp_config:
        opt.frozen_bert_layers = int(exp_config["frozen_bert_layers"])

    if "frozen_detector_blocks" in exp_config:
        opt.frozen_detector_blocks = int(exp_config["frozen_detector_blocks"])

    if "batch_size" in exp_config:
        opt.batch_size = int(exp_config["batch_size"])

    if "grad_acc_steps" in exp_config:
        opt.grad_acc_steps = int(exp_config["grad_acc_steps"])

    if 'max_words' in exp_config:
        opt.max_words = int(exp_config["max_words"])

    # possibly delete
    if 'precomp_target' in exp_config:
        opt.precomp_target = exp_config["precomp_target"]

    # possibly change
    if 'precomp_target' in exp_config:
        opt.tgt_npy_path = exp_config["tgt_npy_path"]

    opt.n_vec = 0
    if exp_config['src_image_file'] or exp_config['precomp_image_file']:
        opt.n_vec += 1

    if exp_config['src_txt_paths']:
        opt.n_vec += len(exp_config['src_txt_paths'])


    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    if opt.logger_name:
        opt.logger_name = opt.logger_name + TIMESTAMP
    else:
        opt.logger_name = f'./runs/{opt.exp_type}/{exp_config["name"]}/log/' + TIMESTAMP

    if not opt.model_name:
        # ./runs/test/checkpoint
        opt.model_name = f'./runs/{opt.exp_type}/{exp_config["name"]}/checkpoint/'

    if not opt.data_name:
        opt.data_name = exp_config["name"]

    return exp_config


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


class MultiModalLoader(data.Dataset):
    """
       Load text-to-text data
       """

    def __init__(self, images_path=None, src_image_file=None, src_txt_paths=None, tgt_txt_path=None,
                 transform=None, precomp_image_file=None, bb_file=None, opt=None, tgt_npy_path=None):

        self.images_path = images_path
        self.src_image_file = src_image_file
        self.src_txt_paths = src_txt_paths
        self.tgt_txt_path = tgt_txt_path
        self.tgt_npy_path = tgt_npy_path
        self.transform = transform
        self.precomp_image_file = precomp_image_file
        self.bb_file = bb_file

        self.tgt = []
        self.txt_src = []

        self.boxes = None

        self.images = None

        self.image_files = None
        self.src_len = 0

        if not opt.no_labels:
            self.get_tgt()
        self.get_src()
        #

    def get_src(self):

        if self.bb_file:
            self.boxes = np.load(self.bb_file)
            self.boxes = torch.Tensor(self.boxes)

        if self.precomp_image_file:
            self.images = np.load(self.precomp_image_file)
            self.images = torch.Tensor(self.images)

        if self.src_image_file:
            with open(self.src_image_file, 'r') as f:
                self.image_files = [line.strip() for line in f]

        if self.src_txt_paths:
            for p in self.src_txt_paths:
                _, extension = os.path.splitext(p)
                if extension == '.npy':
                    src_array = np.load(p)
                    self.src_len = src_array.shape[0]
                    src_array = torch.Tensor(src_array)
                    self.txt_src.append(src_array)

                else:
                    with open(p, 'r') as f:
                        single_src = [line.strip() for line in f]
                        self.src_len = len(single_src)
                        self.txt_src.append(single_src)

        self.txt_src = list(zip(*self.txt_src)) if len(self.txt_src) != 0 else None

    def get_tgt(self):
        print("TARGET PATH:", self.tgt_txt_path)

        # _, extension = os.path.splitext(self.tgt_txt_path)
        if self.tgt_npy_path is not None:
            tgt_array = np.load(self.tgt_npy_path)
            tgt_array = torch.Tensor(tgt_array)
            self.tgt = [tgt_array[i] for i in range(tgt_array.shape[0])]

            self.tgt_txt = []

            with open(self.tgt_txt_path, 'r') as f:
                for line in f:
                    self.tgt_txt.append(line.strip())

        else:
            with open(self.tgt_txt_path, 'r') as f:
                for line in f:
                    self.tgt.append(line.strip())

            self.tgt_txt = self.tgt

        print("TGT ", len(self.tgt))

    def __getitem__(self, index):

        # print("GET ITEM",  self.images_path)
        if self.images_path:
            image = Image.open(os.path.join(self.images_path, self.image_files[index]))
            if image.mode != 'RGB':
                image = image.convert('RGB')

            if self.transform:
                image = self.transform(image)

            image = np.array(image)
            image = torch.Tensor(image)
            image = image.permute(2, 0, 1)
            image = image.unsqueeze(0)
            # print(image.shape)
        else:
            image = self.images[index] if self.images is not None else None

        src_texts = self.txt_src[index] if self.txt_src is not None else None
        tgt_text = self.tgt[index] if len(self.tgt) != 0 else None
        # print(images[0].shape)
        if self.boxes is not None:

            box = self.boxes[index]
            box = F.pad(box, (1, 0))

        else:
            box = None

        # print(box.shape)
        # print(box.shape)
        # print(box)
        # print("ONE ITEM SHAPE", src_texts[0].shape, src_texts[1].shape,tgt_text.shape)

        return image, src_texts, tgt_text, index, box

    def __len__(self):
        if self.images is not None:
            return len(self.images)
        elif self.image_files is not None:
            return len(self.image_files)
        elif self.txt_src is not None:
            return len(self.txt_src)
        else:
            return len(self.tgt)


class TargetLoader(MultiModalLoader):
    def __init__(self, tgt_txt_path, opt):
        super().__init__(tgt_txt_path=tgt_txt_path, opt=opt)
        self.txt2ind = None
        self.tgt_txt_path = tgt_txt_path
        self.get_tgt()

        self.tgt_array = None
        self.precomp_target = opt.precomp_target

        if self.precomp_target is not None:
            root_dir = os.path.dirname(self.tgt_txt_path)
            print("root_dir", root_dir)
            tgt_array = np.load(os.path.join(root_dir, self.precomp_target))
            tgt_array = torch.Tensor(tgt_array)
            self.tgt_array = [tgt_array[i] for i in range(tgt_array.shape[0])]
            # self.txt2ind = {str(txt): i for i, txt in enumerate(self.tgt_array)}

    def get_tgt(self):
        df = pd.read_csv(self.tgt_txt_path)
        self.tgt = list(df['txt'])
        self.codes = list(df['icon'])

        self.txt2ind = {txt: i for i, txt in enumerate(self.tgt)}

    def get_src(self):
        pass

    def __getitem__(self, index):
        image = None
        src_texts = None
        box = None

        tgt_text = self.tgt[index] if self.tgt_array is None else self.tgt_array[index]

        return image, src_texts, tgt_text, index, box

    def __len__(self):
        return len(self.tgt)


def collate_fn(batch, tokenize):
    images, src_texts, tgt_text, ids, boxes = zip(*batch)

    if images[0] is not None:
        if len(images[0].shape) == 4:
            max_width = max([i.shape[3] for i in images])
            max_height = max([i.shape[2] for i in images])
            images = [F.pad(i, [0, max_width - i.shape[3], 0, max_height - i.shape[2]]) for i in images]
            images = torch.cat(images, 0)
        elif len(images[0].shape) == 2:
            images = torch.stack(images, 0)

    else:
        images = None

    ids = np.array(ids)

    if boxes[0] is not None:
        boxes = torch.stack(boxes, 0)
        boxes[:, :, 0] = boxes[:, :, 0] + torch.range(0, ids.shape[0] - 1).view(ids.shape[0], -1)
    else:
        boxes = None

    if tokenize is not None:
        enc_src = [tokenize(list(src_text)) for src_text in zip(*src_texts)] if src_texts[0] is not None else None
    else:
        enc_src = [{"input_ids": torch.stack(src_text, 0)} for src_text in zip(*src_texts)] if src_texts[
                                                                                                   0] is not None else None

    if tokenize is not None:
        enc_tgt = tokenize(list(tgt_text)) if tgt_text[0] is not None else None
    else:

        enc_tgt = {"input_ids": torch.stack(tgt_text, 0)}

    return {"images": images, 'txt': enc_src, 'tgt': enc_tgt, 'ids': ids, "boxes": boxes}


def dict_from_config(config, split):
    root = config["root"]

    experiment = {'images_path': None,
                  'src_image_file': None,
                  'bb_file': None,
                  'precomp_image_file': None,
                  'tgt_npy_path': None,
                  'src_txt_paths': None}

    if config["images_path"]:
        experiment['images_path'] = p_join(root, config["images_path"])

    if config['src_image_file']:
        experiment['src_image_file'] = p_join(split, config["src_image_file"])
        experiment['src_image_file'] = p_join(root, experiment["src_image_file"])

    if config["bb_file"]:
        experiment['bb_file'] = p_join(split, config["bb_file"])
        experiment['bb_file'] = p_join(root, experiment["bb_file"])

    if config['precomp_image_file']:
        experiment['precomp_image_file'] = p_join(split, config["precomp_image_file"])
        experiment['precomp_image_file'] = p_join(root, experiment["precomp_image_file"])

    experiment['tgt_txt_path'] = p_join(split, config["tgt_txt_path"])
    experiment['tgt_txt_path'] = p_join(root, experiment["tgt_txt_path"])

    if "tgt_npy_path" in config and config["tgt_npy_path"]:
        experiment['tgt_npy_path'] = p_join(split, config["tgt_npy_path"])
        experiment['tgt_npy_path'] = p_join(root, experiment["tgt_npy_path"])

    if config['src_txt_paths']:
        experiment['src_txt_paths'] = []
        for item in list(config['src_txt_paths']):
            item = p_join(split, item)
            item = p_join(root, item)
            experiment['src_txt_paths'].append(item)

    return experiment


def get_train(config, batch_size, opt, txt_tokenizer=None, im_transform=None):
    experiment = dict_from_config(config, split="train")

    experiment['opt'] = opt
    experiment['transform'] = im_transform

    ds = MultiModalLoader(**experiment)
    return torch.utils.data.DataLoader(dataset=ds,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=True,
                                       collate_fn=lambda x: collate_fn(x, txt_tokenizer))


def get_val_or_test(config, batch_size, opt, txt_tokenizer=None, im_transform=None, split="dev"):
    experiment = dict_from_config(config, split=split)
    experiment['opt'] = opt
    experiment['transform'] = im_transform

    ds = MultiModalLoader(**experiment)

    val_set = torch.utils.data.DataLoader(dataset=ds,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          collate_fn=lambda x: collate_fn(x, txt_tokenizer))

    tgt_loader = TargetLoader(tgt_txt_path=p_join(config["root"], config["target_loader"]), opt=opt)
    tgt_set = torch.utils.data.DataLoader(dataset=tgt_loader,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          collate_fn=lambda x: collate_fn(x, txt_tokenizer))

    return val_set, tgt_set
