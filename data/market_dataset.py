import os.path
from data.base_dataset import BaseDataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import PIL
import pandas as pd
import torch
import math
import numpy as np
import xml.etree.ElementTree as ET
import imageio
class MarketDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 256))
        parser.set_defaults(structure_nc=3)
        parser.set_defaults(image_nc=3)
        return parser

    def initialize(self, opt):
        self.opt = opt
        default_phase = 'train' if self.opt.isTrain else 'test'
        phase = getattr(self.opt, 'phase', default_phase).lower()
        if phase not in ('train', 'val', 'test'):
            raise ValueError(
                "Unsupported phase '%s'. Expected train, val, or test." % phase
            )
        self.root = self.opt.dataroot
        self.phase = phase
        self.batchSize=opt.batch_size

        # prepare for image (image_dir), image_pair (name_pairs) and bone annotation (annotation_file)
        self.image_dir = os.path.join(self.root, self.phase)
        # self.bone_file = os.path.join(self.root, 'market-annotation-%s.csv' % self.phase)
        pairLst = os.path.join(self.root, 'market-pairs-%s.csv' % self.phase)
        self.name_pairs = self.init_categories(pairLst)
        self.annotation_path = os.path.join(self.root, 'Annotations')
        # self.annotation_file = pd.read_csv(self.bone_file, sep=':')
        # self.annotation_file = self.annotation_file.set_index('name')

        # load image size
        self.load_size = (opt.image_size, opt.image_size)

        # prepare for transformation
        transform_list=[]
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)))
        self.trans = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # prepare for source image Xs and target image Xt
        Xs_name, Xt_name = self.name_pairs[index]
        Xs_path = os.path.join(self.image_dir, Xs_name)
        Xs_annotation_path = os.path.join(
            self.annotation_path, os.path.splitext(Xs_name)[0] + '.xml'
        )
        Xt_path = os.path.join(self.image_dir, Xt_name)
        Xt_annotation_path = os.path.join(
            self.annotation_path, os.path.splitext(Xt_name)[0] + '.xml'
        )
        with open(Xs_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                Xs_WW, Xs_HH = image.size
        with open(Xt_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                Xt_WW, Xt_HH = image.size
        Xs = Image.open(Xs_path).convert('RGB')
        Xt = Image.open(Xt_path).convert('RGB')

        Xs = F.resize(Xs, self.load_size)
        Xt = F.resize(Xt, self.load_size)

        # Ps = self.obtain_bone(Xs_name)
        Xs = self.trans(Xs)
        # Pt = self.obtain_bone(Xt_name)
        Xt = self.trans(Xt)

        Xs_mask = self.obtain_mask(
            Xs_annotation_path, Xs_name, Xs_WW, Xs_HH
        )
        Xt_mask = self.obtain_mask(
            Xt_annotation_path, Xt_name, Xt_WW, Xt_HH
        )

        sample = {'Xs': Xs, 'Ps': Xs_mask, 'Xt': Xt, 'Pt': Xt_mask,
                  'Xs_path': Xs_name, 'Xt_path': Xt_name}

        if self.opt.use_z:
            height, width = Xt_mask.shape[-2:]
            z = torch.randn(self.opt.z_nc, height, width)
            sample['z'] = z

        return sample

    def init_categories(self, pairLst):
        pairs_file_train = pd.read_csv(pairLst)
        size = len(pairs_file_train)
        pairs = []
        print('Loading data pairs ...')
        for i in range(size):
            pair = [pairs_file_train.iloc[i]['from'], pairs_file_train.iloc[i]['to']]
            pairs.append(pair)

        print('Loading data pairs finished ...')
        return pairs

    def obtain_mask(self, annotation, name, WW, HH):
        root = ET.parse(annotation).getroot()
        objects = root.findall('object')

        H, W = self.load_size
        mask_sea = torch.zeros(1, H, W)
        mask_land = torch.zeros(1, H, W)
        mask_ship = torch.zeros(1, H, W)

        def fill_axis_box(mask, xmin, ymin, xmax, ymax):
            xmin = max(0.0, min(float(xmin), WW))
            xmax = max(0.0, min(float(xmax), WW))
            ymin = max(0.0, min(float(ymin), HH))
            ymax = max(0.0, min(float(ymax), HH))

            x1 = round(xmin / WW * W)
            x2 = round(xmax / WW * W)
            y1 = round(ymin / HH * H)
            y2 = round(ymax / HH * H)

            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))

            if x2 <= x1:
                x2 = min(x1 + 1, W)
            if y2 <= y1:
                y2 = min(y1 + 1, H)

            mask[:, y1:y2, x1:x2] = 1

        def fill_rotated_box(mask, cx, cy, bw, bh, angle):
            # roLabelImg angle is usually in radians.
            if abs(angle) > 2 * math.pi + 1e-6:
                angle = math.radians(angle)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            dx = bw / 2.0
            dy = bh / 2.0

            corners = [
                (-dx, -dy),
                (dx, -dy),
                (dx, dy),
                (-dx, dy),
            ]

            points = []
            for x, y in corners:
                px = cx + x * cos_a - y * sin_a
                py = cy + x * sin_a + y * cos_a

                px = px / WW * W
                py = py / HH * H

                px = max(0, min(px, W - 1))
                py = max(0, min(py, H - 1))
                points.append((px, py))

            mask_img = Image.new('L', (W, H), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(points, outline=1, fill=1)

            mask_np = np.array(mask_img, dtype=np.float32)
            # Accumulate all regions of the same class. Assignment here would
            # discard every earlier robndbox and retain only the last object.
            mask[0] = torch.maximum(mask[0], torch.from_numpy(mask_np))

        for obj in objects:
            category = obj.find("name").text.lower().strip()

            if category == 'ground':
                category = 'land'

            if category not in ['ship', 'land', 'sea']:
                continue

            if category == 'ship':
                target_mask = mask_ship
            elif category == 'land':
                target_mask = mask_land
            else:
                target_mask = mask_sea

            bbox = obj.find('bndbox')
            robndbox = obj.find('robndbox')

            if bbox is not None:
                xmin = float(bbox.find('xmin').text.strip())
                xmax = float(bbox.find('xmax').text.strip())
                ymin = float(bbox.find('ymin').text.strip())
                ymax = float(bbox.find('ymax').text.strip())
                fill_axis_box(target_mask, xmin, ymin, xmax, ymax)

            elif robndbox is not None:
                cx = float(robndbox.find('cx').text.strip())
                cy = float(robndbox.find('cy').text.strip())
                bw = float(robndbox.find('w').text.strip())
                bh = float(robndbox.find('h').text.strip())
                angle = float(robndbox.find('angle').text.strip())
                fill_rotated_box(target_mask, cx, cy, bw, bh, angle)

        masks = torch.cat((mask_ship, mask_land), 0)
        masks = torch.cat((masks, mask_sea), 0)

        return masks


    def __len__(self):
        return len(self.name_pairs)

    def name(self):
        return 'MarketDataset'
