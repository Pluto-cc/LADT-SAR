import math
import os.path
from pathlib import Path
from data.base_dataset import BaseDataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageDraw
import PIL
import pandas as pd
import torch
import xml.etree.ElementTree as ET


class MarketSSDD_Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        if is_train:
            parser.set_defaults(load_size=256)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(old_size=(256, 256))
        parser.set_defaults(structure_nc=3)
        parser.set_defaults(image_nc=3)
        parser.add_argument(
            '--annotation_dir',
            type=str,
            default='',
            help='directory containing SSDD XML annotations; defaults to dataroot/Annotations',
        )
        return parser

    def initialize(self, opt):
        self.opt = opt
        if (self.opt.isTrain):
            phase = 'train'
        else:
            phase = 'test'
        self.root = self.opt.dataroot
        self.phase = phase
        self.batchSize=opt.batch_size

        # prepare for image (image_dir), image_pair (name_pairs) and bone annotation (annotation_file)
        self.image_dir = os.path.join(self.root, self.phase)
        # self.bone_file = os.path.join(self.root, 'market-annotation-%s.csv' % self.phase)
        pairLst = os.path.join(self.root, 'market-pairs-%s.csv' % self.phase)
        self.name_pairs = self.init_categories(pairLst)
        self.annotation_path = opt.annotation_dir or os.path.join(self.root, 'Annotations')
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
        Xs_annotation_path = self.xml_path(Xs_name)
        Xt_path = os.path.join(self.image_dir, Xt_name)
        Xt_annotation_path = self.xml_path(Xt_name)
        with open(Xs_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
        Xs = Image.open(Xs_path).convert('RGB')
        Xt = Image.open(Xt_path).convert('RGB')

        Xs = F.resize(Xs, self.load_size)
        Xt = F.resize(Xt, self.load_size)

        # Ps = self.obtain_bone(Xs_name)
        Xs = self.trans(Xs)
        # Pt = self.obtain_bone(Xt_name)
        Xt = self.trans(Xt)

        Xs_mask=self.obtain_mask(Xs_annotation_path,Xs_name,WW,HH)
        Xt_mask=self.obtain_mask(Xt_annotation_path,Xt_name,WW,HH)

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

    def xml_path(self, image_name):
        path = Path(image_name)
        annotation = os.path.join(self.annotation_path, path.stem + '.xml')
        if not os.path.isfile(annotation):
            raise FileNotFoundError(
                f'Annotation not found for {image_name}: {annotation}'
            )
        return annotation


    def obtain_mask(self, annotation, name, WW, HH):
        root = ET.parse(annotation).getroot()
        objects = root.findall('.//object')
        H, W = self.load_size

        mask_images = {
            'ship': Image.new('L', (W, H), 0),
            'land': Image.new('L', (W, H), 0),
            'sea': Image.new('L', (W, H), 0),
        }

        def point(x, y):
            return (
                max(0, min(W - 1, round(float(x) / WW * W))),
                max(0, min(H - 1, round(float(y) / HH * H))),
            )

        def rotated_points(box):
            cx = float(box.findtext('cx'))
            cy = float(box.findtext('cy'))
            bw = float(box.findtext('w'))
            bh = float(box.findtext('h'))
            angle = float(box.findtext('angle'))
            if abs(angle) > 2 * math.pi:
                angle = math.radians(angle)
            ca, sa = math.cos(angle), math.sin(angle)
            corners = [(-bw / 2, -bh / 2), (bw / 2, -bh / 2),
                       (bw / 2, bh / 2), (-bw / 2, bh / 2)]
            return [point(cx + x * ca - y * sa, cy + x * sa + y * ca)
                    for x, y in corners]

        for obj in objects:
            category = (obj.findtext('name') or '').lower().strip()
            if category == 'ground':
                category = 'land'
            if category not in mask_images:
                continue

            draw = ImageDraw.Draw(mask_images[category])
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = float(bbox.findtext('xmin'))
                ymin = float(bbox.findtext('ymin'))
                xmax = float(bbox.findtext('xmax'))
                ymax = float(bbox.findtext('ymax'))
                draw.rectangle([point(xmin, ymin), point(xmax, ymax)], fill=1)
                continue

            robndbox = obj.find('robndbox')
            if robndbox is None:
                raise ValueError(f'{annotation}: {category} has no bndbox or robndbox')

            if robndbox.find('cx') is not None:
                draw.polygon(rotated_points(robndbox), fill=1)
            else:
                vertices = []
                for index in range(1, 5):
                    vertices.append(point(
                        robndbox.findtext(f'x{index}'),
                        robndbox.findtext(f'y{index}'),
                    ))
                draw.polygon(vertices, fill=1)

        def tensor(mask):
            values = torch.from_numpy(__import__('numpy').array(mask, dtype='float32'))
            return values.unsqueeze(0)

        # Keep the original channel order expected by the generator: ship, land, sea.
        return torch.cat((tensor(mask_images['ship']),
                          tensor(mask_images['land']),
                          tensor(mask_images['sea'])), dim=0)


    def __len__(self):
        return len(self.name_pairs)

    def name(self):
        return 'MarketDataset'
