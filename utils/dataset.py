import json
import os
import torch
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, HistogramNormalized,RandCoarseShuffled,RandRotated,RandZoomd,
                              Resized, ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer



class QaTa(Dataset):

    def __init__(self, csv_path=None, root_path=None,mode='train',image_size=[224,224]):

        super(QaTa, self).__init__()

        self.mode = mode
        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)

        self.image_list = list(self.data['path'])
        self.caption_list = list(self.data['text'])

        if mode == 'train':
            self.image_list = self.image_list[:int(0.8*len(self.image_list))]
            self.caption_list = self.caption_list[:int(0.8*len(self.caption_list))]
        elif mode == 'valid':
            self.image_list = self.image_list[int(0.9*len(self.image_list)):]
            self.caption_list = self.caption_list[int(0.9*len(self.caption_list)):]
        else:
            pass   # for mode is test
            # self.image_list = self.image_list[int(0.8*len(self.image_list)):]
            # self.caption_list = self.caption_list[int(0.8*len(self.caption_list)):]

        self.root_path = root_path
        self.image_size = image_size

        # url = '/dssg/home/ai2010812935/zhongyi/code/lib/Bio_ClinicalBERT'
        url = "/dssg/home/ai2010812935/zhongyi/code/lib/BiomedVLP-CXR-BERT-specialized"
        self.tokenizer = AutoTokenizer.from_pretrained(url, trust_remote_code=True)

    def __len__(self):

        return len(self.image_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        image = os.path.join(self.root_path,'Images',self.image_list[idx].replace('mask_',''))
        gt = os.path.join(self.root_path,'GTs', self.image_list[idx])
        caption = self.caption_list[idx]


        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=24, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')

        token,mask = token_output['input_ids'],token_output['attention_mask']

        data = {'image':image, 'gt':gt, 'token':token, 'mask':mask}
        data = trans(data)

        image,gt,token,mask = data['image'],data['gt'],data['token'],data['mask']

        gt = torch.where(gt==255,1,0)

        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)}  #不知道为什么多了一个维度

        return ([image, text], gt)

    def transform(self,image_size):

        if self.mode == 'train':

            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                RandZoomd(['image','gt'],min_zoom=0.95,max_zoom=1.2,mode=["bicubic","nearest"],prob=0.1),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                HistogramNormalized(['image']),
                # RandCoarseShuffled(['image'],holes=4,max_holes=8,spatial_size=7,max_spatial_size=14,prob=0.1),
                # RandRotated(['image','gt'],range_x=15,range_y=15,mode=["bicubic","nearest"],prob=0.3),
                ToTensord(["image","gt","token","mask"]),
            ])
        
        else:
            trans = Compose([
                LoadImaged(["image","gt"], reader='PILReader'),
                EnsureChannelFirstd(["image","gt"]),
                Resized(["image"],spatial_size=image_size,mode='bicubic'),
                Resized(["gt"],spatial_size=image_size,mode='nearest'),
                HistogramNormalized(['image']),
                ToTensord(["image","gt","token","mask"]),

            ])

        return trans



if __name__ == "__main__":

    ds = QaTa('/home/zhongyi/junelin/QaTa-COV19/Train_text_for_Covid19.csv',
                '/home/zhongyi/junelin/QaTa-COV19/train')

    dl = DataLoader(ds, batch_size=1, shuffle=True)

    for data in dl:
        
        image_text, gt = data

        image = image_text[0]
        text = image_text[1]

        token = text['input_ids']
        mask = text['attention_mask']

        print(image.shape,gt.shape,token.shape,mask.shape)

        exit()
