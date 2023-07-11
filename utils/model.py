import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers import GuideDecoder, BottleNeck
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel


BIO_BERT_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/Bio_ClinicalBERT'
CXR_BERT_TYPE = "/dssg/home/ai2010812935/zhongyi/code/lib/BiomedVLP-CXR-BERT-specialized"
SWIN_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/swin-tiny-patch4-window7-224'
RES_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/resnet-50'
CONV_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/convnext-tiny-224'


class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)

        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling

        # let's take only the last hidden layer
        # embed = output['pooler_output']

        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim, checkpoint=None):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)
        
        self.spatial_dim = None

        if vision_type == RES_TYPE:
            self.project_head = nn.Linear(2048, project_dim)
            self.spatial_dim = 2048
        elif vision_type == SWIN_TYPE or CONV_TYPE:
            self.project_head = nn.Linear(768, project_dim)
            self.spatial_dim = 768
        else:
            print(vision_type)
            raise ValueError("TYPE error! VisionType Must be 'RES_TYPE' 'SWIN_TYPE' or 'CONV_TYPE'.")
    

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}



class MedCLIPSeg(nn.Module):

    def __init__(self, bert_type=CXR_BERT_TYPE, vision_type=RES_TYPE, project_dim=512 ,clip_checkpoint=None):

        super(MedCLIPSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224

        if vision_type == SWIN_TYPE:
            feature_dim = [768,384,192,96]
        elif vision_type == RES_TYPE:
            feature_dim = [2048,1024,512,256]
        elif vision_type == CONV_TYPE:
            feature_dim = [768,384,192,96]

        self.bottleneck = BottleNeck(feature_dim[0],1,project_dim)

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)

        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)

        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

        # self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

        if clip_checkpoint:
            self.load_state_dict(torch.load(clip_checkpoint))

    def forward(self, data):

        image = data[0]
        text = data[1]

        if image.shape[1] == 1:   # densenet using 1 channels  using reduce
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4:  # for resnet50 [b c h w]
            image_features = image_features[1:]  # 4 8 16 32   convnext输出有五层是 Embedding + 4层输出
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        # os32 = self.bottleneck(image_features[3],text_project.unsqueeze(dim=1))
        os32 = image_features[3]
        os16 = self.decoder16(os32,image_features[2],text_embeds[-1])
        os8 = self.decoder8(os16,image_features[1],text_embeds[-1])
        os4 = self.decoder4(os8,image_features[0],None)
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out
    



if __name__ == '__main__':


    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    model = MedCLIPSeg()

    image = torch.randn((1,1,224,224))
    text = "Normal chest. The cardiac silhouette and mediastinum size are within normal limits. There is no pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural effusion. There is no evidence of pneumothorax."
    text_token = tokenizer.encode_plus(text,return_tensors='pt')

    print(text_token['input_ids'].shape)

    # image_features, text_embeds = model(image, text_token)

    # for item in image_features:

        # print(item.shape)

        # torch.Size([1, 3136, 96])
        # torch.Size([1, 784, 192])
        # torch.Size([1, 196, 384])
        # torch.Size([1, 49, 768])
        # torch.Size([1, 49, 768])

    # print(image_features, text_embeds.shape)   # torch.Size([1, 49, 768]) torch.Size([1, 512])

    seg_out = model((image, text_token))
    print(seg_out.shape)


