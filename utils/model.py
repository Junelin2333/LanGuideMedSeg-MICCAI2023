import torch
import torch.nn as nn
from einops import rearrange, repeat
from .layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel



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
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):

    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()

        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):

        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)

        return {"feature":output['hidden_states'], "project":project}


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        os32 = image_features[3]
        os16 = self.decoder16(os32,image_features[2], text_embeds[-1])
        os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
        os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out
    
