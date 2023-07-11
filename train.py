import torch
from torch.utils.data import DataLoader
from utils.dataset import QaTa
from torch.optim import lr_scheduler
from utils.wrapper import MedCLIPSegWrapper

import pytorch_lightning as pl    
from torchmetrics import Accuracy,Dice
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_VISIBLE_DEVICES'] = '13'

batch_size = 32
lr = 3e-4
print("batchsize:",batch_size)
print("lr:",lr)
print("no data augment")

BIO_BERT_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/Bio_ClinicalBERT'
CXR_BERT_TYPE = "/dssg/home/ai2010812935/zhongyi/code/lib/BiomedVLP-CXR-BERT-specialized"
SWIN_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/swin-tiny-patch4-window7-224'
RES_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/resnet-50'
CONV_TYPE = '/dssg/home/ai2010812935/zhongyi/code/lib/convnext-tiny-224'


if __name__ == '__main__':

    print("cuda:",torch.cuda.is_available())

    old = '/dssg/home/ai2010812935/zhongyi/data/QaTa-COV19-v2/washed_prompt/Train_text_for_Covid19_washed_changed.csv'
    new = '/dssg/home/ai2010812935/zhongyi/data/QaTa-COV19-v2/stage_prompt/train_text_stage3_64.csv'

    ds_train = QaTa(csv_path=old,
                    root_path='/dssg/home/ai2010812935/zhongyi/data/QaTa-COV19-v2/Train',
                    mode='train')

    ds_valid = QaTa(csv_path=old,
                    root_path='/dssg/home/ai2010812935/zhongyi/data/QaTa-COV19-v2/Train',
                    mode='valid')


    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=batch_size*2)
    dl_valid = DataLoader(ds_valid, batch_size=8, shuffle=False, num_workers=8)

    model = MedCLIPSegWrapper(CXR_BERT_TYPE, CONV_TYPE, 
                             metrics_dict = {"acc":Accuracy(),"dice":Dice(),"MIoU":BinaryJaccardIndex()},
                             lr = lr)

    #1，设置回调函数
    model_ckpt = ModelCheckpoint(
        dirpath='/dssg/home/ai2010812935/zhongyi/save_model',
        filename='convseg',
        monitor='val_loss',
        save_top_k=1,
        # save_weights_only=True,
        mode='min',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'val_loss',
                            patience=20,
                            mode = 'min'
    )

    #2，设置训练参数

    # gpus=0 则使用cpu训练，gpus=1则使用1个gpu训练，gpus=2则使用2个gpu训练，gpus=-1则使用所有gpu训练，
    # gpus=[0,1]则指定使用0号和1号gpu训练， gpus="0,1,2,3"则使用0,1,2,3号gpu训练
    # tpus=1 则使用1个tpu训练
    trainer = pl.Trainer(logger=True,
                        # precision=16,
                        # accumulate_grad_batches=4,
                        min_epochs=20,max_epochs=200,
                        accelerator='gpu', 
                        devices=1,
                        callbacks = [model_ckpt,early_stopping],
                        enable_progress_bar =False,
                        ) 

    ##4，启动训练循环
    print('start training')

    trainer.fit(model,dl_train,dl_valid)

    print('done training')

