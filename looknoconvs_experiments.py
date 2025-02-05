#looknoconvs experiments
import os

import torch
from poolformer import poolformer_s12,poolformer_m48,metaformer_s12_224,metaformer_s12_pppa
import sys
from torcheval.metrics.functional import multiclass_f1_score
import torchvision
from medmnist import BreastMNIST,TissueMNIST,DermaMNIST,PathMNIST
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm,trange
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='dermamnist')
    parser.add_argument('--roto', type=bool, default=False)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--validate', type=str, default="0")
    return parser.parse_args()

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

def roto_patch(input):
    input_equi = input.unsqueeze(0).repeat(8,1,1,1,1)
    for i,f in enumerate(((),(-1,),(-2,-1),(-2))):
        input_equi[i] = input.flip(f)
        input_equi[i+4] = input.transpose(-2,-1).flip(f)
    return input_equi

class roto_conv(nn.Module):
    def __init__(self,conv1) -> None:
        super().__init__()
        self.conv1 = conv1

    def forward(self, x):
        x_equi = roto_patch(x)
        x = self.conv1(x_equi.flatten(0,1)).unflatten(0,(8,-1)).max(0).values
        return x


def load_data(name):
    if name == 'breastmnist':
        data = BreastMNIST('train',download=True,size=128)
        data_val = BreastMNIST('test',download=True,size=128)
    elif name == 'tissuemnist':
        data = TissueMNIST('train',download=True,size=128)
        data_val = TissueMNIST('test',download=True,size=128)
    elif name == 'dermamnist':
        data = DermaMNIST('train',download=True,size=128)
        data_val = DermaMNIST('test',download=True,size=128)
    elif name == 'pathmnist':
        data = PathMNIST('train',download=True,size=128)
        data_val = PathMNIST('test',download=True,size=128)

    data.imgs = torch.from_numpy(data.imgs)
    data_val.imgs = torch.from_numpy(data_val.imgs)
    print(data.imgs.shape)
    if(data.imgs.shape[-1]!=3):
        data.imgs = data.imgs.unsqueeze(-1).repeat(1,1,1,3)
        data_val.imgs = data_val.imgs.unsqueeze(-1).repeat(1,1,1,3)
    print(data.imgs.shape)

    imgs = data.imgs.permute(0,3,1,2).float().div(255)#.cuda()
    labels = torch.from_numpy(data.labels).long().cuda().squeeze(1)

    imgs_val = data_val.imgs.permute(0,3,1,2).float().div(255)#.cuda()
    labels_val = torch.from_numpy(data_val.labels).long().cuda().squeeze(1)
    num_label = int(labels_val.max()+1)

    return imgs, labels, imgs_val, labels_val, num_label

def create_model(name,roto=False):
    if(name == 'poolformer_s12'):
        model = poolformer_s12().cuda()
    elif(name == 'metaformer_s12_pppa'):
        model = metaformer_s12_pppa().cuda()
    elif(name == 'resnet18'):
        model = torchvision.models.resnet.resnet18().cuda()
    if(roto&('former' in name)):
        model.patch_embed.proj = roto_conv(model.patch_embed.proj)
        model.network[1].proj = roto_conv(model.network[1].proj)
        model.network[3].proj = roto_conv(model.network[3].proj)
        model.network[5].proj = roto_conv(model.network[5].proj)
    elif(roto&('resnet' in name)):
        for name, module in model.named_modules():
            if isinstance(module,nn.Conv2d):
                before = get_layer(model, name)
                set_layer(model, name, roto_conv(before))
    return model

def train_model(model,imgs,labels,imgs_val,labels_val,num_label,stride_val=1):

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,1000,2)
    num_iterations = 3000
    run_loss = torch.zeros(num_iterations,2)
    val_acc0 = torch.zeros(num_iterations//50,len(imgs_val))
    val_acc1 = torch.zeros(num_iterations//50,len(imgs_val))
    val_f1_0 = torch.zeros(num_iterations//50,num_label)
    val_f1_1 = torch.zeros(num_iterations//50,num_label)
    with tqdm(total=num_iterations, file=sys.stdout) as pbar:
        for i in range(num_iterations):

            optimizer.zero_grad()
            idx = torch.randperm(len(imgs))[:96]
            img_aff = F.grid_sample(imgs[idx].cuda(),F.affine_grid(torch.eye(2,3).unsqueeze(0).cuda()+torch.randn(96,2,3).cuda()*.05,(96,1,128,128)))
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                
                output = model(img_aff)[:,:num_label]
                loss = nn.CrossEntropyLoss()(output,labels[idx])
            loss.backward()
            optimizer.step()
            scheduler.step()
            run_loss[i,0] = loss.item()
            acc = (output.argmax(1)==labels[idx]).float().data
            run_loss[i,1] = acc.mean()
            if(i%50==49):
                model.eval()
                output_argmax0 = torch.zeros(len(imgs_val)).long()
                output_argmax1 = torch.zeros(len(imgs_val)).long()

                with torch.no_grad():
                    ridx = torch.arange(0,len(imgs_val),stride_val)
                    chk = torch.chunk(ridx,len(imgs_val)//64)
                    #val_acc[i//50] = torch.zeros(13*12)
                    for idx in chk:
                        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                            output = model(imgs_val[idx].cuda())[:,:num_label]#
                        acc = (output.argmax(1)==labels_val[idx]).float().data
                        output_argmax0[idx] = output.argmax(1).cpu()
                        val_acc0[i//50][idx] = acc.cpu().data#.mean()
                        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                            output = model(imgs_val[idx].cuda().permute(0,1,3,2).flip(-1))[:,:num_label]#
                        acc = (output.argmax(1)==labels_val[idx]).float().data
                        output_argmax1[idx] = output.argmax(1).cpu()
                        val_acc1[i//50][idx] = acc.cpu().data#.mean()
                val_f1_0[i//50] = multiclass_f1_score(output_argmax0[ridx],labels_val[ridx].cpu(),num_classes=num_label,average=None)
                val_f1_1[i//50] = multiclass_f1_score(output_argmax1[ridx],labels_val[ridx].cpu(),num_classes=num_label,average=None)

                model.train()
                str1 = f"iter: {i}, acc0: {'%0.3f'%(val_acc0[i//50,ridx].mean())}, acc1: {'%0.3f'%(val_acc1[i//50,ridx].mean())} , f1_0: {'%0.3f'%(val_f1_0[i//50].mean())}, f1_1: {'%0.3f'%(val_f1_1[i//50].mean())} , GPU max/memory: {'%0.2f'%(torch.cuda.max_memory_allocated()*1e-9)} GByte"
                pbar.set_description(str1)
            pbar.update(1)
    return run_loss,val_acc0,val_acc1,val_f1_0,val_f1_1,model#.state_dict()

def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print('using gpu',args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    imgs, labels, imgs_val, labels_val, num_label = load_data(args.name)
    model = create_model(args.model,args.roto)
    if(int(args.validate)==0):
        run_loss,val_acc0,val_acc1,val_f1_0,val_f1_1,model = train_model(model,imgs,labels,imgs_val,labels_val,num_label)
        torch.save(model.state_dict(), f'trained_models/{args.name}_{args.model}_roto{args.roto}.pth')
    else:
        model.load_state_dict(torch.load(f'trained_models/{args.name}_{args.model}_roto{args.roto}.pth',weights_only=True))    


    model.eval()
    output_argmax1 = torch.zeros(len(imgs_val)).long()
    with torch.no_grad():
        ridx = torch.arange(0,len(imgs_val),1)
        chk = torch.chunk(ridx,len(imgs_val)//64)
        for idx in chk:
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
    #            output = model(imgs_val[idx].cuda().permute(0,1,3,2).flip(-1))[:,:num_label]#
                output = model(imgs_val[idx].cuda())[:,:num_label]#
            acc = (output.argmax(1)==labels_val[idx]).float().data
            output_argmax1[idx] = output.argmax(1).cpu()
            
    scores1 = multiclass_f1_score(output_argmax1,labels_val.cpu(),num_classes=num_label,average=None)
    print('vanilla','f1','%0.3f'%scores1.mean().item(),'acc','%0.3f'%(output_argmax1==labels_val.cpu()).float().mean().item())
    output_argmax1 = torch.zeros(len(imgs_val)).long()
    with torch.no_grad():
        ridx = torch.arange(0,len(imgs_val),1)
        chk = torch.chunk(ridx,len(imgs_val)//64)
        for idx in chk:
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                output = model(imgs_val[idx].cuda().permute(0,1,3,2).flip(-1))[:,:num_label]#
    #            output = model(imgs_val[idx].cuda())[:,:num_label]#
            acc = (output.argmax(1)==labels_val[idx]).float().data
            output_argmax1[idx] = output.argmax(1).cpu()
            
    scores1 = multiclass_f1_score(output_argmax1,labels_val.cpu(),num_classes=num_label,average=None)
    print('flipped','f1','%0.3f'%scores1.mean().item(),'acc','%0.3f'%(output_argmax1==labels_val.cpu()).float().mean().item())

if __name__ == '__main__':
    args = get_args()
    main(args)