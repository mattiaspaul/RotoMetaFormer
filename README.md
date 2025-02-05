# Look, No Convs!
## Permutation- and Rotation-invariance for MetaFormers
Code for BVM 2025 paper by Mattias P. Heinrich

### The general concept is shown below
![3694-data_concept](https://github.com/user-attachments/assets/26d9e92f-71df-41b1-802a-143708b232b4)

Our new Roto-MetaFormers achieve permutation and rotation invariance by 1) removing positional encoding and 2) replacing all convolutions with their invariant version (RotoConv).
This work is in part inspired by [https://doi.org/10.1016/j.media.2020.101849]


### RotoConv
Making convolutional layers rotation- and permutation invariant in 2D can be simply achieved by applying 8 versions of the same filter or alternatively eight versions of the image/patch and taking a symmetric aggregation, usually the max operation.
```
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
```
This function and module achieve perfect invariance with 8x higher computational demand. While inserting them into a e.g. ResNet works well, it also slows down inference by 8x.  


### Pool- and Metaformers
Pool- and Metaformers reduce the number of operations that are dependent on input permutation or rotation to leave only the 4 convolutional patch merging / downsampling layers susceptible to this. Poolformers nevertheless maintain a **local inductive bias** by replacing global self-attention and learned positional encoding with a simple average-pooling mask. See details in (https://github.com/sail-sg/poolformer/). The operator can be written as follows (assuming input dimensions B x H x W^2 x C):
```
def poolformer(value):
    B, H, W2, C = value.shape
    W = int(math.sqrt(value.shape[2]))
    value_ = value.permute(0,1,3,2).flatten(0,1).unflatten(2,(W,W))
    out = F.avg_pool2d(value_,3,1,1).flatten(-2,-1).unflatten(0,(B,H)).permute(0,1,3,2)
    return out
```
If you are familiar with [FlexAttention](https://pytorch.org/blog/flexattention/) nearly the same outcome can be obtained with the following example with W^2 = 32^2:
```
def pool32(b, h, q_idx, kv_idx):
    mask = ((q_idx//32-kv_idx//32).abs() <= 1) & ((q_idx%32-kv_idx%32).abs() <= 1)
    return mask
def noscore(score, b, h, q_idx, kv_idx):
    return score*0+1
block_mask = create_block_mask(pool32, B=None, H=None, Q_LEN=S, KV_LEN=S)
attn_out = flex_attention(query, key, value, block_mask=block_mask, score_mod=noscore)
```
This leaves key and query unused (score is set to all ones) and returns a local 2D mask with kernel 3x3. Following the earlier MetaFormer work, we use PoolFormer blocks for all higher resolutions and only resort to full-attention (without positional encoding at all) for the lowest resolution. Future experiments could also consider a combination of self-attention and local masking to better balance inductive bias, rotationally invariant position encoding and the strenghts of self-attention with learned queries and keys.

### Experiments and Validation
We provide trained models (via Google Drive link in respective folder) and our complete train and evaluation scripts. To replicate some key results you may run
``python3 looknoconvs_experiments.py --validate 1 --model metaformer_s12_pppa`` and ``python3 looknoconvs_experiments.py --validate 1 --model metaformer_s12_pppa --roto 0``.
This should reveal the 10% pts gain of Roto-MetaFormer for DermaMNIST (F1=53% vs 43%). You can train own models by removing the ``--validate 1`` argument.


