'''
load checkpoint
take encoder only
loader videos

save embed to disk
load embed and  run k-means

'''
import torch
from iopath.common.file_io import g_pathmgr as pathmgr
from mae_st import models_mae
import mae_st.util.misc as misc
from mae_st.util.pos_embed import interpolate_pos_embed
from mae_st.util.kinetics import Kinetics
import numpy as np
device = 'cuda'

ckpt_path = '/workspace/data/mae_st/vit_large_t4_finetuned/checkpoint-00030.pth'

model_name = 'mae_vit_large_patch16'

model_kwargs = {'decoder_depth': 4,
                'decoder_embed_dim': 512 ,
                't_patch_size': 2,
                'cls_embed': True}

data_dir = '/workspace/data/mae_st/data/clips_4s/'

dataset_params = {
    'mode':'test',
    'path_to_data_dir': data_dir,
    'sampling_rate': 4,
    'num_frames': 16,
    'train_jitter_scales':(256,320),
    'test_crop_size': 224,
    'repeat_aug': 1,
    'rand_aug': False,
    'test_num_ensemble_views': 1,
    'test_num_spatial_crops': 1,
    'num_retries': 20
    }


# is it necessary to load the whole model?
model = models_mae.__dict__[model_name]( ** model_kwargs
)

with pathmgr.open(ckpt_path, "rb") as f:
    checkpoint = torch.load(f, map_location="cpu")

print("Load pre-trained checkpoint from: %s" % ckpt_path)


if "model" in checkpoint.keys():
    checkpoint_model = checkpoint["model"]
else:
    checkpoint_model = checkpoint["model_state"]
    
# what is this  for?
interpolate_pos_embed(model, checkpoint_model)    

checkpoint_model = misc.convert_checkpoint(checkpoint_model)

msg = model.load_state_dict(checkpoint_model, strict=False)
print (msg)
model.to('cuda')
# at test time, mask ratio should be 0?


dataset_test = Kinetics(
    **dataset_params
)

sampler_test = torch.utils.data.SequentialSampler(dataset_test)

loader_params = {    
    'sampler' : sampler_test,
    'batch_size': 1,
    'num_workers': 1,
    'pin_memory': False,    
}

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    **loader_params
)

latent_list = []
video_list = []
for cur_iter, (images, labels, video_idx) in enumerate(data_loader_test):
    with torch.no_grad():
        images = images.to(device, non_blocking=True)

        images = torch.squeeze(images, 0)
        labels = labels.to(device, non_blocking=True)
        video_idx = video_idx.to(device, non_blocking=True)
        latent, mask, ids_restore = model.forward_encoder(images, mask_ratio = 0)
        latent = latent.mean(dim = [-1]).view(1,-1)
        latent_list.append(latent.detach().cpu().numpy())
        video_list.append(video_idx.detach().cpu().numpy())

latent_list = np.concatenate(latent_list, axis = 0)

with open('data/latents.npy', 'wb') as f:
    np.save(f, latent_list)

with open('data/video_labels.npy', 'wb') as f:
    np.save(f, video_list)
