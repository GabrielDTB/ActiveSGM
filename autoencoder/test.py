import os
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ChunkedAutoencoderDataset
from model import Autoencoder
from train import l2_loss,cos_loss
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--encoder_dims',
                        nargs='+',
                        type=int,
                        default=[512, 256, 128, 64, 32, 16],
                        )
    parser.add_argument('--decoder_dims',
                        nargs='+',
                        type=int,
                        default=[32, 64, 128, 256, 256, 512],
                        )
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
    ckpt_path = f"ckpt/{dataset_name}/best_ckpt.pth"

    test_dir = 'val_grid1.0cm_chunk6x6_stride3x3'
    test_data_dir = f'{dataset_path}/{test_dir}'
    output_dir = f"{dataset_path}/language_features_dim16"
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(ckpt_path)
    test_dataset = ChunkedAutoencoderDataset(
        root_dir=test_data_dir,
        files_per_chunk=5,
        shuffle_files=False,  # optional: keep eval deterministic
        shuffle_items=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    # --- wandb init ---
    wandb.init(
        project="autoencoder",
        name=args.dataset_name,
        config={
            "dataset_path": dataset_path,
            "split": test_dir,
            "encoder_dims": encoder_hidden_dims,
            "decoder_dims": decoder_hidden_dims,
            "batch_size": 128,
        },
    )


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, in_feature_dim=768).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    l2 = 0.0
    cos = 0.0

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to(device)
        # with torch.no_grad():
        #     outputs = model.encode(data).to("cpu").numpy()
        # if idx == 0:
        #     features = outputs
        # else:
        #     features = np.concatenate([features, outputs], axis=0)
        dec_data = model(data)
        l2loss = l2_loss(dec_data,data).item()
        cosloss = cos_loss(dec_data,data).item()
        l2 += l2loss * data.size(0)
        cos += cosloss * data.size(0)

        wandb.log(
            {
                "test/l2_loss": l2loss,
                "test/cos_loss": cosloss,
            },
            step=idx,
        )

    avg_l2 = l2 / len(test_dataset)
    avg_cos = cos / len(test_dataset)

    print(f"num_of_samples:{len(test_dataset)}")
    print(f"l2_loss:{avg_l2:.8f}")
    print(f"cos_loss:{avg_cos:.8f}")

    # os.makedirs(output_dir, exist_ok=True)
    # start = 0
    #
    # for k,v in test_dataset.data_dic.items():
    #     path = os.path.join(output_dir, k)
    #     np.save(path, features[start:start+v])
    #     start += v
