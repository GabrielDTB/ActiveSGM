import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import AutoencoderDataset,ChunkedAutoencoderDataset
from model import Autoencoder
from torch.utils.tensorboard import SummaryWriter
import wandb
import argparse

torch.autograd.set_detect_anomaly(True)


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--max_train_iters_per_epoch', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[512, 256, 128, 64, 32, 16],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--dataset_name', type=str, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs

    os.makedirs(f'ckpt/{args.dataset_name}', exist_ok=True)
    train_dir = 'test_grid1.0cm_chunk6x6_stride3x3'
    test_dir = 'train_grid1.0cm_chunk6x6_stride3x3'

    train_data_dir = f'{dataset_path}/{train_dir}'
    test_data_dir = f'{dataset_path}/{test_dir}'

    # train_dataset = AutoencoderDataset(root_dir=train_data_dir)
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     num_workers=16,
    #     drop_last=False
    # )

    train_dataset = ChunkedAutoencoderDataset(
        root_dir=train_data_dir,
        files_per_chunk=5,  # your “load 5 files at a time”
        shuffle_files=True,
        shuffle_items=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=False,  # must be False for IterableDataset
        num_workers=4,  # try 2–4; 16 can be overkill & slow IO
        drop_last=False,
    )

    # test_dataset = AutoencoderDataset(root_dir=test_data_dir)
    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=64,
    #     shuffle=False,
    #     num_workers=16,
    #     drop_last=False
    # )


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
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims, in_feature_dim=768).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logdir = f'ckpt/{args.dataset_name}'

    # --- wandb init ---
    wandb.init(
        project="autoencoder",
        name=args.dataset_name,
        config={
            "dataset_path": dataset_path,
            "num_epochs": num_epochs,
            "lr": args.lr,
            "encoder_dims": encoder_hidden_dims,
            "decoder_dims": decoder_hidden_dims,
            "batch_size": 1024,
        },
    )
    wandb.watch(model, log="gradients", log_freq=100)

    best_eval_loss = float("inf")
    best_epoch = 0

    max_iters_per_epoch = args.max_train_iters_per_epoch
    global_step = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in enumerate(train_loader):

            if idx >= max_iters_per_epoch:
                break

            data = feature.to(device)
            outputs_dim16 = model.encode(data)
            outputs = model.decode(outputs_dim16)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # global_iter = epoch * len(train_loader) + idx

            # --- wandb logging (train) ---
            wandb.log(
                {
                    "train/l2_loss": l2loss.item(),
                    "train/cos_loss": cosloss.item(),
                    "train/total_loss": loss.item(),
                    "epoch": epoch,
                },
                step=global_step,
            )
            # histogram of outputs
            wandb.log(
                {"train/feat_hist": wandb.Histogram(outputs.detach().cpu().numpy())},
                step=global_step,
            )
            global_step += 1

        # --- eval after warmup ---
        if (epoch % 10==0 ) or (epoch > 95):
            model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for _, feature in enumerate(test_loader):
                    data = feature.to(device)
                    outputs = model(data)
                    loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                    eval_loss += loss.item() * data.size(0)

            eval_loss = eval_loss / len(test_dataset)
            print(f"eval_loss:{eval_loss:.8f}")

            # --- wandb logging (eval) ---
            wandb.log(
                {
                    "eval/loss": eval_loss,
                    "epoch": epoch,
                },
                step=global_step,
            )

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    f'ckpt/{args.dataset_name}/best_ckpt.pth',
                )

            if epoch % 10 == 0:
                torch.save(
                    model.state_dict(),
                    f'ckpt/{args.dataset_name}/{epoch}_ckpt.pth',
                )
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))
    wandb.finish()