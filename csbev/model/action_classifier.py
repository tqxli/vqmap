import argparse
import csv
import json
import os
from copy import deepcopy
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import initialize, compose
from hydra.utils import instantiate
from lightning import seed_everything

from csbev.utils.run import count_parameters
from csbev.dataset.loader import filter_by_keys


device = "cuda:0"
actions = [
    'idle',
    'sniff/head',
    'groom',
    'scrunched',
    'crouched',
    'reared',
    'explore',
    'locomotion',
    'error',
 ]


class SeqClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float = 0.2):
        super(SeqClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.skelEmbedding = nn.Linear(self.input_size, self.hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  
        self.softmax = nn.LogSoftmax(dim=1)
        self._set_temporal_layer()
        
        # self.dropout = nn.Dropout(dropout)
    
    def _set_temporal_layer(self):
        self.temp_layer = nn.Identity()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        x = self.skelEmbedding(x)
        out, _ = self.temp_layer(x, h0)
        out = self.fc(out)
        return out


class GRU_Classifier(SeqClassifier):
    def _set_temporal_layer(self):
        self.temp_layer = nn.GRU(
            self.hidden_size, self.hidden_size, self.num_layers,
            batch_first=True
        )


class LSTM_Classifier(SeqClassifier):
    def _set_temporal_layer(self):
        self.temp_layer = nn.LSTM(
            self.hidden_size, self.hidden_size, self.num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes) 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = self.skelEmbedding(x)
        out, _ = self.temp_layer(x, (h0, c0))
        out = self.fc(out)
        return out


class BiLSTM_Classifier(SeqClassifier):
    def _set_temporal_layer(self):
        self.temp_layer = nn.LSTM(
            self.hidden_size, self.hidden_size, self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes) 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        x = self.skelEmbedding(x)
        out, _ = self.temp_layer(x, (h0, c0))
        out = self.fc(out)
        return out


class LatentActionClassifier(nn.Module):
    def __init__(
            self,
            vae: nn.Module,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            seqlen: int = 128,
            input_type: Literal["decoder", "bottleneck", "encoder", "input", "recon"] = "encoder",
            model_type: Literal["LSTM", "GRU", "BiLSTM"] = "LSTM",
    ):
        super().__init__()

        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad = False
        for quantizer in self.vae.bottleneck.sub_quantizers:
            quantizer.reinit_codebook = False
            quantizer.freeze_codebook = True
        
        self.input_type = input_type
        if input_type == "decoder":
            input_size = vae.decoder.decoder_shared.hidden_dim
        elif input_type == "bottleneck":
            input_size = vae.encoder.latent_dim
        elif input_type == "encoder":
            input_size = vae.encoder.fc.in_features
        elif input_type == "input" or input_type == "recon":
            input_size = 23 * 3
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

        self.model_type = model_type
        classifier_class = globals().get(f"{model_type}_Classifier")
        self.classifier = classifier_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
        )

        self.seqlen = seqlen

    def forward(self, batch):
        with torch.no_grad():
            if self.input_type == "decoder":
                z = self.vae.encode(batch)[0]
                feats = self.vae.decoder.decoder_shared.input_proc(z)
                feats = self.vae.decoder.decoder_shared.backbone(feats)
                feats = feats.permute(0, 2, 1)
            elif self.input_type == "bottleneck":
                z = self.vae.encode(batch)[0]
                feats = z.permute(0, 2, 1)
                ds_rate = self.seqlen // feats.size(1)
                feats = feats.unsqueeze(1).repeat(1, 1, ds_rate, 1).flatten(1, 2)
            elif self.input_type == "encoder":
                x = batch["x"]  # [B, T, n_points, D]
                x = x.permute(0, 2, 3, 1)  # [B, n_points, D, T]

                feats = self.vae.encoder._forward(
                    x,
                    pe_indices=batch["be"],
                    batch_tag=batch["tag_in"],
                )  # -> [B, latent_dim, T]
                feats = feats.permute(0, 2, 1)  # [B, T, latent_dim]
                ds_rate = self.seqlen // feats.size(1)
                feats = feats.unsqueeze(2).repeat(1, 1, ds_rate, 1).flatten(1, 2)
                
            elif self.input_type == "input":
                feats = batch["x"].flatten(2, 3)
            elif self.input_type == "recon":
                feats = self.vae.reconstruct(batch)
                feats = feats.permute(0, 2, 1)
        
        out = self.classifier(feats)
        return out


def _forward_batch_converted(model, batch, tag_in, tag_out):
    batch['x'] = batch['x'].reshape(*batch['x'].shape[:2], -1, 3)
    batch['tag_in'] = tag_in
    batch["tag_out"] = tag_out
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        
    outputs = model(batch)
    outputs = outputs.view(-1, outputs.shape[-1])
    return outputs


def _forward_batch(model, batch):
    batch['x'] = batch['x'].reshape(*batch['x'].shape[:2], -1, 3)
    # batch['tag_in'] = batch['tag_out'] = args.dataset
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        
    outputs = model(batch)
    outputs = outputs.view(-1, outputs.shape[-1])
    targets = batch['action'].to(device).view(-1)
    return outputs, targets
    

def train_epoch(model, optimizer, criterion, train_loader):
    total_loss = []
    num_correct, num_tot = 0, 0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        outputs, targets = _forward_batch(model, batch)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
        
        num_tot += len(targets)
        num_correct += (outputs.argmax(dim=1) == targets).sum().item()
    
    return np.mean(total_loss), num_correct / num_tot * 100


def evaluate_epoch(model, criterion, test_loader):
    total_loss = []
    num_correct, num_tot = 0, 0
    num_tot_by_class = np.zeros(9)
    num_correct_by_class = np.zeros(9)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            outputs, targets = _forward_batch(model, batch)
            loss = criterion(outputs, targets)
            total_loss.append(loss.item())
            num_tot += len(targets)
            num_correct += (outputs.argmax(dim=1) == targets).sum().item()
            
            for i in range(len(targets)):
                num_tot_by_class[targets[i]] += 1
                if outputs.argmax(dim=1)[i] == targets[i]:
                    num_correct_by_class[targets[i]] += 1
    
    test_acc = num_correct / num_tot * 100
    test_acc_by_class = num_correct_by_class / num_tot_by_class * 100
    
    return np.mean(total_loss), test_acc, test_acc_by_class


def prepare_rat23_dataset(seqlen):
    # rat23 dataset with labels
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="dataset/rat23.yaml").dataset
        del cfg.split
    datapaths = sorted(list(np.load(cfg.dataroot, allow_pickle=True)[()].keys()))
    cfg['_target_'] = 'csbev.dataset.base.RatActionDataset'
    
    cfg["seqlen"] = seqlen

    train_ids = ["M1", "M2", "M3", "M4", "M5"]
    # train_ids = ["2022_09_15_M1"]
    cfg_train = deepcopy(cfg)
    cfg_train.datapaths = [datapaths[idx] for idx in filter_by_keys(train_ids, datapaths)]

    test_ids = ["M6"]
    cfg_test = deepcopy(cfg)
    cfg_test.datapaths = [datapaths[idx] for idx in filter_by_keys(test_ids, datapaths)]

    train_dataset = instantiate(cfg_train)
    test_dataset = instantiate(cfg_test)
    return train_dataset, test_dataset


def prepare_mouse23_dataset(seqlen):
    # mouse23 dataset with labels
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="dataset/mouse23.yaml").dataset
        del cfg.split
    datapaths = sorted(list(np.load(cfg.dataroot, allow_pickle=True)[()].keys()))
    cfg['_target_'] = 'csbev.dataset.base.MouseActionDataset'
    cfg["seqlen"] = seqlen

    train_ids = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    # train_ids = ["W1", "B1"]
    cfg_train = deepcopy(cfg)
    cfg_train.datapaths = [datapaths[idx] for idx in filter_by_keys(train_ids, datapaths)]
    
    test_ids = ["W8", "B8"]
    cfg_test = deepcopy(cfg)
    cfg_test.datapaths = [datapaths[idx] for idx in filter_by_keys(test_ids, datapaths)]
    
    train_dataset = instantiate(cfg_train)
    test_dataset = instantiate(cfg_test)
    return train_dataset, test_dataset


def train(args, expdir, model, optimizer, criterion, num_epochs: int = 100, seqlen: int = 128):
    # dataset
    if args.dataset == "rat23":
        train_dataset, test_dataset = prepare_rat23_dataset(seqlen)

    elif args.dataset == "mouse23":
        train_dataset, test_dataset = prepare_mouse23_dataset(seqlen)
    
    elif args.dataset == "rat23+mouse23":
        train_dataset_rat, test_dataset_rat = prepare_rat23_dataset(seqlen)
        train_dataset_mouse, test_dataset_mouse = prepare_mouse23_dataset(seqlen)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_rat, train_dataset_mouse])
        test_dataset = torch.utils.data.ConcatDataset([test_dataset_rat, test_dataset_mouse])
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print( f"Train dataset: {len(train_dataset)} | Test dataset: {len(test_dataset)}")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs, shuffle=False,
    )

    # start training
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_dataloader)
        test_loss, test_acc, test_acc_by_class = evaluate_epoch(model, criterion, test_dataloader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f} Accuracy: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Accuracy: {test_acc:.2f}%')
    
        torch.save(model.state_dict(), os.path.join(expdir, f"action_{model.model_type}_classifier_{model.input_type}_s{seqlen}.pth"))

        metric_dict = {action: test_acc_by_class[i] for i, action in enumerate(actions)}
        metric_dict["macro"] = test_acc
        
        savefile = os.path.join(expdir, f"metrics_{model.input_type}_{model.model_type}_s{seqlen}.csv")
        with open(savefile, 'a') as f:
            w = csv.writer(f)
            if epoch == 0:
                w.writerow(metric_dict.keys())
            w.writerow(metric_dict.values())
    
    return model, test_acc, test_acc_by_class


def load_vae(model_root: str):
    ckpt_path = os.path.join(model_root, 'checkpoints', 'model.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu')

    state_dict = ckpt["model"]
    vq = instantiate(ckpt["config"].model)
    vq.load_state_dict(state_dict)
    return vq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", type=str)
    parser.add_argument("--input_type", type=str, default="bottleneck")
    parser.add_argument("--dataset", type=str, default="rat23")
    parser.add_argument("--model_type", type=str, default="LSTM")
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    assert os.path.exists(args.model_root)
    expdir = os.path.join(args.model_root, "classification", args.dataset)
    if not os.path.exists(expdir):
        os.makedirs(expdir)
        
    seed_everything(42)
    
    # model
    vae = load_vae(args.model_root)
    model = LatentActionClassifier(
        vae,
        hidden_size=256,
        num_layers=3,
        num_classes=len(actions),
        input_type=args.input_type,
        seqlen=args.seqlen,
        model_type=args.model_type,
    ).to(device)
    print(f"Model: {count_parameters(model)} M")

    # optimizer
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    model, test_acc, test_acc_by_class = train(
        args,
        expdir,
        model,
        optimizer,
        criterion,
        num_epochs=args.num_epochs,
        seqlen=args.seqlen,
    )