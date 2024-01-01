import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import OrderedDict
import numpy as np
import os
import time


class GaussianGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(GaussianGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device))
        self.hidden = hidden
        return hidden

    def reparameterize(self, mu, logvar):
        s_var = logvar.mul(0.5).exp_()
        eps = s_var.data.new(s_var.size()).normal_()
        return eps.mul(s_var).add_(mu)

    def forward(self, inputs):
        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.gru[i](h_in, self.hidden[i])
            h_in = self.hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar, h_in

    def get_normal_noise(self, num_samples, device):
        return torch.randn(num_samples, self.output_size, device=device).float().requires_grad_(False)


class DecoderGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, device):
        super(DecoderGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.device = device
        self.embed = nn.Linear(input_size, hidden_size)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])

        self.output = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, num_samples=None):
        batch_size = num_samples if num_samples is not None else self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append(torch.zeros(batch_size, self.hidden_size).requires_grad_(False).to(self.device))
        self.hidden = hidden
        return hidden

    def forward(self, inputs):
        embedded = self.embed(inputs.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.gru[i](h_in, self.hidden[i])
            h_in = self.hidden[i]
        return self.output(h_in), h_in


def get_action2motion(
    input_size, output_size,
    batch_size,
    device,
    lambdas,
    **kwargs,
):
    return Action2Motion(input_size, output_size, batch_size, device, lambdas, **kwargs)
    

class Action2Motion(nn.Module):
    def __init__(
        self, 
        input_size, output_size,
        batch_size,
        device="cuda",
        lambdas=None,
        dim_z=64, hidden_size=128, time_counter=True,
        prior_hidden_layers=1,
        posterior_hidden_layers=1,
        decoder_hidden_layers=2,
        tf_ratio=0.6,
        skip_prob=0,
    ):
        super().__init__()
        
        if time_counter:
            input_size = input_size + 1

        self.prior_net = GaussianGRU(
            input_size, dim_z, hidden_size,
            prior_hidden_layers, batch_size, device)
        self.posterior_net = GaussianGRU(
            input_size, dim_z, hidden_size,
            posterior_hidden_layers, batch_size, device)
        self.decoder = DecoderGRU(
            input_size + dim_z, output_size, hidden_size,
            decoder_hidden_layers,
            batch_size, device)

        self.input_size = input_size
        self.lambdas = lambdas
        self.device = device
        self.batch_size = batch_size
        self.tf_ratio = tf_ratio
        self.recons_loss = nn.MSELoss()

    def ones_like(self, t, val=1):
        return torch.Tensor(t.size()).fill_(val).requires_grad_(False).to(self.device)

    def zeros_like(self, t, val=0):
        return torch.Tensor(t.size()).fill_(val).requires_grad_(False).to(self.device)

    def tensor_fill(self, tensor_size, val=0):
        return torch.zeros(tensor_size).fill_(val).requires_grad_(False).to(self.device)


    def kl_criterion(self, mu1, logvar1, mu2, logvar2):
        # KL( N(mu1, sigma2_1) || N(mu_2, sigma2_2))
        # loss = log(sigma2/sigma1) / 2 + (sigma1 + (mu1 - mu2)^2)/(2*sigma2) - 1/2
        sigma1 = logvar1.mul(0.5).exp()
        sigma2 = logvar2.mul(0.5).exp()
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1-mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return kld.sum() / self.batch_size

    def forward(self, batch):
        self.prior_net.init_hidden()
        self.posterior_net.init_hidden()
        self.decoder.init_hidden()

        data = batch["motion"]
        motion_length = data.shape[1]

        # dim(batch_size, pose_dim), initial prior is a zero vector
        prior_vec = self.tensor_fill((data.shape[0], data.shape[2]), 0)

        mse = 0
        kld = 0
        lossdict = {}

        generate_batch = []
        opt_step_cnt = 0

        teacher_force = True if (random.random() < self.tf_ratio) or (not self.training) else False
        for i in range(0, motion_length):
            time_counter = i / (motion_length - 1)
            time_counter_vec = self.tensor_fill((data.shape[0], 1), time_counter)
            condition_vec = time_counter_vec
            # print(prior_vec.shape, condition_vec.shape)
            h = torch.cat((prior_vec, condition_vec), dim=1)
            h_target = torch.cat((data[:, i], condition_vec), dim=1)

            z_t, mu, logvar, h_in_p = self.posterior_net(h_target)
            z_p, mu_p, logvar_p, _ = self.prior_net(h)

            h_mid = torch.cat((h, z_t), dim=1)
            x_pred, h_in = self.decoder(h_mid)

            if teacher_force:
                prior_vec = x_pred
            else:
                prior_vec = data[:, i]
            
            generate_batch.append(x_pred.unsqueeze(1))
            opt_step_cnt += 1
            mse += self.recons_loss(x_pred, data[:, i])
            kld += self.kl_criterion(mu, logvar, mu_p, logvar_p)
        
        lossdict['recons'] = mse / opt_step_cnt
        lossdict['kl'] = kld / opt_step_cnt
        
        mixed_loss = lossdict["recons"] * self.lambdas["recons"] + lossdict["kl"] * self.lambdas["kl"]
        
        losses = {k:v.clone().detach().item() for k,v in lossdict.items()}
        
        generate_batch = torch.cat(generate_batch, dim=1)

        return generate_batch, mixed_loss, losses

    def generate(self, motion_length, num_samples=1):
        self.prior_net.eval()
        self.decoder.eval()
        with torch.no_grad():
            prior_vec = self.tensor_fill((num_samples, self.input_size-1), 0)
            self.prior_net.init_hidden(num_samples)
            self.decoder.init_hidden(num_samples)

            # z_t_p = self.posterior_net.get_normal_noise(num_samples, self.device)
            generate_batch = []
            for i in range(0, motion_length):
                time_counter = i / (motion_length - 1)
                time_counter_vec = self.tensor_fill((num_samples, 1), time_counter)
                condition_vec = time_counter_vec
                # print(prior_vec.shape, condition_vec.shape)
                h = torch.cat((prior_vec, condition_vec), dim=1)
                
                z_t_p, mu_p, logvar_p, h_in_p = self.prior_net(h)

                h_mid = torch.cat((h, z_t_p), dim=1)
                x_pred, _ = self.decoder(h_mid)
                prior_vec = x_pred
                generate_batch.append(x_pred.unsqueeze(1))


            generate_batch = torch.cat(generate_batch, dim=1)

        return generate_batch.cpu().squeeze().reshape(motion_length, -1, 3)



if __name__ == "__main__":
    input_size = 69
    output_size = input_size
    batch_size = 10
    motion_length = 64
    dim_z = 64
    
    lambdas = {"recons": 1.0, "kl": 0.0001}
    
    batch = {"motion": torch.randn(batch_size, motion_length, input_size)}
    model = get_action2motion(input_size, output_size, batch_size, "cpu", lambdas)
    
    x_pred, mixed_loss, losses = model(batch)
    print(x_pred.shape)
    print(mixed_loss, losses)
    
    x_gen = model.generate(128)
    print(x_gen.shape)