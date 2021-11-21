import torch
import torch.nn as nn
import numpy as np

class MIOLoss(nn.Module):
    def __init__(self, batch_size, p_temperature = 0.5, n_temperature = 0.5, lambda_loss = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.p_temperature = p_temperature
        self.n_temperature = n_temperature
        self.lambda_loss = lambda_loss
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
            mask[i, i] = 0
        return mask

    def forward(self, z1, z2):
        N = 2 * self.batch_size #* self.world_size
        z1 = torch.nn.functional.normalize(z1, dim=-1, p=2)
        z2 = torch.nn.functional.normalize(z2, dim=-1, p=2)
        z = torch.cat((z1, z2), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))# / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(-1) / self.p_temperature
        negative_samples = sim[self.mask].reshape(-1) / self.n_temperature
        
        #MIO
        posloss = self.criterion(positive_samples, torch.from_numpy(np.array([1]*positive_samples.shape[0])).to(positive_samples.device, positive_samples.dtype))
        negloss = self.criterion(negative_samples, torch.from_numpy(np.array([0]*negative_samples.shape[0])).to(negative_samples.device, negative_samples.dtype))
        loss = posloss + negloss

        # TO PREVENT DIVERGENCE
        sim_loss = torch.nn.functional.mse_loss(z1, z2)

        loss = loss + self.lambda_loss * sim_loss

        return loss

