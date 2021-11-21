import torch
import torch.nn as nn
import numpy as np

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size #* self.world_size
        #z_i_ = z_i / torch.sqrt(torch.sum(torch.square(z_i),dim = 1, keepdim = True))
        #z_j_ = z_j / torch.sqrt(torch.sum(torch.square(z_j),dim = 1, keepdim = True))
        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        #labels was torch.zeros(N)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

class MoCoLoss(nn.Module):
    def __init__(self,
                 temperature):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, q, k, queue):
        #print(q.shape, k.shape)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = self.criterion(logits, labels)

        return loss

class SymNegCosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.similarity_f = nn.CosineSimilarity(dim=-1)

    def _neg_cosine_simililarity(self, x, y):
        v = - self.similarity_f(x, y.detach()).mean()
        return v

    def forward(self,
                out1: torch.Tensor,
                out2: torch.Tensor):
        """Forward pass through Symmetric Loss.
            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Expects the tuple to be of the form (z0, p0), where z0 is
                    the output of the backbone and projection mlp, and p0 is the
                    output of the prediction head.
                out1:
                    Output projections of the second set of transformed images.
                    Expects the tuple to be of the form (z1, p1), where z1 is
                    the output of the backbone and projection mlp, and p1 is the
                    output of the prediction head.

            Returns:
                Contrastive Cross Entropy Loss value.
            Raises:
                ValueError if shape of output is not multiple of batch_size.
        """
        _, z1, p1 = out1
        _, z2, p2 = out2

        loss = self._neg_cosine_simililarity(p1, z2) / 2 + \
               self._neg_cosine_simililarity(p2, z1) / 2

        return loss

class BYOLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x1 = torch.nn.functional.normalize(x1, dim=-1, p=2)
        x2 = torch.nn.functional.normalize(x2, dim=-1, p=2)
        return 2 - 2*torch.mean(torch.einsum('nc,nc->n', [x1, x2]),dim=-1)

class BTLoss(nn.Module):
    def __init__(self, batch_size, lambd):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_a, z_b):
        c = z_a.T@z_b
        c.div_(self.batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

class VICRegLoss(nn.Module):
    def __init__(self, batch_size, lambd, mu, nu):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    def forward(self, z_a, z_b):

        D = z_a.shape[1]

        # invariance loss
        sim_loss = torch.nn.functional.mse_loss(z_a, z_b)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(self.relu1(1 - std_z_a))
        std_loss = std_loss + torch.mean(self.relu2(1 - std_z_b))
        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (self.batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_z_a).pow_(2).sum() / D
        cov_loss = cov_loss + self.off_diagonal(cov_z_b).pow_(2).sum() / D
        # loss
        loss = self.lambd * sim_loss + self.mu * std_loss + self.nu * cov_loss

        return loss


class CUMILoss(nn.Module):
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
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(-1) / self.p_temperature
        negative_samples = sim[self.mask].reshape(-1) / self.n_temperature
        #CUMI1
        #labels = torch.from_numpy(np.array([1]*positive_samples.shape[0] + [0]*negative_samples.shape[0])).reshape(-1).to(positive_samples.device, torch.int64).float()
        #labels was torch.zeros(N)
        #logits = torch.cat((positive_samples, negative_samples))
        #loss = self.criterion(logits, labels)
        #CUMI2
        posloss = self.criterion(positive_samples, torch.from_numpy(np.array([1]*positive_samples.shape[0])).to(positive_samples.device, positive_samples.dtype))
        negloss = self.criterion(negative_samples, torch.from_numpy(np.array([0]*negative_samples.shape[0])).to(negative_samples.device, negative_samples.dtype))
        loss = posloss + negloss

        # TO PREVENT DIVERGENCE
        sim_loss = torch.nn.functional.mse_loss(z1, z2)

        loss = loss + self.lambda_loss * sim_loss

        return loss

class MDMIaLoss(nn.Module):
    def __init__(self, batch_size, lambd = 25.0, mu = 25.0, nu = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.kldivloss = nn.KLDivLoss(log_target = False, reduction = 'batchmean')

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_a, z_b):
        D = z_a.shape[-1]

        #KLDiv Loss
        log_z_a = torch.log(z_a)
        log_z_b = torch.log(z_b)
        m = 0.5 * (z_a + z_b)
        js_div_loss = self.kldivloss(log_z_a, m) + self.kldivloss(log_z_b, m)

        # variance loss
        std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        std_loss = torch.mean(self.relu(1 - std_z_a))
        std_loss = std_loss + torch.mean(self.relu(1 - std_z_b))
        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (self.batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_z_a).pow_(2).sum() / D
        cov_loss = cov_loss + self.off_diagonal(cov_z_b).pow_(2).sum() / D
        # loss
        loss = self.lambd * js_div_loss + self.mu * std_loss + self.nu * cov_loss

        return loss

class MDMIbLoss(nn.Module):
    def __init__(self, batch_size, lambd = 25.0, mu = 25.0, nu = 1.0):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.mu = mu
        self.nu = nu
        self.kldivloss = nn.KLDivLoss(log_target = False, reduction = 'batchmean')

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_a, z_b):
        D = z_a.shape[-1]

        #KLDiv Loss
        log_z_a = torch.log(z_a)
        log_z_b = torch.log(z_b)
        m_a = torch.normal(0.0, 1.0, size = z_a.shape)
        m_b = torch.normal(0.0, 1.0, size = z_b.shape)
        div_loss = self.kldivloss(log_z_a, m_a) + self.kldivloss(log_z_b, m_b)

        # variance loss
        # std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
        # std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
        # std_loss = torch.mean(relu(1 - std_z_a))
        # std_loss = std_loss + torch.mean(relu(1 - std_z_b))
        # covariance loss
        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)
        cov_z_a = (z_a.T @ z_a) / (self.batch_size - 1)
        cov_z_b = (z_b.T @ z_b) / (self.batch_size - 1)
        cov_loss = self.off_diagonal(cov_z_a).pow_(2).sum() / D
        cov_loss = cov_loss + self.off_diagonal(cov_z_b).pow_(2).sum() / D
        # loss
        loss = self.lambd * div_loss + self.nu * cov_loss #+ self.mu * std_loss

        return loss

# class MDMIcLoss(nn.Module):
#     def __init__(self, batch_size, lambd = 25.0, mu = 25.0, nu = 1.0):
#         super().__init__()
#         self.batch_size = batch_size
#         self.lambd = lambd
#         self.mu = mu
#         self.nu = nu
#         self.kldivloss = nn.KLDivLoss(log_target = False, reduction = 'batchmean')

#     def off_diagonal(self, x):
#         # return a flattened view of the off-diagonal elements of a square matrix
#         n, m = x.shape
#         assert n == m
#         return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

#     def forward(self, z_a, z_b):
#         D = z_a.shape[-1]

#         #KLDiv Loss
#         log_z_a = torch.log(z_a)
#         log_z_b = torch.log(z_b)
#         m = 0.5 * (z_a + z_b)
#         jsdivloss = self.kldivloss(log_z_a, m) + self.kldivloss(log_z_b, m)

#         # variance loss
#         std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
#         std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)
#         std_loss = torch.mean(relu(1 - std_z_a))
#         std_loss = std_loss + torch.mean(relu(1 - std_z_b))
#         # covariance loss
#         z_a = z_a - z_a.mean(dim=0)
#         z_b = z_b - z_b.mean(dim=0)
#         cov_z_a = (z_a.T @ z_a) / (self.batch_size - 1)
#         cov_z_b = (z_b.T @ z_b) / (self.batch_size - 1)
#         cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D
#         cov_loss = cov_loss + off_diagonal(cov_z_b).pow_(2).sum() / D
#         # loss
#         loss = self.lambd * jsdiv_loss + self.mu * std_loss + self.nu * cov_loss

#         return loss
