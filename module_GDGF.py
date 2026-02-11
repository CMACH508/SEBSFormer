import torch
import torch.nn as nn
import torch.nn.functional as F

class GDGF(nn.Module):
    def __init__(self, feature_dim, num_regions=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_proj = nn.Linear(feature_dim, feature_dim)
        self.adj_scaler = nn.Sequential(
            nn.Linear(feature_dim, 1), 
            nn.Sigmoid())
        self.region_proj = nn.Linear(feature_dim, feature_dim)
        self.global_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x, lead_regions=None):
        assert not torch.isnan(x).any(), "NaN!"
        batch_size = x.size(0)
        proj_feat = self.feature_proj(x)
        x_normalized = F.normalize(proj_feat, p=2, dim=-1)
        sim_matrix = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        adj_scaler = self.adj_scaler(proj_feat.unsqueeze(2).expand(-1, -1, sim_matrix.shape[-1], -1))
        adj_scaler = adj_scaler.squeeze(-1)
        adj_scaler = (adj_scaler + adj_scaler.transpose(1, 2)) / 2
        adj_scaler = adj_scaler.clamp(min=1e-6, max=1.0)
        adj_matrix = adj_scaler * sim_matrix
        adj_matrix = adj_matrix.clamp(min=1e-6)
        degree = torch.sum(adj_matrix, dim=2, keepdim=True)
        norm_adj = adj_matrix / (degree.clamp_min(1e-6))
        region_feat = torch.bmm(norm_adj, proj_feat)
        region_feat = self.region_proj(region_feat)
        global_feat = region_feat.mean(dim=1, keepdim=True)
        global_feat = self.global_proj(global_feat)
        input_global = x.mean(dim=1, keepdim=True)
        output = self.norm(global_feat + input_global)
        
        return output, adj_matrix.detach()