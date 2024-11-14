import torch.nn as nn
import torch.nn.functional as F
from models.modules.mlp import MLP
from models.hifigan.hifiwrapper import HifiGan


class ReferenceCrossAttention(nn.Module):
    def __init__(
        self,
        lm_model: nn.Module,
        fusion: nn.Module,
        film: nn.Module,
        emb_dim=1024,
        lm_first = False,
        hifi_path = None,
        hifi_config = None,
    ):
        super().__init__()
        ## kmeans parameter
        self.lm = lm_model
        if emb_dim != 1024:
            print("using mlp ....")
            self.mlp_front = MLP(1024, 2 * 1024, emb_dim)
            self.mlp_back = MLP(emb_dim, 2 * emb_dim, 1024)
        else:
            self.mlp_front = nn.Identity()
            self.mlp_back = nn.Identity()
        self.fusion = fusion
        self.film = film
        self.fusion_norm = nn.GroupNorm(1, emb_dim, 1e-8)
        self.lm_first = lm_first
        if hifi_path is not None:
            self.hifi = HifiGan(hifi_path, hifi_config)
        print(f"model parameters {sum(p.numel() for p in self.parameters())}")

    def calc_reg_loss(self, prediction, target):
        l1_loss = F.l1_loss(prediction, target, reduction="mean")
        l2_loss = 0.5 * F.mse_loss(prediction, target, reduction="mean")
        return l1_loss * 0.5 + l2_loss * 0.5, l1_loss, l2_loss

    def forward(self, discrete, clean, regi, inference=False):
        """
        Args:
            discrete: [B, T, E] wavlm embedding
            clean : [B, T, E]
            regi: [B, T', E]
        Returns:
            - the probability of shape (B, C, T')
            - the clean codec
        """
        if inference:
            return self.inference(discrete, regi)
        emb, regi_emb = (self.mlp_front(discrete), self.mlp_front(regi))
        if self.lm_first:
            aux = self.lm(aux)
            aux = self.fusion(emb, regi_emb)[0]
            aux = self.film(emb, aux)
            aux = self.fusion_norm(aux.transpose(1, 2)).transpose(1, 2)  # [B, T, E]
            aux = self.mlp_back(aux)
        else:
            aux = self.fusion(emb, regi_emb)[0]
            aux = self.film(emb, aux)
            aux = self.fusion_norm(aux.transpose(1, 2)).transpose(1, 2)  # [B, T, E]
            aux = self.lm(aux)  # [B, T, E]
            aux = self.mlp_back(aux)
        if clean is not None:
            loss, _, _ = self.calc_reg_loss(aux, clean)
            return loss, aux
        else:
            return aux
    
    def recon_audio(self, discrete, regi):
        """
        Args:
            discrete: [B, T, E] discrete emb
            regi: [B, T, E] reference embedding
        Returns:
            Audio: [B, T] 
        """ 
        aux = self.forward(discrete, None, regi)
        return self.hifi(aux)

