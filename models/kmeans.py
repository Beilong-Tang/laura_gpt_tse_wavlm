import torch
import joblib
import warnings
import torch.nn as nn


class KMeansQuantizer(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
        self.kmeans_model = self.load_kmeans_model(checkpoint_path)

    def emb(self, x):
        """[B,T] -> [B, T, E]"""
        batch, seq = x.shape
        x = x.view(batch * seq)
        center = torch.from_numpy(
            self.kmeans_model.cluster_centers_[x.cpu().numpy()]
        ).to(self.device)
        center = center.view(batch, seq, center.size(-1))
        return center.float()

    def forward(self, x: torch.Tensor):
        """[B,T,E] -> [B, T]"""
        batch, seq, t = x.shape
        x = x.view(batch * seq, -1)
        res = self._extract(x)
        res = res.view(batch, seq)
        return res

    def _extract(self, x):
        """[T,E] -> T"""
        return (
            torch.from_numpy(self.kmeans_model.predict(x.double().cpu().numpy()))
            .to(self.device)
            .long()
        )

    @property
    def vocab_size(self) -> int:
        return self.kmeans_model.n_clusters

    @property
    def device(self):
        return self._float_tensor.device

    @staticmethod
    def load_kmeans_model(checkpoint_path: str):
        with open(checkpoint_path, "rb") as fd:
            with warnings.catch_warnings():
                # produces lots of version warnings which can be annoying when we have many workers
                warnings.simplefilter("ignore")
                kmeans_model = joblib.load(fd)
                # some of the GSLM checkpoints (CPC) were saved under a different scikit version
                if not hasattr(kmeans_model, "_n_threads"):
                    kmeans_model._n_threads = 40

        kmeans_model.verbose = False
        return kmeans_model
