import torch.nn as nn
import dac
import torch
from einops import rearrange


class Dac(nn.Module):
    def __init__(self, layer=6):
        super().__init__()
        print("starct Dac initialization")
        self.dac = dac.DAC.load(dac.utils.download(model_type="16khz"))
        self.dac.eval()
        self.layer = layer
        print("dac initialized successfully")
        pass

    @torch.no_grad()
    def codes(self, x, fs=16000, preprocess=True, layer = None):
        """
        :params x [B,T]
        :return codebook [B, T', N] where N stands for the number of codebooks
        """
        with torch.no_grad():
            x = rearrange(x, "b t -> b 1 t")
            if preprocess:
                x = self.dac.preprocess(x, fs)
                res = self.dac.encode(x)[1]
                if layer is not None:
                    return res[:, : layer].transpose(1, 2)
            return res[:, : self.layer].transpose(1, 2)

    @torch.no_grad()
    def encode(self, x, fs=16000, preprocess=True):
        """
        :params x [B,T]
        :return embedding [B, T, D] where D is the embedding dimension
        """
        with torch.no_grad():
            x = rearrange(x, "b t -> b 1 t")
            if preprocess:
                x = self.dac.preprocess(x, fs)
            return self.dac.encode(x)[0].transpose(1, 2)

    @torch.no_grad()
    def infer(self, codes):
        """
        :params codes: [B, T, N]
        :return reconstructed audio [B, T]
        """
        with torch.no_grad():
            z = self.dac.quantizer.from_codes(codes.transpose(1, 2))[0]
            return self.dac.decode(z).squeeze(1)

    @torch.no_grad()
    def decode(self, emb):
        """
        Arguments:
            emb: [B, T, D]
        Returns:
            Audio: [B, T]
        """
        with torch.no_grad():
            return self.dac.decode(emb.transpose(1, 2)).squeeze(1)


if __name__ == "__main__":
    import torchaudio

    wav, rate = torchaudio.load("/home/bltang/Downloads/test/clean.wav")
    audio = wav[:, :48080]

    codec = Dac()

    codes = codec.codes(audio)
    print("codes shape, ", codes.shape)
    res_audio = codec.infer(codes)
    print("res_audio ", res_audio.shape)
    torchaudio.save("out.wav", res_audio, sample_rate=16000)

    emb = codec.encode(audio)
    print("emb shape ", emb.shape)

    res_audio = codec.decode(emb)
    print("res audio", res_audio.shape)
    torchaudio.save("out1.wav", res_audio, sample_rate=16000)

    pass
