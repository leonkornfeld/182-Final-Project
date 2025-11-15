import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm




def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            out_dim=conf.out_dim,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, out_dim, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self.out_dim = out_dim
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, out_dim)

    @staticmethod
    def _combine(xs_b, ys_b):
        """
        Interleave x and y into a single sequence. No padding needed:
        xs_b, ys_b both have shape (B, points, dim) with the SAME dim.
        """
        bsize, points, dim = xs_b.shape
        # stack along a new axis=2: [..., 2, dim] then reshape to (B, 2*points, dim)
        zs = torch.stack((xs_b, ys_b), dim=2).view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        """
        xs:   (B, points, n_dims)
        ys:   (B, points, n_dims)  # full signals, same n_dims
        inds: indices into points to select which x-positions to return
              (default: all)
        Returns:
          (B, len(inds), out_dim)
        """
        B, P, D = xs.shape
        assert D == self.n_dims, f"xs dim {D} != n_dims {self.n_dims}"
        assert ys.shape == (B, P, D), "ys must match xs shape (full signal vectors)"

        if inds is None:
            inds = torch.arange(P, device=xs.device)
        else:
            inds = torch.as_tensor(inds, device=xs.device)
            if inds.max().item() >= P or inds.min().item() < 0:
                raise ValueError("inds out of bounds for number of points")

        # Interleave [x1,y1,x2,y2,...]  -> (B, 2P, n_dims)
        zs = self._combine(xs, ys)

        # Project to embeddings and run GPT-2 with causal mask
        embeds = self._read_in(zs)                               # (B, 2P, n_embd)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state  # (B, 2P, n_embd)

        # Predict at x positions only: positions 0,2,4,... -> ::2
        preds_all = self._read_out(output)                       # (B, 2P, out_dim)
        preds_x = preds_all[:, ::2, :]                           # (B, P, out_dim)

        return preds_x[:, inds, :]                               # (B, len(inds), out_dim)


