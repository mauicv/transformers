
from pytfex.model.make_model import init_from_yml_string
import torch

def test_ViT():
    hdn_dim = 12
    config = f"""
    type: 'GPT'
    params:
      dropout: 0.5
      hidden_dim: {hdn_dim}
      num_heads: 4
      dropout: 0.5
      embedder:
        type: 'PatchEmbedder'
        params:
          img_size: '28,28'
          patch_size: '7,7'
          in_channels: 1
          hidden_dim: {hdn_dim}
      head:
        type: 'InversePatch'
        params:
          img_size: '28,28'
          patch_size: '7,7'
          in_channels: 1
          hidden_dim: {hdn_dim}
      layers:
        - num: 2
          type: 'TransformerLayer'
          params:
            hidden_dim: {hdn_dim}
            attn:
              type: 'Attention'
              params:
                hidden_dim: {hdn_dim}
                num_heads: 4
                dropout: 0.5
            mlp:
              type: 'MLP'
              params:
                hidden_dim: {hdn_dim}
                dropout: 0.5
    """
    model = init_from_yml_string(config)
    t1 = torch.randn((1, 1, 28, 28))
    t1 = model.embedder.get_patches(t1)
    assert t1.shape == (1, 16, 7*7)
    t2 = model(t1)
    assert t2.shape == (1, 16, 49)
    t3 = model.head.get_images(t1)
    assert t3.shape == (1, 1, 28, 28)