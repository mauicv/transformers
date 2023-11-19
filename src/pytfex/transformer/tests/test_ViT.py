
from pytfex.transformer.make_model import init_from_yml_string
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
          overlap: '0,0'
          in_channels: 1
          hidden_dim: {hdn_dim}
      head:
        type: 'InversePatch'
        params:
          img_size: '28,28'
          patch_size: '7,7'
          overlap: '0,0'
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

def test_ViT2():
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
          overlap: '0,0'
          in_channels: 1
          hidden_dim: {hdn_dim}
      head:
        type: 'InversePatch'
        params:
          img_size: '28,28'
          patch_size: '7,7'
          overlap: '0,0'
          in_channels: 1
          hidden_dim: {hdn_dim}
      layers:
        - num: 0
    """
    model = init_from_yml_string(config)
    t = torch.randn((1, 1, 28, 28))
    t_patched = model.embedder.get_patches(t)
    assert t_patched.shape == (1, 16, 7*7)
    t_image = model.head.get_images(t_patched)
    assert t_image.shape == (1, 1, 28, 28)
    assert torch.allclose(t_image, t)


def test_ViT2():
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
          overlap: '0,0'
          in_channels: 3
          hidden_dim: {hdn_dim}
      head:
        type: 'InversePatch'
        params:
          img_size: '28,28'
          patch_size: '7,7'
          overlap: '0,0'
          in_channels: 3
          hidden_dim: {hdn_dim}
      layers:
        - num: 0
    """
    model = init_from_yml_string(config)
    t = torch.randn((1, 3, 28, 28))
    t_patched = model.embedder.get_patches(t)
    assert t_patched.shape == (1, 16, 3*7*7)
    t_image = model.head.get_images(t_patched)
    assert t_image.shape == (1, 3, 28, 28)
    assert torch.allclose(t_image, t)


def test_ViT3():
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
          overlap: '3,3'
          in_channels: 3
          hidden_dim: {hdn_dim}
      head:
        type: 'InversePatch'
        params:
          img_size: '28,28'
          patch_size: '7,7'
          overlap: '3,3'
          in_channels: 3
          hidden_dim: {hdn_dim}
      layers:
        - num: 0
    """
    model = init_from_yml_string(config)
    t = torch.randn((1, 3, 28, 28))
    t_patched = model.embedder.get_patches(t)
    assert t_patched.shape == (1, 36, 3*7*7)
    t_image = model.head.get_images(t_patched)
    assert t_image.shape == (1, 3, 28, 28)


def test_ViT3():
    hdn_dim = 12
    config = f"""
    type: 'GPT'
    params:
      dropout: 0.5
      hidden_dim: {hdn_dim}
      num_heads: 4
      dropout: 0.5
      embedder:
        type: 'MultiEmbedder'
        params:
          embedders:
            - type: 'LinearEmbedder' 
              params:
                input_dim: 10
                hidden_dim: {hdn_dim}
            - type: 'PositionEmbedder'
              params:
                num_positions: 5
                hidden_dim: {hdn_dim}
      head:
        type: 'ClassificationHead'
        params:
          vocab_size: 10
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
    t = torch.randn((1, 5, 10))
    t = model(t)
    assert t.shape == (1, 5, 10)
