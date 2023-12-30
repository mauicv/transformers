def get_moemof_gpt_config(config):
    return f"""
        type: 'GPT'
        params:
            dropout: 0.5
            hidden_dim: {config.hdn_dim}
            num_heads: 4
            dropout: 0.5
            embedder:
                type: 'MultiEmbedder'
                params:
                    embedders:
                        -   type: 'TokenEmbedder' 
                            params:
                                dictionary_size: {config.vcb_size}
                                hidden_dim: {config.hdn_dim}
                        -   type: 'PositionEmbedder'
                            params:
                                num_positions: {config.blk_size}
                                hidden_dim: {config.hdn_dim}
            layers:
                -   num: {config.num_layers}
                    type: 'TransformerLayer'
                    params:
                        hidden_dim: {config.hdn_dim}
                        attn:
                            type: 'MoF'
                            params:
                                hidden_dim: {config.hdn_dim}
                                num_proj: {config.num_proj}
                                k: {config.k}
                                model:
                                    type: 'Attention'
                                    params:
                                        hidden_dim: {int(config.hdn_dim/config.num_proj)}
                                        num_heads: 8
                                        dropout: 0.5
                        mlp:
                            type: 'MoF'
                            params:
                                hidden_dim: {config.hdn_dim}
                                num_proj: {config.num_proj}
                                k: {config.k}
                                model:
                                    type: 'MoE'
                                    params:
                                        hidden_dim: {int(config.hdn_dim/config.num_proj)}
                                        c: {config.c}
                                        experts:
                                            -   num: {config.num_experts}
                                                type: 'MLP'
                                                params:
                                                    hidden_dim: {int(config.hdn_dim/config.num_proj)}
                                                    intermediate_dim: {4*int(config.hdn_dim/config.num_proj)}
                                                    dropout: 0.5
            head:
                type: 'ClassificationHead'
                params:
                    hidden_dim: {config.hdn_dim}
                    vocab_size: {config.vcb_size}
        """