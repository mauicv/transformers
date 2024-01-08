def get_mof_gpt_config(config):
    return f"""
        type: 'GPT'
        params:
            dropout: {config.dropout}
            hidden_dim: {config.hdn_dim}
            num_heads: 4
            dropout: {config.dropout}
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
                                num_groups: {config.num_groups}
                                k: {config.k}
                                model:
                                    type: 'Attention'
                                    params:
                                        hidden_dim: {int(config.hdn_dim/config.num_groups)}
                                        num_heads: 8
                                        dropout: {config.dropout}
                        mlp:
                            type: 'MoF'
                            params:
                                hidden_dim: {config.hdn_dim}
                                num_groups: {config.num_groups}
                                k: {config.k}
                                model:
                                    type: 'MLP'
                                    params:
                                        hidden_dim: {int(config.hdn_dim/config.num_groups)}
                                        intermediate_dim: {config.mlp_hdn_dim}
                                        dropout: {config.dropout}
            head:
                type: 'ClassificationHead'
                params:
                    hidden_dim: {config.hdn_dim}
                    vocab_size: {config.vcb_size}
        """
