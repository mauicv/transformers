def get_sparse_mlp_gpt_config(config):

    return f"""
        type: 'GPT'
        params:
            dropout: {config.dropout}
            hidden_dim: {config.hdn_dim}
            num_heads: {config.num_heads}
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
            head:
                type: 'ClassificationHead'
                params:
                    hidden_dim: {config.hdn_dim}
                    vocab_size: {config.vcb_size}
            layers:
                -   num: {config.num_layers}
                    type: 'TransformerLayer'
                    params:
                        hidden_dim: {config.hdn_dim}
                        attn:
                            type: 'Attention'
                            params:
                                hidden_dim: {config.hdn_dim}
                                num_heads: {config.num_heads}
                                dropout: {config.dropout}
                        mlp:
                            type: 'SparseMLP'
                            params:
                                hidden_dim: {config.hdn_dim}
                                dropout: {config.dropout}
                                k: {config.k}
        """