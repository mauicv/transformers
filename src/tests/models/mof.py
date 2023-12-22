def get_mof_gpt_config(
        vcb_size,
        hdn_dim,
        blk_size,
        c,
        num_groups,
    ):

    return f"""
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
                        -   type: 'TokenEmbedder' 
                            params:
                                dictionary_size: {vcb_size}
                                hidden_dim: {hdn_dim}
                        -   type: 'PositionEmbedder'
                            params:
                                num_positions: {blk_size}
                                hidden_dim: {hdn_dim}
            layers:
                -   num: 2
                    type: 'TransformerLayer'
                    params:
                        hidden_dim: {hdn_dim}
                        attn:
                            type: 'MoF'
                            params:
                                hidden_dim: {hdn_dim}
                                num_groups: {num_groups}
                                k: {c}
                                model:
                                    type: 'Attention'
                                    params:
                                        hidden_dim: {c*int(hdn_dim/num_groups)}
                                        num_heads: 4
                                        dropout: 0.5
                        mlp:
                            type: 'MoF'
                            params:
                                hidden_dim: {hdn_dim}
                                num_groups: {num_groups}
                                k: {c}
                                model:
                                    type: 'MLP'
                                    params:
                                        hidden_dim: {c*int(hdn_dim/num_groups)}
                                        intermediate_dim: {4*c*int(hdn_dim/num_groups)}
                                        dropout: 0.5
            head:
                type: 'ClassificationHead'
                params:
                    hidden_dim: {hdn_dim}
                    vocab_size: {vcb_size}
        """
