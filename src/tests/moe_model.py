def get_moe_gpt_config(
        vcb_size,
        hdn_dim,
        blk_size,
        num_experts,
        c
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
                            type: 'Attention'
                            params:
                                hidden_dim: {hdn_dim}
                                num_heads: 4
                                dropout: 0.5
                        mlp:
                            type: 'MoEMLP'
                            params:
                                hidden_dim: {hdn_dim}
                                c: {c}
                                experts:
                                    -   num: {num_experts}
                                        type: 'MLP'
                                        params:
                                            hidden_dim: {hdn_dim}
                                            intermediate_dim: {hdn_dim}
                                            dropout: 0.5
            head:
                type: 'ClassificationHead'
                params:
                    hidden_dim: {hdn_dim}
                    vocab_size: {vcb_size}
        """
