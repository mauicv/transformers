def get_basic_wee_soupy_config(config):

    return f"""
        type: 'WeeSoupyBoy'
        params:
            dropout: {config.dropout}
            hidden_dim: {config.hdn_dim}
            num_heads: {config.num_heads}
            dropout: {config.dropout}
            blk_size: {config.blk_size}
            depth: {config.depth}

            attention_soup:
                type: 'RelativeAttentionSoup'
                params:
                    hidden_dim: {config.hdn_dim}
                    num_heads: {config.num_heads}
                    num_experts: {config.num_experts}
                    num_positions: {config.blk_size}
                    dropout: {config.dropout}
            mlp_soup:
                type: 'ExpertChoiceSoup'
                params:
                    hidden_dim: {config.hdn_dim}
                    experts:
                        -   num: {config.num_experts}
                            type: 'MLP'
                            params:
                                hidden_dim: {config.hdn_dim}
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
        """