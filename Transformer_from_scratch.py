
import torch
import torch.nn as nn
import math


# ============================================================
# 1. SELF-ATTENTION
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads

        # Number of dimensions assigned to each attention head
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, \
            "Embed size needs to be divisible by heads"

        # Learnable linear transformations for Values, Keys and Queries
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Combines the outputs from all attention heads
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):

        # Get batch size
        N = query.shape[0]

        # Get sequence lengths
        value_len = values.shape[1]
        key_len = keys.shape[1]
        query_len = query.shape[1]

        # Split embeddings into multiple attention heads
        # Shape: (N, sequence_length, heads, head_dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Apply learnable linear transformations
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate Query-Key similarity scores
        # Output shape: (N, heads, query_len, key_len)
        energy = torch.einsum(
            "nqhd,nkhd->nhqk",
            [queries, keys]
        )

        # Apply attention mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Scale scores and convert them into attention probabilities
        attention = torch.softmax(
            energy / math.sqrt(self.head_dim),
            dim=3
        )

        # Multiply attention weights by Values
        # This creates a weighted combination of the Values
        out = torch.einsum(
            "nhqk,nkhd->nqhd",
            [attention, values]
        )

        # Combine all attention heads into one embedding
        out = out.reshape(
            N,
            query_len,
            self.heads * self.head_dim
        )

        # Final linear transformation
        out = self.fc_out(out)

        return out


# ============================================================
# 2. TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        # Multi-head attention
        self.attention = SelfAttention(embed_size, heads)

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        # Multi-head attention
        attention = self.attention(value, key, query, mask)

        # First residual connection, normalization and dropout
        x = self.dropout(
            self.norm1(attention + query)
        )

        # Feed-forward network
        forward = self.feed_forward(x)

        # Second residual connection, normalization and dropout
        out = self.dropout(
            self.norm2(forward + x)
        )

        return out


# ============================================================
# 3. ENCODER
# ============================================================

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device

        # Converts input token IDs into embeddings
        self.word_embedding = nn.Embedding(
            src_vocab_size,
            embed_size
        )

        # Gives each position in the sequence a positional embedding
        self.position_embedding = nn.Embedding(
            max_length,
            embed_size
        )

        # Stack multiple Transformer blocks
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        # N = batch size
        # seq_length = number of tokens
        N, seq_length = x.shape

        # Generate position indices for every token
        positions = torch.arange(
            0, seq_length, device=x.device
        ).unsqueeze(0).expand(N, seq_length)

        # Add word embeddings and positional embeddings
        out = self.dropout(
            self.word_embedding(x) +
            self.position_embedding(positions)
        )

        # Pass the sequence through every Transformer block
        for layer in self.layers:

            # Encoder uses self-attention:
            # Values, Keys and Queries all come from the same input
            out = layer(out, out, out, mask)

        return out


# ============================================================
# 4. DECODER BLOCK
# ============================================================

class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        forward_expansion,
        dropout,
        device
    ):
        super(DecoderBlock, self).__init__()

        # Masked self-attention for the decoder
        self.attention = SelfAttention(embed_size, heads)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_size)

        # Cross-attention and feed-forward processing
        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):

        # Masked self-attention:
        # The decoder cannot look at future target tokens
        attention = self.attention(
            x, x, x, trg_mask
        )

        # Residual connection, normalization and dropout
        query = self.dropout(
            self.norm(attention + x)
        )

        # Cross-attention:
        # Decoder queries attend to encoder Keys and Values
        out = self.transformer_block(
            value,
            key,
            query,
            src_mask
        )

        return out


# ============================================================
# 5. DECODER
# ============================================================

class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super(Decoder, self).__init__()

        self.device = device

        # Converts target token IDs into embeddings
        self.word_embedding = nn.Embedding(
            trg_vocab_size,
            embed_size
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            max_length,
            embed_size
        )

        # Stack multiple Decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    forward_expansion,
                    dropout,
                    device
                )
                for _ in range(num_layers)
            ]
        )

        # Converts final embeddings into vocabulary scores
        self.fc_out = nn.Linear(
            embed_size,
            trg_vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):

        N, seq_length = x.shape

        # Generate positional indices
        positions = torch.arange(
            0, seq_length, device=x.device
        ).unsqueeze(0).expand(N, seq_length)

        # Add word and positional embeddings
        x = self.dropout(
            self.word_embedding(x) +
            self.position_embedding(positions)
        )

        # Pass through each Decoder block
        for layer in self.layers:

            # x is the decoder input
            # enc_out provides the encoder's Keys and Values
            x = layer(
                x,
                enc_out,
                enc_out,
                src_mask,
                trg_mask
            )

        # Convert decoder embeddings into vocabulary scores
        out = self.fc_out(x)

        return out


# ============================================================
# 6. COMPLETE TRANSFORMER
# ============================================================

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        # Create the Encoder
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        # Create the Decoder
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

    # --------------------------------------------------------
    # SOURCE MASK
    # --------------------------------------------------------

    def make_src_mask(self, src):

        # Identify non-padding source tokens
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask.to(self.device)

    # --------------------------------------------------------
    # TARGET MASK
    # --------------------------------------------------------

    def make_trg_mask(self, trg):

        N, trg_len = trg.shape

        # Create a lower triangular matrix.
        # This prevents each position from seeing future tokens.
        causal_mask = torch.tril(
            torch.ones(
                (trg_len, trg_len),
                device=trg.device
            )
        ).bool()

        # Identify non-padding target tokens
        padding_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # Combine causal and padding masks
        trg_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        trg_mask = trg_mask & padding_mask

        return trg_mask.to(self.device)

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    def forward(self, src, trg):

        # Create source and target masks
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # Pass source sequence through the Encoder
        enc_src = self.encoder(src, src_mask)

        # Pass target sequence and encoder output through Decoder
        out = self.decoder(
            trg,
            enc_src,
            src_mask,
            trg_mask
        )

        return out
