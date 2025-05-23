# Placeholder for flow_matching_model.py

import torch
import torch.nn as nn
import math

# TODO: Implement Flow Matching model (referencing flow_matching library)

class PositionalEmbedding(nn.Module):
    """Positional embedding for time, similar to Transformer."""
    def __init__(self, dim, max_len=5000):
        super().__init__()
        self.dim = dim # Store dim as an instance attribute
        # Compute the positional encodings once in log space.
        # The original Transformer PE is not directly used here, switched to simpler sin/cos embedding.
        # pe = torch.zeros(max_len, dim)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0) # (1, max_len, dim)
        # self.register_buffer('pe', pe) # Not using this precomputed PE matrix for now

    def forward(self, t):
        """
        Args:
            t (Tensor): time steps, shape (batch_size,) or (batch_size, 1).
                       Expected to be scaled to an integer range for embedding lookup if direct.
                       Or, if t is float [0,1], we can scale it to map to `max_len`.
        Returns:
            Tensor: positional embeddings, shape (batch_size, dim)
        """
        # Simpler time embedding (often used in diffusion models)
        # t is expected to be of shape (B,)
        if t.ndim == 1:
            t = t.unsqueeze(-1) # (B, 1)
        
        device = t.device
        # half_dim should use self.dim
        if self.dim == 0: # Avoid division by zero if dim is 0, though unlikely
            return torch.zeros_like(t).repeat(1,0) # return (B,0) tensor
        if self.dim == 1: # Special case for dim=1 to avoid log(10000)/0 if half_dim-1 becomes 0
            # For dim=1, can just use sin(t) or a linear projection of t.
            # Or, for simplicity, just return t or sin(t) directly for dim=1 case.
            # This implementation of PE works best for dim >= 2.
            # Let's make it always have at least one sin and one cos term if dim >= 2
            # Fallback for dim=1 (e.g. just sin(t)) or ensure dim is always even / >=2 for this PE type.
            # For robustness in case dim=1 is passed, though not typical for PE:            
            embeddings = torch.sin(t * math.pi) # simple periodic embedding for dim=1
            return embeddings
            
        half_dim = self.dim // 2
        
        # Denominator for div_term in Transformer PE: (D/2 - 1) or (D-2)/2 for 0-indexed
        # For a simpler embedding here: log(max_period) / (half_dim -1)
        # Using math.log(10000) as max_period_log_val like in many diffusion models
        if half_dim <= 1 and self.dim > 1: # e.g. if self.dim is 2 or 3, half_dim is 1
            # Avoid division by zero in log(10000)/(half_dim-1)
            # Use a simpler scaling for few dimensions if half_dim is 1
            # For self.dim = 2, half_dim = 1. Embeddings will be [sin(t*c), cos(t*c)]
            # Let's use a fixed frequency for the base for simplicity if half_dim is 1.
            # When half_dim=1, torch.arange(half_dim) is just tensor([0.])
            # So embeddings becomes exp(0 * -log_val) = 1.0
            # Then t * embeddings is just t.
            # So we'd get [sin(t), cos(t)] if no further scaling
            # To make it more like standard PE, let's just use a single frequency factor
            if self.dim == 2:
                 freq = torch.tensor([1.0], device=device) # single frequency
                 embeddings = t * freq 
                 embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
                 return embeddings
            elif self.dim == 3: # half_dim = 1
                 freq = torch.tensor([1.0], device=device)
                 embeddings = t * freq 
                 embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
                 embeddings = torch.nn.functional.pad(embeddings, (0,1)) # Pad for odd dim
                 return embeddings
        
        # Original logic for half_dim > 1
        div_term_log = math.log(10000) / (half_dim - 1) # Denominator is half_dim -1 for 0 to half_dim-1 terms
        frequencies = torch.exp(torch.arange(half_dim, device=device) * -div_term_log)
        
        # t * frequencies will be (B, half_dim)
        angular_terms = t * frequencies
        embeddings = torch.cat((angular_terms.sin(), angular_terms.cos()), dim=-1) # (B, self.dim or self.dim-1)
        
        if self.dim % 2 == 1 and embeddings.shape[1] < self.dim : # Zero pad if dim is odd and concat resulted in dim-1
            embeddings = torch.nn.functional.pad(embeddings, (0,1))
        return embeddings

class VelocityFieldModel(nn.Module):
    """MLP model to learn the velocity field v_psi(z, t)."""
    def __init__(self, latent_dim, time_embed_dim=128, hidden_dims=None, activation_fn=nn.SiLU):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 512] # Default MLP structure
        
        self.time_embed_dim = time_embed_dim
        self.time_embedding = PositionalEmbedding(dim=time_embed_dim)
        
        layers = []
        current_dim = latent_dim + time_embed_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation_fn()) # Swish/SiLU is common
            current_dim = h_dim
        
        layers.append(nn.Linear(current_dim, latent_dim)) # Output matches latent_dim
        
        self.network = nn.Sequential(*layers)

    def forward(self, z, t):
        """
        Args:
            z (Tensor): Latent codes, shape (batch_size, latent_dim).
            t (Tensor): Time steps, shape (batch_size,) or (batch_size, 1). Values in [0, 1].
        Returns:
            Tensor: Estimated velocity, shape (batch_size, latent_dim).
        """
        if t.ndim == 1:
            t = t.unsqueeze(1)
        if t.size(1) != 1:
            raise ValueError(f"Time tensor t expected to be (B, 1) or (B,), got {t.shape}")
        
        t_embed = self.time_embedding(t)
        zt_concat = torch.cat([z, t_embed], dim=1)
        velocity = self.network(zt_concat)
        return velocity

if __name__ == '__main__':
    print("Testing VelocityFieldModel...")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim_test = 100 # Matches VAE latent dim
    batch_size_test = 16
    time_embed_dim_test = 64
    hidden_dims_test = [256, 256]

    model = VelocityFieldModel(
        latent_dim=latent_dim_test,
        time_embed_dim=time_embed_dim_test,
        hidden_dims=hidden_dims_test
    ).to(test_device)
    print(model)

    dummy_z = torch.randn(batch_size_test, latent_dim_test, device=test_device)
    dummy_t = torch.rand(batch_size_test, device=test_device) # Time values in [0, 1]

    print(f"\nInput z shape: {dummy_z.shape}")
    print(f"Input t shape: {dummy_t.shape}")

    try:
        output_velocity = model(dummy_z, dummy_t)
        print(f"Output velocity shape: {output_velocity.shape}")
        assert output_velocity.shape == (batch_size_test, latent_dim_test), "Output shape mismatch!"
        print("\nVelocityFieldModel test passed!")
    except Exception as e:
        print(f"\nError during VelocityFieldModel test: {e}")
        import traceback
        traceback.print_exc() 