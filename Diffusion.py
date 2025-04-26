import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock3D(nn.Module):
    """3D Residual Block with time conditioning."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Ensure residual connection matches output channels
        self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        # x: [B, C, D, H, W]
        # t: [B, time_emb_dim] projected conditioning
        h = F.relu(self.bn1(self.conv1(x)))

        # Project time embedding and add
        time_emb = F.relu(self.time_mlp(t))
        # Add time embedding across channel dimension, need to unsqueeze for D, H, W
        # Shape becomes [B, out_channels, 1, 1, 1] for broadcasting
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        h = self.dropout(h)
        h = F.relu(self.bn2(self.conv2(h)))

        # Add residual connection
        return h + self.residual_conv(x)

class DownBlock3D(nn.Module):
    """3D Downsampling block."""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout):
        super().__init__()
        self.res_block = ResidualBlock3D(in_channels, out_channels, time_emb_dim, dropout)
        # Downsample spatial dimensions (D, H, W) by stride 2
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t):
        x = self.res_block(x, t)
        x = self.downsample(x)
        return x

class UpBlock3D(nn.Module):
    """3D Upsampling block with skip connection."""
    def __init__(self, in_channels_up, skip_channels, out_channels, time_emb_dim, dropout):
        super().__init__()
        self.in_channels = in_channels_up
        self.out_channels = out_channels

        # Upsamples the input from the layer below (D, H, W). Output channels = out_channels
        self.upsample = nn.ConvTranspose3d(in_channels_up, out_channels, kernel_size=4, stride=2, padding=1)

        # ResBlock processes the concatenated tensor
        # Input channels = channels from skip connection + channels after upsampling
        self.res_block = ResidualBlock3D(skip_channels + out_channels, out_channels, time_emb_dim, dropout)

    def forward(self, x, skip_x, t):
        # x: Input from layer below (e.g., [B, in_channels_up, D/2, H/2, W/2])
        # skip_x: Input from corresponding DownBlock ([B, skip_channels, D, H, W])
        # t: Projected conditioning [B, time_emb_dim]

        x = self.upsample(x) # Shape: [B, out_channels, D', H', W'] (D',H',W' might differ slightly from D,H,W)

        # Handle potential size mismatch after ConvTranspose3d
        diffD = skip_x.size()[2] - x.size()[2]
        diffH = skip_x.size()[3] - x.size()[3]
        diffW = skip_x.size()[4] - x.size()[4]

        x = F.pad(x, [diffW // 2, diffW - diffW // 2,   # Pad W
                       diffH // 2, diffH - diffH // 2,   # Pad H
                       diffD // 2, diffD - diffD // 2])  # Pad D

        x = torch.cat([skip_x, x], dim=1) # Shape: [B, skip_channels + out_channels, D, H, W]
        x = self.res_block(x, t)          # Shape: [B, out_channels, D, H, W]
        return x


class UNetPromptConditional3D(nn.Module):
    def __init__(self, in_channels, out_channels, prompt_embedder, prompt_tokenizer, dim=64, dim_mults=(1, 2, 4, 8), time_embed_dim=128, dropout=0.1, device='cpu'):
        super().__init__()

        # --- Embeddings ---
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_embed_dim),
            nn.GELU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.device = device

        # Prompt embedding processing MLP
        self.prompt_tokenizer = prompt_tokenizer
        self.prompt_embedder = prompt_embedder
        

        # MLP to combine time and processed prompt embeddings
        # Output dimension matches the one used in Residual Blocks (time_embed_dim)
        combined_cond_dim = time_embed_dim + prompt_embedder.config.hidden_size
        
        self.combined_cond_mlp = nn.Sequential(
            nn.Linear(combined_cond_dim, time_embed_dim), # Project combined to time_embed_dim
            nn.GELU()
        )
        # --- End Embeddings ---

        # --- 3D U-Net Architecture (Structure remains the same) ---
        unet_dims = [dim] + [dim * m for m in dim_mults]
        down_in_out = list(zip(unet_dims[:-1], unet_dims[1:]))
        num_resolutions = len(down_in_out)

        self.init_conv = nn.Conv3d(in_channels, dim, kernel_size=3, padding=1)

        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(down_in_out):
            self.downs.append(
                DownBlock3D(dim_in, dim_out, time_embed_dim, dropout)
            )

        mid_dim = unet_dims[-1]
        self.mid_block1 = ResidualBlock3D(mid_dim, mid_dim, time_embed_dim, dropout)
        # Optional: Add 3D attention block here if needed
        self.mid_block2 = ResidualBlock3D(mid_dim, mid_dim, time_embed_dim, dropout)

        self.ups = nn.ModuleList([])
        for i, (dim_out_down, dim_in_down) in enumerate(reversed(down_in_out)):
             in_ch_up = unet_dims[num_resolutions - i]
             skip_ch = dim_out_down
             out_ch = dim_out_down
             self.ups.append(
                 UpBlock3D(in_ch_up, skip_ch, out_ch, time_embed_dim, dropout)
             )

        self.final_conv = nn.Conv3d(dim, out_channels, kernel_size=1)

    def forward(self, x, time, prompts):
        
        t_emb = self.time_mlp(time) # [batch, time_embed_dim]

        p_emb = self.encode_prompts(prompts) # [batch, prompt_embed_dim]

        cond_inputs = [t_emb, p_emb]
        cond_emb = torch.cat(cond_inputs, dim=1) # [batch, time_embed_dim + prompt_proj_dim]

        cond = self.combined_cond_mlp(cond_emb) # [batch, time_embed_dim]
    
        x = x.unsqueeze(1)

        x = self.init_conv(x) # [batch, dim, D, H, W]

        skip_connections = []
        for down_block in self.downs:
            skip_connections.append(x)
            x = down_block(x, cond)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        skip_connections = reversed(skip_connections)
        for up_block, skip_x in zip(self.ups, skip_connections):
             x = up_block(x, skip_x, cond) # Pass final projected conditioning

        out = self.final_conv(x) # [batch, out_channels, D, H, W]
        return out
    
    @torch.no_grad()
    def encode_prompts(self, prompts: list[str], max_length=128):
        self.prompt_embedder.eval()
        
        prompt_batch = self.prompt_tokenizer(
            prompts,
            padding=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )


        input_ids = prompt_batch["input_ids"].to(self.device)
        attention_mask = prompt_batch["attention_mask"].to(self.device)

        outputs = self.prompt_embedder(input_ids=input_ids, attention_mask=attention_mask)
        # Shape: [batch_size, sequence_length, hidden_size]
        last_hidden_states = outputs.last_hidden_state

        masked_states = last_hidden_states * attention_mask.unsqueeze(-1).float()
        sum_states = masked_states.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).unsqueeze(-1).float()
        mean_pooled_embeddings = sum_states / torch.clamp(sum_mask, min=1e-9)

        return mean_pooled_embeddings


# Beta schedules (linear_beta_schedule, cosine_beta_schedule) remain the same. Assume defined above.
def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionProcess:
    # __init__, _extract, q_sample remain the same as in the previous 3D version.
    def __init__(self, timesteps, beta_schedule='cosine', device='cpu'):
        self.timesteps = timesteps
        self.device = device

        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps).to(device)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps).to(device)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, prompts, loss_type="l1"):
       
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Pass prompt embeddings to the model
        predicted_noise = denoise_model(x_noisy, t, prompts)
        noise = noise.unsqueeze(1)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

    @torch.no_grad()
    def p_sample(self, denoise_model, x, t, t_index, prompt_embeddings):
        # x: [B, C, D, H, W] (current noisy voxel grid x_t)
        # t: [B] (timestep)
        # prompt_embeddings: [B, prompt_embed_dim]
        # denoise_model: UNetPromptConditional3D instance
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(torch.sqrt(1.0 / self.alphas), t, x.shape)
       
        # Predict noise using the prompt-conditioned 3D U-Net
        predicted_noise = denoise_model(x, t, prompt_embeddings)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, denoise_model, shape, prompts):
        # shape: tuple (batch_size, channels, depth, height, width)
        # prompt_embeddings: [batch_size, prompt_embed_dim]
        # denoise_model: UNetPromptConditional3D instance
        device = self.betas.device
        batch_size = shape[0]
        assert len(prompts) == batch_size, "Batch size mismatch between shape and prompt_embeddings"

        voxels = torch.randn(shape, device=device)
        voxel_grids = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling loop time step', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # Pass prompt embeddings to p_sample
            voxels = self.p_sample(denoise_model, voxels, t, i, prompts)
            voxels = voxels.squeeze(0) 
            
        voxel_grids.append(voxels.cpu())
        return voxel_grids
