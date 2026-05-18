"""
VLN Fine-grained Memory Module

Components:
1. CrossTransformerBlock: 2-layer Cross-Attention retrieval
2. GateFusion: Adaptive weight fusion
3. VLNMemoryBank: Memory management and query

Configuration:
- feature_dim: 512 (CLIP feature dimension)
- mem_length: 8 (history length)
- retrieval_layers: 1 (Transformer layers)
- fusion_type: 'gate' (fusion method)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from vlnce_baselines.models.graph_utils import calculate_vp_rel_pos_fts, get_angle_fts, MAX_DIST
import numpy as np

class CrossTransformerBlock(nn.Module):
    """
    Cross-Attention Transformer Block
    
    Usage: 
    - Query: current direction class token [B, 1, D]
    - K/V: historical patch tokens [B, T*196, D]
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_norm = nn.LayerNorm(feature_dim)

        # Feed-Forward Network (4x expansion)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        
        # Learnable logit scale (init to log(10) ≈ 2.3) to sharpen attention
        # Higher scale -> sharper attention distribution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10))

    def forward(self,
                query: torch.Tensor,  # (B, N, D)
                k: torch.Tensor,      # (B, M, D)
                v: torch.Tensor,      # (B, M, D)
                attn_bias: torch.Tensor = None, # (B, 1, M) or (1, 1, M)
                return_attn: bool = False
                ) -> torch.Tensor:
        """
        Cross-Attention forward
        
        Args:
            query: [B, N, D] - working memory (current class token)
            k: [B, M, D] - memory bank keys (historical patch tokens)
            v: [B, M, D] - memory bank values (historical patch tokens)
            attn_bias: [B, 1, M] - attention bias (e.g. temporal decay)
            return_attn: bool - whether to return attention weights
        
        Returns:
            [B, N, D] - retrieved features
            (optional) [B, N, M] - attention weights
        """
        q = self.q_proj(query)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Manual Scaled Dot-Product Attention (PyTorch 1.x compatible)
        # Q: [B, N, D], K: [B, M, D], V: [B, M, D]
        d_k = q.size(-1)
        
        # Attention scores: [B, N, M]
        # Apply scaling to sharpen distribution
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        scores = scores * self.logit_scale.exp()
        
        # Apply attention bias (e.g., temporal decay)
        if attn_bias is not None:
            scores = scores + attn_bias
        
        # Attention weights: [B, N, M]
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum: [B, N, D]
        attn_out = torch.matmul(attn_weights, v)

        # Residual + LayerNorm (Post-Norm)
        x = self.attn_norm(query + attn_out)

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        output = self.ffn_norm(x + ffn_out)
        
        if return_attn:
            return output, attn_weights
        return output


class GateFusion(nn.Module):
    """
    Gate Fusion Module
    
    Adaptively fuse current observation and historical memory:
    - scale = sigmoid(Linear(concat(x1, x2)))
    - output = scale * x1 + (1-scale) * x2
    """
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim * 2, dim)
        '''        
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.proj.bias, mean=0.0, std=1e-3)
        '''
        # Mild bias strategy: initial scale ≈ 0.95 (favors current observation while preserving gradient flow)
        # Weight close to zero to avoid initial perturbation
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-4)
        # Bias initialized to 2.0, making sigmoid(2) ≈ 0.88
        # Initially: fused ≈ 0.88 * x1 + 0.12 * x2
        # - Favors current observation for training stability
        # - Preserves historical memory gradient to avoid slow convergence
        # - Model will adjust this ratio during training
        nn.init.constant_(self.proj.bias, 2.0)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, attn_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Gate Fusion forward
        
        Args:
            x1: [*, D] - current observation (class token)
            x2: [*, D] - historical memory (retrieved features)
            attn_weights: [*, M] - attention weights (optional, for debugging)
        
        Returns:
            [*, D] - fused features
        """
        scale = torch.sigmoid(
            self.proj(
                torch.cat([x1, x2], dim=-1)
            )
        )
        fused = scale * x1 + (1 - scale) * x2
        # Debug output disabled
        # if(random.random() < 0.2):
        #     print(f"scale sample: {scale[0,0,5:10]}")
        #     print(f"scale mean: {scale.mean().item():.4f}")
        #     cos_sim = F.cosine_similarity(fused.flatten(0, -2), x1.flatten(0, -2), dim=-1).mean().item()
        #     print(f"cosine_sim(fused, x1): {cos_sim:.4f}")
        #     if attn_weights is not None:
        #         entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean().item()
        #         print(f"attention entropy: {entropy:.4f}")
            
        return fused


class VLNMemoryBank(nn.Module):
    """
    VLN Memory Bank
    
    Features:
    1. Store historical forward patch tokens
    2. Query history using class token
    3. Gate Fusion for feature fusion
    4. FIFO memory management
    
    Configuration:
    - feature_dim: 512
    - mem_length: 8
    - retrieval_layers: 2
    - fusion_type: 'gate'
    """
    def __init__(
        self,
        mem_length,
        retrieval_layers,
        feature_dim=512,
        fusion_type='gate',
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.mem_length = mem_length
        self.fusion_type = fusion_type
        
        # Learnable time decay factor (init to small value 0.1)
        # bias = -abs(time_decay) * step_diff
        self.time_decay = nn.Parameter(torch.ones([]) * 0.1)
        
        # 2-layer CrossTransformerBlock
        self.retrieval_blocks = nn.ModuleList([
            CrossTransformerBlock(feature_dim)
            for _ in range(retrieval_layers)
        ])
        
        # Gate Fusion
        if fusion_type == 'gate':
            self.fusion = GateFusion(feature_dim)
        
        # Position encoding: 7-dim (angle_feat_size=4 + 3 distance features) -> 512-dim
        # Same structure as gmap_pos_embeddings
        self.pos_embeddings = nn.Sequential(
            nn.Linear(7, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        # Position encoding init: zero init (safe start, let model learn position importance gradually)
        nn.init.zeros_(self.pos_embeddings[0].weight)
        nn.init.zeros_(self.pos_embeddings[0].bias)
        
        # Step encoding: embedding lookup (same structure as gmap_step_embeddings)
        # mem_length=8, step_diff range [0, 8], so num_embeddings=9
        self.step_embeddings = nn.Embedding(
            num_embeddings=mem_length + 1,  # [0, mem_length]
            embedding_dim=feature_dim
        )
        # Step encoding init: zero init (safe start, let model learn temporal importance gradually)
        nn.init.zeros_(self.step_embeddings.weight)
        
        # Memory storage: List[Dict]
        # Each item contains: {'features', 'position', 'heading', 'step_id'}
        # Shared globally, reset at each episode
        self.memory_bank = []
    
    def query(
        self,
        class_tokens: torch.Tensor,  # [B, 12, 512]
        cur_position: tuple,          # (x, y, z) current position
        cur_heading: float,           # current heading
        cur_step: int,                # current step
    ) -> torch.Tensor:
        """
        Query memory bank and fuse features
        
        Args:
            class_tokens: [B, 12, 512] - class tokens for 12 directions
            cur_position: (x, y, z) - current agent position
            cur_heading: float - current heading angle
            cur_step: int - current step number
        
        Returns:
            [B, 12, 512] - features fused with history
        """
        B, num_dirs, D = class_tokens.shape
        assert D == self.feature_dim, f"Feature dim mismatch: {D} vs {self.feature_dim}"
        
        # If memory is empty, return directly
        if len(self.memory_bank) == 0:
            return class_tokens
        
        # BUG FIX: Save original class_tokens for GateFusion's x1
        # Position encoding is only for Cross-Attention retrieval, should not pollute original observation during fusion
        original_class_tokens = class_tokens.clone()
        
        # ===== Build position encoding for Query =====
        query_pos_encs = []
        
        for angle_id in range(num_dirs):  # 12 directions
            # 1. Calculate relative angle (consistent with candidate point generation)
            # angle_id=0: 0° (front), angle_id=1: -30°, ..., angle_id=11: +30°
            rel_heading = (-angle_id * np.pi / 6.0) % (2 * np.pi)
            rel_elevation = 0.0  # horizontal direction
            rel_dist = 0.0       # query has no distance info
            
            # 2. Angle features (4-dim)
            ang_fts = get_angle_fts(
                np.array([rel_heading]), 
                np.array([rel_elevation]), 
                angle_feat_size=4
            )  # [1, 4]
            
            # 3. Distance features (3-dim, all zeros)
            dist_fts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # [1, 3]
            
            # 4. Combine into 7-dim position features
            pos_fts = torch.from_numpy(
                np.concatenate([ang_fts, dist_fts], axis=1)
            ).to(class_tokens.device)  # [1, 7]
            
            # 5. Project to 512-dim (shared pos_embeddings)
            pos_enc = self.pos_embeddings(pos_fts)  # [1, 512]
            query_pos_encs.append(pos_enc)
        
        # Stack: [num_dirs, 1, 512]
        query_pos_encs = torch.stack(query_pos_encs, dim=0).squeeze(1)  # [num_dirs, 512]
        
        # Broadcast to batch: [B, num_dirs, 512]
        query_pos_encs = query_pos_encs.unsqueeze(0).expand(B, -1, -1)
        
        # Add to class_tokens (position encoding)
        class_tokens = class_tokens + query_pos_encs  # [B, 12, 512]
        
        # ===== Calculate position and step encoding =====
        hist_features = []
        hist_pos_encs = []
        hist_step_encs = []
        hist_step_diffs = [] # Store step differences for bias
        
        for mem_item in self.memory_bank:
            # 1. Extract historical features
            hist_features.append(mem_item['features'])  # [196, 512]
            
            # 2. Calculate relative position (7-dim)
            mem_pos = mem_item['position']  # (x, y, z)
            
            # Same calculation as gmap
            rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                cur_position, mem_pos, 
                base_heading=cur_heading, 
                base_elevation=0,
                to_clock=True
            )
            
            # Angle features (4-dim)
            ang_fts = get_angle_fts(
                np.array([rel_heading]), 
                np.array([rel_elevation]), 
                angle_feat_size=4
            )  # [1, 4]
            
            # Distance features (3-dim): [euclidean_dist, 0, 0]
            # Note: memory has no graph structure, so shortest_dist and shortest_step are set to 0
            dist_fts = np.array([
                [rel_dist / MAX_DIST, 0, 0]
            ], dtype=np.float32)  # [1, 3]
            
            # Combine into 7-dim position features
            pos_fts = torch.from_numpy(
                np.concatenate([ang_fts, dist_fts], axis=1)
            ).to(class_tokens.device)  # [1, 7]
            
            # Project to 512-dim
            pos_enc = self.pos_embeddings(pos_fts)  # [1, 512]
            # Broadcast to 196 patches
            pos_enc = pos_enc.expand(196, -1)  # [196, 512]
            hist_pos_encs.append(pos_enc)
            
            # 3. Calculate step difference encoding
            step_diff = cur_step - mem_item['step_id']
            # Clamp to [0, mem_length] range
            step_diff = min(max(cur_step - mem_item['step_id'], 0), self.mem_length)
            
            # Store raw step diff for bias calculation (clamped or raw? typically raw or clamped)
            # Use same logic as embedding index for consistency, but float for bias
            hist_step_diffs.append(step_diff)
            
            step_id_tensor = torch.LongTensor([step_diff]).to(class_tokens.device)
            step_enc = self.step_embeddings(step_id_tensor)  # [1, 512]
            # Broadcast to 196 patches
            step_enc = step_enc.expand(196, -1)  # [196, 512]
            hist_step_encs.append(step_enc)
            
        # Stack all history
        hist_patches = torch.stack(hist_features, dim=0)  # [T, 196, 512]
        hist_pos_encs = torch.stack(hist_pos_encs, dim=0)  # [T, 196, 512]
        hist_step_encs = torch.stack(hist_step_encs, dim=0)  # [T, 196, 512]
        
        # ===== ⭐ Calculate Temporal Bias =====
        # hist_step_diffs: [T] values
        # We need bias for [T*196] tokens
        # bias = -abs(decay) * step_diff
        
        # [T, 1]
        T_steps = len(hist_step_diffs)
        step_diff_tensor = torch.tensor(hist_step_diffs, dtype=torch.float32, device=class_tokens.device).unsqueeze(1)
        
        # [T, 196] -> flatten -> [1, T*196]
        step_diff_expanded = step_diff_tensor.expand(T_steps, 196).flatten().unsqueeze(0)
        
        # [1, 1, T*196] suitable for [B, N, T*196] scores
        attn_bias = -torch.abs(self.time_decay) * step_diff_expanded.unsqueeze(0) 
        
        # Temporarily add position and step encoding to features (don't modify original memory)
        hist_patches_with_enc = hist_patches + hist_pos_encs + hist_step_encs  # [T, 196, 512]
        
        # Reshape for attention
        T = hist_patches_with_enc.shape[0]
        hist_patches_with_enc = hist_patches_with_enc.reshape(-1, D).unsqueeze(0)  # [1, T*196, 512]
        
        # ===== Subsequent Cross-Attention logic unchanged =====
        enhanced_tokens = []
        for dir_id in range(num_dirs):
            # query_with_pos: query with position encoding, for Cross-Attention retrieval
            query_with_pos = class_tokens[:, dir_id:dir_id+1, :]  # [B, 1, 512]
            # original_query: original class token, for GateFusion's x1
            original_query = original_class_tokens[:, dir_id:dir_id+1, :]  # [B, 1, 512]
            
            # 2-layer CrossTransformerBlock iterative retrieval
            retrieved = query_with_pos
            attn_weights = None
            for layer_idx, layer in enumerate(self.retrieval_blocks):
                is_last_layer = (layer_idx == len(self.retrieval_blocks) - 1)
                if is_last_layer:
                    # Last layer returns attention weights
                    retrieved, attn_weights = layer(
                        query=retrieved,               # [B, 1, 512]
                        k=hist_patches_with_enc,       # [1, T*196, 512] use encoded features
                        v=hist_patches_with_enc,       # [1, T*196, 512]
                        attn_bias=attn_bias,           # pass temporal bias
                        return_attn=True
                    )
                else:
                    retrieved = layer(
                        query=retrieved,               # [B, 1, 512]
                        k=hist_patches_with_enc,       # [1, T*196, 512]
                        v=hist_patches_with_enc,       # [1, T*196, 512]
                        attn_bias=attn_bias            # pass temporal bias
                    )
            
            # Gate Fusion
            # BUG FIX: Use original_query as x1, not query with position encoding
            if self.fusion_type == 'gate':
                fused = self.fusion(original_query, retrieved, attn_weights)  # [B, 1, 512]
            elif self.fusion_type == 'add':
                fused = (original_query + retrieved) * 0.5
            else:
                fused = retrieved
            
            enhanced_tokens.append(fused)
        
        # [B, 12, 512]
        enhanced_tokens = torch.cat(enhanced_tokens, dim=1)
        return enhanced_tokens
    
    def update(
        self,
        forward_patches: torch.Tensor,  # [B, 196, 512]
        cur_position: tuple,             # (x, y, z) current position
        cur_heading: float,              # current heading
        cur_step: int,                   # current step
    ):
        """
        Update memory bank
        
        Args:
            forward_patches: [B, 196, 512] - forward patch tokens
            cur_position: (x, y, z) - current position
            cur_heading: float - current heading
            cur_step: int - current step
        """
        B = forward_patches.shape[0]
        assert B == 1, "Currently only supports batch_size=1"
        
        # Store complete info (dict structure)
        memory_item = {
            'features': forward_patches.squeeze(0).detach(),  # [196, 512]
            'position': cur_position,  # (x, y, z)
            'heading': cur_heading,
            'step_id': cur_step
        }
        self.memory_bank.append(memory_item)
        
        # FIFO strategy
        if len(self.memory_bank) > self.mem_length:
            self.memory_bank.pop(0)
    
    def reset(self):
        """
        Reset memory bank (called at start of each episode)
        
        Note: Only clear list, don't call torch.cuda.empty_cache()
        Reasons:
        1. PyTorch auto-manages tensor lifecycle and GPU memory
        2. Keep GPU memory cache pool stable, avoid memory fluctuation
        3. Frequent empty_cache() calls degrade performance
        """
        # Simply clear list, PyTorch will auto-reclaim GPU memory when tensor ref count is 0
        self.memory_bank.clear()
    
    def get_memory_size(self):
        """
        Get current memory size
        """
        return len(self.memory_bank)
