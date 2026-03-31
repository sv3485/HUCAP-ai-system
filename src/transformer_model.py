"""Multi-task Protein Transformer classifier with attention-based feature fusion."""

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ProteinTransformerClassifier(nn.Module):
    """
    Research-Grade Multi-Task Classifier with Feature Fusion.
    Integrates ESM2 embeddings + PSSM (20-dim) + Physicochemical features (3-dim).
    """

    def __init__(
        self,
        model_name: str,
        num_terms_dict: Dict[str, int],
        dropout: float = 0.3,
        unfreeze_last_n_layers: int = 2,
        aspects: list = None,
    ) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.encoder = EsmModel.from_pretrained(model_name, attn_implementation="eager")
        except Exception:
            self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.aspects = aspects or list(num_terms_dict.keys())

        esm_hidden_size = self.encoder.config.hidden_size

        # Additional feature dimensions: PSSM (20) + Phys (3)
        self.pssm_dim = 20
        self.phys_dim = 3

        fused_dim = esm_hidden_size + self.pssm_dim + self.phys_dim

        # Freeze encoder completely first
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers to allow fine-tuning
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer
            for layer in layers[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.dropout = nn.Dropout(dropout)

        # Structured Attention-based Feature Fusion
        self.esm_proj = nn.Linear(esm_hidden_size, esm_hidden_size)
        self.pssm_proj = nn.Linear(self.pssm_dim, esm_hidden_size)
        self.phys_proj = nn.Linear(self.phys_dim, esm_hidden_size)
        
        # Learned attention context vector
        self.fusion_query = nn.Parameter(torch.randn(esm_hidden_size, 1))
        nn.init.xavier_uniform_(self.fusion_query)

        self.fusion_norm = nn.LayerNorm(esm_hidden_size)
        self.activation = nn.GELU()

        # Multi-task heads for each GO aspect
        self.heads = nn.ModuleDict()
        for aspect, num_terms in num_terms_dict.items():
            self.heads[aspect] = nn.Sequential(
                nn.Linear(esm_hidden_size, esm_hidden_size // 2),
                nn.LayerNorm(esm_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(esm_hidden_size // 2, num_terms),
            )

    def mean_pooling(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over non-padding tokens.
        """
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pssm: torch.Tensor = None,
        phys: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Pool ESM embeddings
        esm_pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)

        # If external features are not provided (e.g. during simple API test without them), pad with zeros
        batch_size = esm_pooled.size(0)
        device = esm_pooled.device

        if pssm is None:
            # PSSM is sequence level usually, but we need a aggregated representation
            # For this architecture, we expect pssm to be pooled already or we pool it
            pssm = torch.zeros((batch_size, self.pssm_dim), device=device)
        else:
            if pssm.dim() == 3:  # (Batch, Seq, 20)
                pssm = self.mean_pooling(pssm, attention_mask)

        if phys is None:
            phys = torch.zeros((batch_size, self.phys_dim), device=device)
        else:
            if phys.dim() == 3:  # (Batch, Seq, 3)
                phys = self.mean_pooling(phys, attention_mask)

        # Structured Attention-based Fusion
        esm_p = self.esm_proj(esm_pooled)
        pssm_p = self.pssm_proj(pssm)
        phys_p = self.phys_proj(phys)
        
        # Stack feature projections: (Batch, 3, Hidden)
        stacked_features = torch.stack([esm_p, pssm_p, phys_p], dim=1)
        
        # Core alignment attention
        # tanh(stacked) -> (B, 3, D)  dot (D, 1) -> (B, 3, 1)
        attn_scores = torch.matmul(torch.tanh(stacked_features), self.fusion_query)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Re-weight and context-sum: (Batch, Hidden)
        fused_context = torch.sum(stacked_features * attn_weights, dim=1)

        x = self.fusion_norm(fused_context)
        x = self.activation(x)
        x = self.dropout(x)
        features = x + esm_pooled  # Residual connection to original ESM embeddings

        # Compute logits for each branch
        logits_dict = {}
        for aspect, head in self.heads.items():
            logits_dict[aspect] = head(features)

        if kwargs.get("output_attentions", False):
            return logits_dict, outputs.attentions

        return logits_dict

    def get_token_attributions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract token-level attribution (importance scores) from the final attention layer.
        Average across all attention heads for the [CLS] token (index 0) attending to the sequence.
        
        Returns:
            torch.Tensor of shape (batch, seq_len) with normalized importance scores.
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # attentions is a tuple of (layer_1, layer_2, ..., layer_n)
        # Each layer is a tensor of shape (batch, num_heads, seq_len, seq_len)
        final_layer_attn = outputs.attentions[-1]
        
        # Average across all heads
        # Shape: (batch, seq_len, seq_len)
        avg_heads = final_layer_attn.mean(dim=1)
        
        # Extract the attention of the [CLS] token (index 0) to all other tokens
        # Shape: (batch, seq_len)
        cls_attn = avg_heads[:, 0, :]
        
        # Zero out padding tokens
        cls_attn = cls_attn * attention_mask
        
        # Normalize to sum to 1.0 (excluding CLS and EOS tokens, but for simplicity we normalize overall)
        cls_attn = cls_attn / torch.clamp(cls_attn.sum(dim=-1, keepdim=True), min=1e-9)
        
        return cls_attn

    @torch.no_grad()
    def predict_with_mc_dropout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pssm: torch.Tensor = None,
        phys: torch.Tensor = None,
        n_passes: int = 10,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Monte Carlo Dropout inference (Gal & Ghahramani, 2016).

        Enables dropout during inference and runs N stochastic forward passes
        to estimate predictive uncertainty via empirical variance.

        Returns:
            Dict keyed by aspect, each containing:
              - 'mean_logits':  (B, C) mean logits across passes
              - 'variance':     (B, C) variance of logits across passes
              - 'mean_probs':   (B, C) mean sigmoid probabilities
              - 'prob_variance':(B, C) variance of probabilities
        """
        # Enable dropout layers only (keep batchnorm in eval)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

        all_logits = {asp: [] for asp in self.aspects}

        for _ in range(n_passes):
            logits_dict = self.forward(input_ids, attention_mask, pssm, phys)
            for asp, logits in logits_dict.items():
                all_logits[asp].append(logits.unsqueeze(0))

        # Restore eval mode
        self.eval()

        results = {}
        for asp in self.aspects:
            stacked = torch.cat(all_logits[asp], dim=0)  # (N, B, C)
            mean_logits = stacked.mean(dim=0)             # (B, C)
            var_logits = stacked.var(dim=0)               # (B, C)

            # Probability-space statistics
            prob_samples = torch.sigmoid(stacked)
            mean_probs = prob_samples.mean(dim=0)
            prob_var = prob_samples.var(dim=0)

            results[asp] = {
                "mean_logits": mean_logits,
                "variance": var_logits,
                "mean_probs": mean_probs,
                "prob_variance": prob_var,
            }

        return results
