# -*- coding: utf-8 -*-
"""Inference-time implementation of SC-TDFNet.

SC-TDFNet predicts continuous oxygen uptake from subject-level covariates and
non-gas cardiopulmonary exercise testing variables. This public module contains
the architecture used by the released reference checkpoint. Training losses,
optimisation utilities, and experiment-management code are intentionally kept
outside the inference module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class GatedResidualNetwork(nn.Module):
    """GRN block used by the static encoder and temporal fusion feed-forward path."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int | None = None,
        context_size: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = nn.Linear(context_size, hidden_size, bias=False) if context_size is not None else None
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(output_size, output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.fc1(x)
        if context is not None and self.context is not None:
            context_hidden = self.context(context)
            while context_hidden.dim() < hidden.dim():
                context_hidden = context_hidden.unsqueeze(1)
            hidden = hidden + context_hidden
        hidden = self.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        gated = torch.sigmoid(self.gate(hidden)) * hidden
        return self.norm(residual + gated)


class CausalConvBlock(nn.Module):
    """Length-preserving causal Conv1d block for sequence tensors shaped [B, T, D]."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0.0)
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.conv(self.pad(y))
        y = self.norm(y)
        y = self.dropout(self.activation(y))
        return y.transpose(1, 2)


class StaticContextEncoder(nn.Module):
    """Embedding + GRN static context encoder producing subject context c."""

    def __init__(
        self,
        feature_to_idx: Mapping[str, int],
        static_categorical: Sequence[str],
        static_continuous: Sequence[str],
        category_counts: Mapping[str, int],
        embedding_dim: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feature_to_idx = dict(feature_to_idx)
        self.static_categorical = list(static_categorical)
        self.static_continuous = list(static_continuous)
        self.embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(max(int(category_counts.get(name, 1)), 1), embedding_dim)
                for name in self.static_categorical
            }
        )
        input_size = len(self.static_continuous) + len(self.static_categorical) * embedding_dim
        self.encoder = GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_step = x[:, -1, :]
        parts: List[torch.Tensor] = []

        if self.static_continuous:
            indices = [self.feature_to_idx[name] for name in self.static_continuous]
            parts.append(static_step[:, indices])

        for name in self.static_categorical:
            values = static_step[:, self.feature_to_idx[name]].round().long()
            values = values.clamp(0, self.embeddings[name].num_embeddings - 1)
            parts.append(self.embeddings[name](values))

        return self.encoder(torch.cat(parts, dim=-1))


class FeatureSelectionNetwork(nn.Module):
    """Dual FSN gates alpha_trend(t,k) and alpha_detail(t,k) over dynamic inputs."""

    def __init__(
        self,
        dynamic_size: int,
        hidden_size: int,
        dropout: float,
        use_gate: bool = True,
        input_mode: str = "raw_mixed",
    ) -> None:
        super().__init__()
        self.dynamic_size = dynamic_size
        self.use_gate = bool(use_gate)
        if input_mode not in {"raw_mixed", "raw_only", "mixed_only"}:
            raise ValueError(f"Unsupported FSN input mode: {input_mode}")
        self.input_mode = input_mode
        self.value_projection = nn.Linear(2, hidden_size)
        self.feature_embedding = nn.Parameter(torch.randn(dynamic_size, hidden_size) * 0.02)
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        self.trend_gate = self._make_gate(hidden_size, dropout)
        self.detail_gate = self._make_gate(hidden_size, dropout)
        self.trend_temporal_embedding = CausalConvBlock(hidden_size, kernel_size=3, dilation=1, dropout=dropout)
        self.detail_temporal_embedding = CausalConvBlock(hidden_size, kernel_size=3, dilation=1, dropout=dropout)
        self.trend_norm = nn.LayerNorm(hidden_size)
        self.detail_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _make_gate(hidden_size: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _select(
        self,
        token: torch.Tensor,
        gates: torch.Tensor,
        temporal_embedding: CausalConvBlock,
        norm: nn.LayerNorm,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_tokens = token * gates.unsqueeze(-1)
        selected_sequence = selected_tokens.sum(dim=2) / max(self.dynamic_size**0.5, 1.0)
        selected_sequence = norm(temporal_embedding(selected_sequence) + selected_sequence)
        return selected_sequence, selected_tokens

    def forward(
        self,
        x_dynamic: torch.Tensor,
        context: torch.Tensor,
        mixed_dynamic: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mixed_dynamic is None:
            mixed_dynamic = x_dynamic
        if self.input_mode == "raw_only":
            mixed_dynamic = x_dynamic
        elif self.input_mode == "mixed_only":
            x_dynamic = mixed_dynamic
        token_input = torch.stack([x_dynamic, mixed_dynamic], dim=-1)
        token = self.value_projection(token_input)
        token = token + self.feature_embedding.view(1, 1, self.dynamic_size, -1)
        context_token = self.context_projection(context).unsqueeze(1).unsqueeze(2).expand_as(token)
        gate_input = torch.cat([token, context_token], dim=-1)
        if self.use_gate:
            trend_gates = torch.sigmoid(self.trend_gate(gate_input).squeeze(-1))
            detail_gates = torch.sigmoid(self.detail_gate(gate_input).squeeze(-1))
        else:
            trend_gates = torch.ones_like(x_dynamic)
            detail_gates = torch.ones_like(x_dynamic)

        trend_sequence, trend_tokens = self._select(
            token,
            trend_gates,
            self.trend_temporal_embedding,
            self.trend_norm,
        )
        detail_sequence, detail_tokens = self._select(
            token,
            detail_gates,
            self.detail_temporal_embedding,
            self.detail_norm,
        )
        return trend_sequence, detail_sequence, trend_tokens, detail_tokens, trend_gates, detail_gates


class ModernDetailConvBlock(nn.Module):
    """Lightweight ModernTCN-style branch for local VO2 detail fluctuations."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        branch_specs = [(3, 1), (3, 2), (5, 2)]
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConstantPad1d(((kernel - 1) * dilation, 0), 0.0),
                    nn.Conv1d(hidden_size, hidden_size, kernel, dilation=dilation, groups=hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.GELU(),
                )
                for kernel, dilation in branch_specs
            ]
        )
        self.pointwise = nn.Conv1d(hidden_size * len(branch_specs), hidden_size * 2, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        mixed = torch.cat([branch(y) for branch in self.branches], dim=1)
        mixed = F.glu(self.pointwise(mixed), dim=1)
        mixed = self.dropout(mixed).transpose(1, 2)
        return self.grn(self.norm(x + mixed))


class DualPathLayeredConvolutionMechanism(nn.Module):
    """D-LCM: trend path for smooth context and detail path for local residuals."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        trend_branch_specs = [
            ((5, 1), (5, 1)),
            ((7, 1), (7, 2)),
            ((9, 1), (7, 3)),
        ]
        self.trend_branches = nn.ModuleList(
            [
                nn.Sequential(
                    CausalConvBlock(hidden_size, kernel_size=first_kernel, dilation=first_dilation, dropout=dropout),
                    CausalConvBlock(hidden_size, kernel_size=second_kernel, dilation=second_dilation, dropout=dropout),
                )
                for (first_kernel, first_dilation), (second_kernel, second_dilation) in trend_branch_specs
            ]
        )
        self.trend_pointwise = nn.Linear(hidden_size * len(trend_branch_specs), hidden_size)
        self.trend_norm = nn.LayerNorm(hidden_size)
        self.trend_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.detail_path = ModernDetailConvBlock(hidden_size, dropout)

    def forward(self, trend_input: torch.Tensor, detail_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend_merged = torch.cat([branch(trend_input) for branch in self.trend_branches], dim=-1)
        trend_feat = self.trend_norm(trend_input + self.trend_pointwise(trend_merged))
        trend_feat = self.trend_grn(trend_feat)
        detail_feat = self.detail_path(detail_input)
        return trend_feat, detail_feat


class SpatialAttentionModule(nn.Module):
    """SAM feature attention over selected dynamic feature tokens."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.sequence_projection = nn.Linear(hidden_size, hidden_size)
        self.token_projection = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, sequence: torch.Tensor, selected_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequence_projection(sequence).unsqueeze(2)
        token = self.token_projection(selected_tokens)
        scores = self.score(torch.tanh(seq + token)).squeeze(-1)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.sum(selected_tokens * attention.unsqueeze(-1), dim=2)
        return self.norm(sequence + self.dropout(self.output(attended))), attention


class TemporalFusionBlock(nn.Module):
    """Residual + LayerNorm + multi-head attention temporal fusion block."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        heads = num_attention_heads if hidden_size % num_attention_heads == 0 else 1
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.use_causal_mask = bool(use_causal_mask)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_mask = None
        if self.use_causal_mask:
            length = x.shape[1]
            attn_mask = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        attended, attention = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        fused = self.attention_norm(x + attended)
        return self.feed_forward(fused), attention


class InvertedFeatureMixerBlock(nn.Module):
    """iTransformer-style variable-token mixer before feature selection."""

    def __init__(self, dynamic_size: int, hidden_size: int, num_attention_heads: int, dropout: float) -> None:
        super().__init__()
        heads = num_attention_heads if hidden_size % num_attention_heads == 0 else 1
        self.dynamic_size = int(dynamic_size)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.feature_embedding = nn.Parameter(torch.randn(dynamic_size, hidden_size) * 0.02)
        self.feature_attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.context_projection = nn.Linear(hidden_size, dynamic_size)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(dynamic_size)

    def forward(self, dynamic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, dynamic_size = dynamic.shape
        encoded = dynamic.transpose(1, 2).reshape(batch_size * dynamic_size, 1, length)
        tokens = self.temporal_encoder(encoded).mean(dim=-1).reshape(batch_size, dynamic_size, -1)
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        attended, attention = self.feature_attention(tokens, tokens, tokens, need_weights=True, average_attn_weights=False)
        tokens = self.token_norm(tokens + self.dropout(attended))
        mixed_context = torch.einsum("btc,bch->bth", dynamic, tokens) / max(float(dynamic_size) ** 0.5, 1.0)
        residual = self.context_projection(mixed_context)
        mixed_dynamic = dynamic + torch.tanh(self.residual_scale) * self.dropout(residual)
        return self.output_norm(mixed_dynamic), attention


FEATURE_COLUMNS = [
    "sex",
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "label",
    "power",
    "hr",
    "hrr",
    "p_syst",
    "p_diast",
    "spo2",
    "bf",
    "bf_rate",
    "hr_rate",
    "power_rate",
    "hrr_rate",
]

STATIC_FEATURES = ["sex", "age", "height_cm", "weight_kg", "bmi", "label"]

DYNAMIC_FEATURES = [
    "power",
    "hr",
    "hrr",
    "p_syst",
    "p_diast",
    "spo2",
    "bf",
    "bf_rate",
    "hr_rate",
    "power_rate",
    "hrr_rate",
]

STATIC_CATEGORICAL = ["sex", "label"]
CATEGORY_COUNTS = {"sex": 3, "label": 6}
WINDOW_SIZE = 32


class SCTDFNet(nn.Module):
    """Subject-Conditioned Trend-Detail Fusion Network for continuous VO2."""

    def __init__(
        self,
        feature_columns: Sequence[str] = FEATURE_COLUMNS,
        static_features: Sequence[str] = STATIC_FEATURES,
        dynamic_features: Sequence[str] = DYNAMIC_FEATURES,
        static_categorical: Sequence[str] = STATIC_CATEGORICAL,
        category_counts: Mapping[str, int] = CATEGORY_COUNTS,
        hidden_size: int = 32,
        embedding_dim: int = 8,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        detail_gate_min: float = 0.2,
        use_static_context: bool = True,
        use_fsn: bool = True,
        use_sam: bool = True,
        use_temporal_fusion: bool = True,
        use_lcm: bool = True,
        use_itransformer_detail: bool = True,
        use_dual_head: bool = True,
        fsn_input_mode: str = "raw_mixed",
        detail_fusion_mode: str = "gated_residual",
        use_detail_gate: bool = True,
        use_causal_mask: bool = True,
        detail_gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.static_features = list(static_features)
        self.base_dynamic_features = list(dynamic_features)
        self.use_static_context = bool(use_static_context)
        self.use_fsn = bool(use_fsn)
        self.use_sam = bool(use_sam)
        self.use_temporal_fusion = bool(use_temporal_fusion)
        self.use_lcm = bool(use_lcm)
        self.use_itransformer_detail = bool(use_itransformer_detail)
        self.use_dual_head = bool(use_dual_head)
        self.use_causal_mask = bool(use_causal_mask)
        if detail_fusion_mode not in {"gated_residual", "concat_mlp", "add_no_gate", "fixed_gamma", "fused_only"}:
            raise ValueError(f"Unsupported detail fusion mode: {detail_fusion_mode}")
        self.detail_fusion_mode = detail_fusion_mode
        self.use_detail_gate = bool(use_detail_gate)
        self.detail_gate_min = min(max(float(detail_gate_min), 0.0), 0.95)
        self.dynamic_features = list(self.base_dynamic_features)
        self.feature_to_idx = {name: idx for idx, name in enumerate(self.feature_columns)}
        self.dynamic_indices = [self.feature_to_idx[name] for name in self.base_dynamic_features]
        static_continuous = [name for name in self.static_features if name not in set(static_categorical)]

        self.static_encoder = (
            StaticContextEncoder(
                feature_to_idx=self.feature_to_idx,
                static_categorical=static_categorical,
                static_continuous=static_continuous,
                category_counts=category_counts,
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                dropout=dropout,
            )
            if self.use_static_context
            else None
        )
        self.feature_selector = FeatureSelectionNetwork(
            len(self.dynamic_features),
            hidden_size,
            dropout,
            use_gate=self.use_fsn,
            input_mode=fsn_input_mode,
        )
        self.lcm = DualPathLayeredConvolutionMechanism(hidden_size, dropout)
        self.sam = SpatialAttentionModule(hidden_size, dropout)
        self.temporal_fusion = TemporalFusionBlock(
            hidden_size,
            num_attention_heads,
            dropout,
            use_causal_mask=self.use_causal_mask,
        )
        self.inverted_feature_mixer = (
            InvertedFeatureMixerBlock(len(self.dynamic_features), hidden_size, num_attention_heads, dropout)
            if self.use_itransformer_detail
            else None
        )
        detail_fusion_input_size = hidden_size * 2
        self.detail_gate_signal_names = [
            "power",
            "hr",
            "bf",
            "hrr",
            "power_rate",
            "hr_rate",
            "bf_rate",
            "hrr_rate",
        ]
        self.detail_gate_signal_indices = [
            self.base_dynamic_features.index(name)
            for name in self.detail_gate_signal_names
            if name in self.base_dynamic_features
        ]
        detail_gate_signal_size = len(self.detail_gate_signal_indices)
        if self.use_dual_head:
            self.trend_head = nn.Sequential(
                CausalConvBlock(hidden_size, kernel_size=15, dilation=1, dropout=dropout),
                nn.Linear(hidden_size, 1),
            )
            self.detail_fusion_gate = None
            self.detail_fusion_mlp = None
            if self.detail_fusion_mode == "gated_residual":
                self.detail_fusion_gate = nn.Sequential(
                    nn.Linear(detail_fusion_input_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                )
            elif self.detail_fusion_mode == "concat_mlp":
                self.detail_fusion_mlp = nn.Sequential(
                    nn.Linear(detail_fusion_input_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_size),
                )
            self.detail_head = nn.Sequential(
                CausalConvBlock(hidden_size, kernel_size=3, dilation=1, dropout=dropout),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            self.detail_gate_signal_encoder = nn.Sequential(
                nn.Linear(detail_gate_signal_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_size),
            )
            self.detail_reliability_gate = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            nn.init.constant_(self.detail_reliability_gate[-1].bias, float(detail_gate_bias_init))
            self.prediction_head = None
        else:
            self.trend_head = None
            self.detail_fusion_gate = None
            self.detail_fusion_mlp = None
            self.detail_head = None
            self.detail_gate_signal_encoder = None
            self.detail_reliability_gate = None
            self.prediction_head = nn.Sequential(
                CausalConvBlock(hidden_size, kernel_size=3, dilation=1, dropout=dropout),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )

        self.last_outputs: Dict[str, torch.Tensor] = {}

    def _detail_gate_signals(
        self,
        dynamic_raw: torch.Tensor,
    ) -> torch.Tensor:
        if self.detail_gate_signal_indices:
            return dynamic_raw[:, :, self.detail_gate_signal_indices]
        return dynamic_raw.new_zeros(dynamic_raw.shape[0], dynamic_raw.shape[1], 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("Expected an input tensor with shape [batch, time, features].")
        if x.shape[1] != WINDOW_SIZE or x.shape[2] != len(self.feature_columns):
            raise ValueError(
                f"Expected [batch, {WINDOW_SIZE}, {len(self.feature_columns)}], "
                f"received {tuple(x.shape)}."
            )
        if self.static_encoder is not None:
            context = self.static_encoder(x)
        else:
            context = x.new_zeros((x.shape[0], self.feature_selector.context_projection.in_features))
        dynamic_raw = x[:, :, self.dynamic_indices]
        mixed_dynamic = dynamic_raw
        inverted_detail_attention = None
        if self.inverted_feature_mixer is not None:
            mixed_dynamic, inverted_detail_attention = self.inverted_feature_mixer(dynamic_raw)
        (
            trend_sequence,
            detail_sequence,
            trend_tokens,
            detail_tokens,
            trend_gates,
            detail_gates,
        ) = self.feature_selector(dynamic_raw, context, mixed_dynamic)

        if self.use_lcm:
            trend_feat, detail_feat = self.lcm(trend_sequence, detail_sequence)
        else:
            trend_feat, detail_feat = trend_sequence, detail_sequence

        if self.use_sam:
            sam_out, feature_attention = self.sam(trend_feat, trend_tokens)
        else:
            feature_attention = x.new_full(
                (x.shape[0], x.shape[1], len(self.dynamic_features)),
                1.0 / max(len(self.dynamic_features), 1),
            )
            sam_out = trend_feat
        if self.use_temporal_fusion:
            fused, temporal_attention = self.temporal_fusion(sam_out)
        else:
            fused = sam_out
            temporal_attention = torch.full(
                (sam_out.size(0), sam_out.size(1)),
                1.0 / max(sam_out.size(1), 1),
                device=sam_out.device,
                dtype=sam_out.dtype,
            )
        trend_seq = None
        detail_seq = None
        raw_detail_seq = None
        detail_gate = None
        detail_gate_signals = None
        detail_fusion_gamma = None
        if self.use_dual_head:
            trend_seq = self.trend_head(fused).squeeze(-1)
            detail_fusion_input = torch.cat([detail_feat, fused], dim=-1)
            if self.detail_fusion_mode == "gated_residual":
                detail_fusion_gamma = torch.sigmoid(self.detail_fusion_gate(detail_fusion_input))
                detail_context = fused + detail_fusion_gamma * detail_feat
            elif self.detail_fusion_mode == "concat_mlp":
                detail_context = self.detail_fusion_mlp(detail_fusion_input)
            elif self.detail_fusion_mode == "add_no_gate":
                detail_context = fused + detail_feat
            elif self.detail_fusion_mode == "fixed_gamma":
                detail_fusion_gamma = torch.full_like(detail_feat, 0.5)
                detail_context = fused + 0.5 * detail_feat
            elif self.detail_fusion_mode == "fused_only":
                detail_context = fused
            else:
                raise RuntimeError(f"Unsupported detail fusion mode: {self.detail_fusion_mode}")
            raw_detail_seq = self.detail_head(detail_context).squeeze(-1)
            if self.use_detail_gate:
                context_seq = context.unsqueeze(1).expand(-1, x.shape[1], -1)
                detail_gate_signals = self._detail_gate_signals(dynamic_raw)
                gate_signal_context = self.detail_gate_signal_encoder(detail_gate_signals)
                gate_raw = torch.sigmoid(
                    self.detail_reliability_gate(
                        torch.cat([fused, detail_context, context_seq, gate_signal_context], dim=-1)
                    )
                ).squeeze(-1)
                detail_gate = self.detail_gate_min + (1.0 - self.detail_gate_min) * gate_raw
                detail_seq = detail_gate * raw_detail_seq
            else:
                detail_seq = raw_detail_seq
            pred_seq = trend_seq + detail_seq
        else:
            pred_seq = self.prediction_head(fused).squeeze(-1)

        avg_gates = 0.5 * (trend_gates + detail_gates)
        outputs: Dict[str, torch.Tensor] = {
            "pred": pred_seq[:, -1],
            "context": context,
            "feature_gates": avg_gates,
            "feature_gates_trend": trend_gates,
            "feature_gates_detail": detail_gates,
            "trend_feature_gate": trend_gates,
            "detail_feature_gate": detail_gates,
            "feature_attention": feature_attention,
            "temporal_attention": temporal_attention,
            "pred_seq": pred_seq,
        }
        if inverted_detail_attention is not None:
            outputs["inverted_detail_attention"] = inverted_detail_attention
        if trend_seq is not None and detail_seq is not None:
            outputs["trend_seq"] = trend_seq
            outputs["detail_seq"] = detail_seq
        if raw_detail_seq is not None:
            outputs["raw_detail_seq"] = raw_detail_seq
            outputs["detail_raw_seq"] = raw_detail_seq
        if detail_gate is not None:
            outputs["detail_gate"] = detail_gate
        if detail_fusion_gamma is not None:
            outputs["detail_fusion_gamma"] = detail_fusion_gamma
        if detail_gate_signals is not None:
            outputs["detail_gate_signals"] = detail_gate_signals
        self.last_outputs = outputs
        return outputs

    @torch.no_grad()
    def get_interpretability(self) -> Dict[str, object]:
        gates = self.last_outputs.get("feature_gates")
        trend_gates = self.last_outputs.get("feature_gates_trend")
        detail_gates = self.last_outputs.get("feature_gates_detail")
        feature_attention = self.last_outputs.get("feature_attention")
        pred_seq = self.last_outputs.get("pred_seq")
        trend_seq = self.last_outputs.get("trend_seq")
        detail_seq = self.last_outputs.get("detail_seq")
        raw_detail_seq = self.last_outputs.get("raw_detail_seq")
        detail_gate = self.last_outputs.get("detail_gate")

        result: Dict[str, object] = {"dynamic_features": self.dynamic_features}
        if gates is not None:
            result["feature_gate_mean"] = gates.detach().mean(dim=(0, 1)).cpu().tolist()
            result["temporal_gate_mean"] = gates.detach().mean(dim=2).cpu()
        if trend_gates is not None:
            result["feature_gate_trend_mean"] = trend_gates.detach().mean(dim=(0, 1)).cpu().tolist()
            result["temporal_trend_gate_mean"] = trend_gates.detach().mean(dim=2).cpu()
        if detail_gates is not None:
            result["feature_gate_detail_mean"] = detail_gates.detach().mean(dim=(0, 1)).cpu().tolist()
            result["temporal_detail_gate_mean"] = detail_gates.detach().mean(dim=2).cpu()
        if feature_attention is not None:
            result["feature_attention_mean"] = feature_attention.detach().mean(dim=(0, 1)).cpu().tolist()
            result["temporal_feature_attention"] = feature_attention.detach().cpu()
        if pred_seq is not None:
            result["pred_seq"] = pred_seq.detach().cpu()
        if trend_seq is not None:
            result["trend_seq"] = trend_seq.detach().cpu()
        if detail_seq is not None:
            result["detail_seq"] = detail_seq.detach().cpu()
        if raw_detail_seq is not None:
            result["raw_detail_seq"] = raw_detail_seq.detach().cpu()
        if detail_gate is not None:
            result["detail_gate"] = detail_gate.detach().cpu()
        return result


def build_model() -> SCTDFNet:
    """Construct the manuscript SC-TDFNet architecture."""
    return SCTDFNet()


def load_pretrained(
    checkpoint: str | Path,
    map_location: str | torch.device = "cpu",
) -> SCTDFNet:
    """Load the released state dictionary into the manuscript architecture."""
    model = build_model()
    state_dict = torch.load(Path(checkpoint), map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model








