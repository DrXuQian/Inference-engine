# Qwen3.5-35B-A3B-GPTQ-Int4 TP=2 Split Notes

## Config Changes (6 fields)

| Field | Original | After Split |
|---|---|---|
| `num_attention_heads` | 16 | 8 |
| `num_key_value_heads` | 2 | 1 |
| `linear_num_key_heads` | 16 | 8 |
| `linear_num_value_heads` | 32 | 16 |
| `moe_intermediate_size` | 512 | 256 |
| `shared_expert_intermediate_size` | 512 | 256 |

All other fields unchanged (hidden_size, head_dim, num_experts, vocab_size,
vision_config, etc.). In the MIG simulation rank directory, `embed_tokens` and
`lm_head` stay physically replicated so rank_0 can be loaded and run as a
standalone model.

## Weight Size Breakdown

Use the generated rank directory metadata for exact physical size. The MIG
simulation layout intentionally keeps some tensors full-size so each rank can
run standalone; it is not identical to logical distributed TP storage.

Remaining replicated weights include:

| Replicated Component | Size | Reason |
|---|---|---|
| embed/lm_head | model-dependent | Kept full so a single rank can run on MIG |
| visual encoder | 0.9 GB | Vision model replicated entirely |
| GPTQ g_idx | 0.17 GB | g_idx replicated for column-parallel gate/up_proj |
| MoE router + norms + misc | 0.05 GB | Routing weights and layer norms not split |

Performance compensation still models `lm_head` as logical vocab-parallel
(`/TP`) when estimating tail time and decode bandwidth. That is separate from
the physical MIG simulation layout above.

## Splitting Strategy Per Component

### Embeddings / LM Head — bf16, physical MIG layout

- `embed_tokens.weight` [248320, 2048] → [248320, 2048]: replicated
- `lm_head.weight` [248320, 2048] → [248320, 2048]: replicated
- Logical TP timing/bandwidth accounting uses `lm_head / TP`.

### Full Attention (self_attn) — bf16, not GPTQ-quantized

- `q_proj.weight` [8192, 2048] → [4096, 2048]: column-parallel (dim 0). Note: 8192 = num_heads x head_dim x 2 due to `attn_output_gate=true`.
- `k_proj.weight` [512, 2048] → [256, 2048]: column-parallel (dim 0)
- `v_proj.weight` [512, 2048] → [256, 2048]: column-parallel (dim 0)
- `o_proj.weight` [2048, 4096] → [2048, 2048]: row-parallel (dim 1)
- `q_norm.weight` [256], `k_norm.weight` [256]: replicated (per-head-dim norm)

### Linear / Mamba2 Attention (linear_attn) — bf16

- `in_proj_qkv.weight` [8192, 2048] → [4096, 2048]: column-parallel. (Q:2048 + K:2048 + V:4096 = 8192)
- `in_proj_z.weight` [4096, 2048] → [2048, 2048]: column-parallel
- `in_proj_a.weight` [32, 2048] → [16, 2048]: column-parallel (num_value_heads)
- `in_proj_b.weight` [32, 2048] → [16, 2048]: column-parallel
- `out_proj.weight` [2048, 4096] → [2048, 2048]: row-parallel
- `conv1d.weight` [8192, 1, 4] → [4096, 1, 4]: split channels (dim 0)
- `A_log` [32] → [16]: split heads
- `dt_bias` [32] → [16]: split heads
- `norm.weight` [128]: replicated (per value_head_dim)

### MoE Experts — GPTQ int4 (majority layers, intermediate=512)

For each of the 256 experts:

- `gate_proj.qweight` [256, 512] → [256, 256]: GPTQ column-parallel (split dim 1)
- `gate_proj.scales` [16, 512] → [16, 256]: GPTQ column-parallel
- `gate_proj.qzeros` [16, 64] → [16, 32]: GPTQ column-parallel
- `gate_proj.g_idx` [2048]: replicated
- `up_proj`: same as gate_proj
- `down_proj.qweight` [64, 2048] → [32, 2048]: GPTQ row-parallel (split dim 0)
- `down_proj.scales` [4, 2048] → [2, 2048]: GPTQ row-parallel
- `down_proj.qzeros` [4, 256] → [2, 256]: GPTQ row-parallel
- `down_proj.g_idx` [512] → [256]: split + regenerated sequential (desc_act=false)

### Shared Expert — bf16 (not quantized)

- `gate_proj.weight` [512, 2048] → [256, 2048]: column-parallel
- `up_proj.weight` [512, 2048] → [256, 2048]: column-parallel
- `down_proj.weight` [2048, 512] → [2048, 256]: row-parallel

### Replicated (no split)

- All visual encoder weights
- Layer norms (`input_layernorm`, `post_attention_layernorm`, `norm`)
- MoE router (`mlp.gate.weight`)
- Shared expert gate (`shared_expert_gate.weight`)
- MTP helper layers (`mtp.fc`, `mtp.norm`, `mtp.pre_fc_norm_*`)
