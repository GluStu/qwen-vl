"""The full vision-language model: vision tower -> resampler -> causal LLM."""

import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwenvl_siglip.constants import SPECIAL_TOKENS
from qwenvl_siglip.modeling.resampler import Resampler
from qwenvl_siglip.modeling.vision import VisionTower


class QwenVLStyle(nn.Module):
    def __init__(
        self,
        llm_name: str,
        siglip_name: str,
        dinov2_name: str | None = None,
        llm_dtype=torch.bfloat16,
        vit_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name, dtype=llm_dtype)
        if len(self.tokenizer) > self.llm.get_input_embeddings().num_embeddings:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.img_pad_id = self.tokenizer.convert_tokens_to_ids("<imgpad>")
        hidden = self.llm.config.hidden_size
        self.vision = VisionTower(siglip_name, dinov2_name, vit_dtype)
        self.adapter = Resampler(hidden, self.vision.out_dim, hidden // 128, self.vision.grid)
        self.vit_trainable = False

    # ------------------------------------------------------------ construction
    @classmethod
    def from_run_dir(cls, run_dir: str, ckpt: str = "trainable.pt", device: str = "cuda"):
        """Rebuild a model from a training run directory (args.json + trainable weights)."""
        run = Path(run_dir)
        cfg = json.loads((run / "args.json").read_text())
        model = cls(cfg["llm"], cfg["siglip"], cfg.get("dinov2"))
        if cfg.get("lora") and cfg["stage"] in (2, 3):
            model.set_trainable(cfg["stage"])
            model.apply_lora(cfg["lora"])
        model.vit_trainable = False
        state = torch.load(run / ckpt, map_location="cpu")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            warnings.warn(f"checkpoint keys not found in model: {result.unexpected_keys[:5]}", stacklevel=2)
        return model.to(device).eval()

    def set_trainable(self, stage: int, train_vit: bool = False) -> None:
        """Select the trainable parts for a stage (call BEFORE `apply_lora`)."""

        def flag(module: nn.Module, value: bool) -> None:
            for p in module.parameters():
                p.requires_grad = value

        flag(self.adapter, True)
        flag(self.llm, stage in (2, 3))
        flag(self.vision, train_vit if stage == 1 else stage == 2)  # stage 3: ViT frozen
        self.vit_trainable = any(p.requires_grad for p in self.vision.parameters())

    def apply_lora(self, r: int = 64) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError("LoRA needs peft: pip install 'qwenvl-siglip[lora]'") from e
        cfg = LoraConfig(
            r=r,
            lora_alpha=2 * r,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )  # fmt: skip
        self.llm = get_peft_model(self.llm, cfg)

    # ------------------------------------------------------------------ forward
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.vit_trainable):
            feats = self.vision(pixel_values)
        return self.adapter(feats)

    def _embed(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.llm.get_input_embeddings()(input_ids)
        vis = self.encode_images(pixel_values).to(embeds.dtype)  # (B, 256, D)
        mask = (input_ids == self.img_pad_id).unsqueeze(-1)
        if int(mask.sum()) != vis.size(0) * vis.size(1):
            raise ValueError("each sample must contain exactly 256 <imgpad> tokens")
        return embeds.masked_scatter(mask, vis)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        embeds = self._embed(pixel_values, input_ids)
        return self.llm(inputs_embeds=embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask, **kwargs):
        embeds = self._embed(pixel_values, input_ids)
        return self.llm.generate(inputs_embeds=embeds, attention_mask=attention_mask, **kwargs)
