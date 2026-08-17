import os
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_CLIP_MODEL_PATH = os.environ.get(
    'CLIP_MODEL_PATH',
    '/path/to/clip-vit-base-patch32',
)


def require_local_clip_safetensors(model_path: str | os.PathLike[str]) -> Path:
    """Return a local CLIP Safetensors file or fail with conversion guidance."""
    path = Path(model_path).expanduser().resolve()
    weights = path / 'model.safetensors'
    if weights.is_file():
        return weights
    index = path / 'model.safetensors.index.json'
    if index.is_file():
        return index
    raise FileNotFoundError(
        f'Local CLIP model at {path} has no Safetensors weights. '
        'PyTorch 2.4 with Transformers 4.57 cannot load pytorch_model.bin. Run '
        '`python tools/convert_clip_to_safetensors.py --source-model-path=... '
        '--output-model-path=... --trust-local-pickle` and use the output path.'
    )


class CLIPTextFeatureExtractor(nn.Module):
    def __init__(self, model_path: str | None = None):
        super().__init__()
        try:
            from transformers import CLIPModel
        except ImportError as exc:
            raise ImportError(
                'Language conditioning requires a compatible transformers installation. '
                'The local CLIP weights are read from `language_model_path`, but the Python '
                'environment must still provide transformers and a compatible huggingface-hub.'
            ) from exc

        model_path = DEFAULT_CLIP_MODEL_PATH if model_path is None else model_path
        require_local_clip_safetensors(model_path)
        self.language_model = CLIPModel.from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
        )
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False
        self.language_model.eval()

    def encode_raw_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self.language_model.eval()
        with torch.no_grad():
            return self.language_model.get_text_features(
                input_ids=input_ids.long(),
                attention_mask=attention_mask.long(),
            )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode_raw_features(input_ids, attention_mask)


class CLIPTextTokenizer:
    def __init__(self, model_path: str | None = None, max_length: int = 77):
        try:
            from transformers import CLIPTokenizer
        except ImportError as exc:
            raise ImportError(
                'Language conditioning requires a compatible transformers installation. '
                'The local CLIP tokenizer files are read from `language_model_path`, but the '
                'Python environment must still provide transformers and a compatible huggingface-hub.'
            ) from exc

        model_path = DEFAULT_CLIP_MODEL_PATH if model_path is None else model_path
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.max_length = int(max_length)

    def tokenize(self, texts, device=None):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        return input_ids, attention_mask


class SmolVLMTextTokenizer:
    def __init__(self, model_path: str, max_length: int = 50):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                'SmolVLM context conditioning requires transformers and a compatible huggingface-hub.'
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = int(max_length)

    def tokenize(self, texts, device=None):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        return input_ids, attention_mask


def build_text_tokenizer(
    tokenizer_type: str,
    *,
    model_path: str,
    max_length: int,
):
    tokenizer_type = str(tokenizer_type).strip().lower()
    if tokenizer_type == 'clip':
        return CLIPTextTokenizer(model_path=model_path, max_length=max_length)
    if tokenizer_type in ('smolvlm', 'smol_vlm'):
        return SmolVLMTextTokenizer(model_path=model_path, max_length=max_length)
    raise ValueError(f'Unsupported language tokenizer type: {tokenizer_type!r}.')
