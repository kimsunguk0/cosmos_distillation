from src.model.tokenizer_ext import distill_trainable_token_ids, ensure_special_tokens, missing_special_tokens


def test_missing_special_tokens_detects_required_items() -> None:
    missing = missing_special_tokens({"<|cot_start|>"})
    assert "<|cot_end|>" in missing
    assert "<i0>" in missing


class _DummyTokenizer:
    def __init__(self) -> None:
        self._vocab = {"hello": 0}
        self.pad_token = None
        self.eos_token = "</s>"

    def get_vocab(self):
        return self._vocab

    def add_tokens(self, tokens):
        for token in tokens:
            self._vocab[token] = len(self._vocab)

    def add_special_tokens(self, payload):
        for token in payload.get("additional_special_tokens", []):
            self._vocab[token] = len(self._vocab)
        if "pad_token" in payload:
            self.pad_token = payload["pad_token"]
            self._vocab[self.pad_token] = len(self._vocab)

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]


def test_ensure_special_tokens_adds_missing_items() -> None:
    tokenizer = _DummyTokenizer()
    added = ensure_special_tokens(tokenizer)
    assert "<|cot_start|>" in added
    assert "<|meta_action_start|>" in added
    assert "<i0>" in added
    assert tokenizer.pad_token == "</s>"


def test_distill_trainable_token_ids_cover_control_and_traj_vocab() -> None:
    tokenizer = _DummyTokenizer()
    ensure_special_tokens(tokenizer)
    token_ids = distill_trainable_token_ids(tokenizer)
    assert tokenizer.convert_tokens_to_ids("<|cot_start|>") in token_ids
    assert tokenizer.convert_tokens_to_ids("<|cot_end|>") in token_ids
    assert tokenizer.convert_tokens_to_ids("<i0>") in token_ids
    assert tokenizer.convert_tokens_to_ids("<i3999>") in token_ids
