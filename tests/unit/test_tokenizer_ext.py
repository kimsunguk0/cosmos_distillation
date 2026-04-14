from src.model.tokenizer_ext import ensure_special_tokens, missing_special_tokens


def test_missing_special_tokens_detects_required_items() -> None:
    missing = missing_special_tokens({"<|cot_start|>"})
    assert "<|cot_end|>" in missing


class _DummyTokenizer:
    def __init__(self) -> None:
        self._vocab = {"hello": 0}
        self.pad_token = None
        self.eos_token = "</s>"

    def get_vocab(self):
        return self._vocab

    def add_special_tokens(self, payload):
        for token in payload.get("additional_special_tokens", []):
            self._vocab[token] = len(self._vocab)
        if "pad_token" in payload:
            self.pad_token = payload["pad_token"]
            self._vocab[self.pad_token] = len(self._vocab)


def test_ensure_special_tokens_adds_missing_items() -> None:
    tokenizer = _DummyTokenizer()
    added = ensure_special_tokens(tokenizer)
    assert "<|cot_start|>" in added
    assert "<|meta_action_start|>" in added
    assert tokenizer.pad_token == "</s>"
