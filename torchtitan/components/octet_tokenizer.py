# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.components.tokenizer import BaseTokenizer


class OctetTokenizer(BaseTokenizer):
    """Byte-level tokenizer for ByteLlama.

    Maps each UTF-8 byte to a token ID with a fixed offset, plus special tokens
    for BOS (1), EOS (2), and PAD (0). Vocab size is 387
    (3 special + 256 byte + 128 supplemental).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    OFFSET = 3
    SUPPL_TOKEN_OFFSET = OFFSET + 256
    VOCAB_SIZE = SUPPL_TOKEN_OFFSET + 128  # 387

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def encode(self, text, add_bos=True, add_eos=True, **kwargs):
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(b + self.OFFSET for b in text.encode("utf-8"))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids, **kwargs):
        raw = [
            id - self.OFFSET
            for id in ids
            if self.OFFSET <= id < self.SUPPL_TOKEN_OFFSET
        ]
        return bytes(raw).decode("utf-8", errors="ignore")

    def get_vocab_size(self):
        return self.VOCAB_SIZE
