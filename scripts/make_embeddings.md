# Regenerating the input embeddings

The dense representations used as model input are not shipped with the repository
because of their size (several hundred GB). They can be regenerated from data/text/.

Encoder and settings used in the paper:

| Item | Value |
|---|---|
| backbone | Qwen2-7B-Instruct (last hidden state) |
| max tokens | 256 |
| pooling | none (full token sequence kept) |
| output shape | [N, 256, 768], float32 |

Steps:

1. Load `data/text/**/文本icmt_*.npy` with `numpy.load(..., allow_pickle=True)`.
2. Tokenise each sentence to at most 256 tokens and run the encoder.
3. Save the last hidden state per sentence as float32 with the same relative path under
   `data/embeddings/`.
4. Point the `DATA_ROOT` variable in `src/hcemor/*.py` to the new directory.

Additional embeddings consumed by the LLM-UFDE module (related-sentence embeddings) follow
the same recipe but are computed over a related-sentence corpus rather than the comment text.
