"""The embedding first stage for long menus (`--shortlist embedding`). Off by default: not measured yet.

Model: `Qwen/Qwen3-Embedding-0.6B` at a pinned revision, Apache-2.0, loaded with plain transformers
(no extra dependency), bf16, on the same GPU as the decision model. It is downloaded from the
Hugging Face Hub the first time a server is started with `--shortlist embedding`, and never
otherwise. Its conventions, from its model card: last-token pooling (the tokenizer appends
`<|endoftext|>`), L2-normalised vectors, and a one-line instruction in front of a query but not in
front of a document.

Here the query is the request's document (the state), with the question in the instruction, and
each option is embedded as the exact line the decision prompt shows for it. An option's score is
the dot product of the two normalised vectors. Each text is embedded on its own, never padded into
a batch, so an option's vector depends only on its text and a cached vector is the same number the
first computation gave. Option vectors are kept in a bounded least-recently-used cache: a routing
menu is embedded once and each later request costs one embedding of its state.

torch is imported only when an `Embedder` is built, so the rest of this module imports without it.
"""

from __future__ import annotations

from collections import OrderedDict

from hopper_decisions.shortlist import EMBEDDER, EMBEDDER_REVISION

TASK = "Given a document and a question about it, retrieve the option that answers the question."
MAX_TOKENS = 8192      # the model takes 32k; a JevBench-length state is ~6k
CACHE_SIZE = 20_000    # option vectors kept, 4 KiB each in float32 (1,024 dimensions): ~80 MiB at most


def query_text(document, question):
    """The model card's query format: 'Instruct: {task}\\nQuery:{query}'."""
    task = f"{TASK} Question: {question}" if question else TASK
    return f"Instruct: {task}\nQuery:{document or question}"


class Embedder:
    def __init__(self, model=EMBEDDER, revision=EMBEDDER_REVISION, device="cuda", cache_size=CACHE_SIZE):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.name, self.revision, self.device, self.cache_size = model, revision, device, cache_size
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModel.from_pretrained(model, revision=revision, dtype=torch.bfloat16,
                                               device_map=device).eval()
        self.cache = OrderedDict()

    def vector(self, text):
        torch = self.torch
        ids = self.tokenizer(text, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            last = self.model(input_ids=ids.to(self.device)).last_hidden_state[0, -1].float()
            return torch.nn.functional.normalize(last, dim=0)

    def option_vectors(self, texts):
        out = []
        for text in texts:
            if text in self.cache:
                self.cache.move_to_end(text)
            else:
                self.cache[text] = self.vector(text)
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            out.append(self.cache[text])
        return self.torch.stack(out)

    def similarities(self, document, question, texts):
        """One score per option text: the dot product of normalised embeddings, as Python floats."""
        with self.torch.inference_mode():
            return (self.option_vectors(texts) @ self.vector(query_text(document, question))).tolist()
