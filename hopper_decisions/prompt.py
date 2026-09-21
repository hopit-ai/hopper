"""How a decision becomes a prompt. Imports no ML framework.

The user turn is one JSON object in the layout of SemIf's `core.direct_messages`
(github.com/TheoLeeCJ/SemIf, MIT, Copyright (c) 2026 TheoLeeCJ), re-implemented from its published
source; no SemIf code is copied. The system turn is our own instruction.
"""

from __future__ import annotations

import json

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = ("You make decisions about a document under a policy. Read only what is written in the document. "
          "Reply with the letter of the correct option and nothing else.")


def option_lines(example):
    """(label, text shown) per option, in the example's order."""
    if example["kind"] == "noul":
        return [("true", "true"), ("false", "false")]
    return [(o["name"], o["name"] if o["name"] == o["description"] else f"{o['name']}: {o['description']}")
            for o in example["options"]]


def body(example, question, shown):
    """One JSON object, the letters inside it; the policy rides along in the criterion."""
    criterion = f"{example['policy']}\n\n{question}" if example["policy"] else question
    return json.dumps({"evidence": example["document"] or question, "criterion": criterion,
                       "options": [{"letter": LETTERS[i], "description": text}
                                   for i, (_, text) in enumerate(shown)]}, ensure_ascii=False)


def messages(example, question, shown):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body(example, question, shown)}]


def chat_ids(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids, list) else ids["input_ids"])  # some tokenizer versions return a mapping


def letter_token_ids(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError("Every option letter must be a single token for this tokenizer.")
    return [i[0] for i in ids]
