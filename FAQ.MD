# FAQ

## What is LOSS?

LOSS is a command line tool that cleans and normalizes text.  
It is mainly intended for text that comes from large language models and needs to read more like ordinary human writing.

---

## Is LOSS meant to hide AI usage?

No.  
LOSS removes stylistic artifacts and boilerplate, but it does not attempt to guarantee that text cannot be identified as AI assisted.

It is a cleanup tool, not an evasion system.

---

## What kind of changes does LOSS make?

LOSS may:

- Remove common LLM phrases
- Strip markdown and emojis
- Normalize punctuation and spacing
- Simplify overly formal vocabulary
- Slightly adjust sentence rhythm
- Lightly rephrase some sentences

The meaning of the text is preserved.

---

## Does LOSS rewrite the entire text?

No.  
Only a portion of sentences are rephrased, based on the configured ratio.  
Most transformations are small and local.

---

## Is the output deterministic?

Yes, if a seed is provided.

Using the `--seed` flag ensures the same input produces the same output.

Without a seed, randomness is used.

---

## Does LOSS require an AI model?

No.  
LOSS works entirely without a model by default.

A local OpenAI compatible API can be enabled optionally for better rephrasing.

---

## What models are supported?

Any OpenAI compatible chat completion API is supported.  
This includes local models exposed through tools such as llama.cpp servers or similar.

LOSS does not include a model itself.

---

## Can LOSS break grammar?

In rare cases, yes.  
The tool favors natural variation over perfect grammar.

This behavior is intentional and usually limited to minor imperfections.

---

## Is LOSS suitable for academic or professional writing?

It depends on the configuration.

With low rephrase ratios and minimal vocabulary flattening, it can be used for professional text.  
Aggressive settings are better suited for casual or informal writing.

---

## Why does the output sometimes feel uneven?

Humans write unevenly.  
LOSS intentionally avoids producing perfectly polished text.

Slight irregularities are a feature, not a bug.

---

## Is LOSS safe to use on sensitive content?

LOSS does not transmit data unless local model support is enabled.  
When using a local API, all data stays within your environment.

Always review output before publishing.

---

## Can I extend LOSS?

Yes.  
The code is structured as a pipeline of stages and is easy to modify.

Adding or removing stages is straightforward.

---

## Where should I use LOSS?

LOSS is best used for:

- Draft cleanup
- Internal notes
- Personal writing
- Normalizing mixed human and AI text

It is not intended for legal, medical, or critical documents without review.
