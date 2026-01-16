# LOSS Specification

This document defines the behavior and processing model of LOSS.  
It exists to allow reimplementation of LOSS in other languages such as C, Python, Rust, Ruby, or Java.

The specification describes observable behavior, not a required internal structure.

---

## Overview

LOSS is a text sanitization pipeline.

Input:
- Arbitrary UTF-8 text from standard input

Output:
- Sanitized UTF-8 text written to standard output

LOSS processes text in ordered stages.  
Each stage takes text as input and produces modified text as output.

Stages must be applied sequentially.

---

## Design Goals

- Preserve meaning
- Remove common large language model stylistic artifacts
- Avoid full rewriting
- Introduce small, human-like irregularities
- Allow deterministic output via a seed
- Operate without requiring a model by default

LOSS is not intended to guarantee detection avoidance.

---

## Processing Pipeline

The reference pipeline consists of the following stages, in order:

1. Structural normalization  
2. Punctuation normalization  
3. Emoji removal  
4. Sentence rhythm de-optimization  
5. Vocabulary flattening  
6. Partial sentence rephrasing  
7. LLM phrase suppression  
8. Final cleanup and normalization  

Implementations may add optional stages, but these stages define baseline behavior.

---

## Stage 1: Structural Normalization

Purpose:
- Remove formatting artifacts that are common in AI generated text

Required behavior:
- Remove fenced code blocks
- Remove inline code markers
- Remove markdown headers
- Remove bold and italic formatting
- Remove bullet markers and numbering

Result:
- Plain text with paragraph structure preserved

---

## Stage 2: Punctuation Normalization

Purpose:
- Reduce overly polished punctuation patterns

Required behavior:
- Convert em dashes and en dashes to hyphens or periods
- Normalize smart quotes to ASCII quotes
- Randomly reduce excessive use of semicolons and colons
- Preserve sentence-ending punctuation

Behavior may be randomized but must be seedable.

---

## Stage 3: Emoji Removal

Purpose:
- Remove visual markers strongly associated with casual AI output

Required behavior:
- Remove Unicode emoji characters
- Preserve surrounding text

Emoji detection may be implemented using Unicode ranges or libraries.

---

## Stage 4: LLM Phrase Suppression

Purpose:
- Remove common boilerplate phrases frequently used by language models

Required behavior:
- Remove phrases such as:
  - "In conclusion"
  - "It is important to note that"
  - "Overall"
  - "That being said"
  - "The key takeaway is"
- Matching should be case insensitive
- Removal should occur at sentence starts
- Removal must be cumulative

This stage must operate on sentence-level units.

---

## Stage 5: Sentence Rhythm De-optimization

Purpose:
- Break overly uniform sentence structure

Required behavior:
- Occasionally merge adjacent sentences
- Occasionally split very long sentences
- Preserve meaning
- Preserve readability

Randomness must be controlled by a seed.

---

## Stage 6: Vocabulary Flattening

Purpose:
- Replace overly formal or corporate vocabulary with simpler alternatives

Required behavior:
- Replace words such as:
  - "utilize" -> "use"
  - "facilitate" -> "help"
  - "demonstrate" -> "show"
- Replacement must respect word boundaries
- Replacement must not change meaning

This stage may optionally call a local model for sentence-level simplification.

---

## Stage 7: Partial Sentence Rephrasing

Purpose:
- Introduce small variation without rewriting the entire text

Required behavior:
- Select a subset of sentences based on a ratio
- Selection may be weighted by sentence length or position
- Rephrasing must preserve meaning
- Rephrasing must be minimal

If no model is used, simple rule-based substitutions are acceptable.

---

## Stage 8: Final Cleanup and Normalization

Purpose:
- Fix artifacts introduced by earlier stages

Required behavior:
- Normalize whitespace
- Fix spacing around punctuation
- Capitalize sentence starts
- Capitalize the first character of the text
- Remove empty sentences

This stage must not introduce new content.

---

## Randomness and Determinism

LOSS uses randomness for:
- Sentence merging and splitting
- Punctuation variation
- Sentence selection for rephrasing

If a seed is provided:
- Output must be deterministic

If no seed is provided:
- Output may vary between runs

---

## Optional Local Model Integration

Implementations may support calling a local OpenAI compatible API.

Model usage is optional and must be disabled by default.

Model calls may be used for:
- Vocabulary flattening
- Sentence rephrasing

When enabled, all data must remain local to the user environment.

---

## Input and Output Requirements

- Input must be treated as UTF-8 text
- Output must be UTF-8 text
- Line endings may be normalized
- No metadata should be emitted

---

## Error Handling

- If a model call fails, fallback behavior must be used
- The program should never crash on malformed text
- Invalid configuration values should result in safe defaults

---

## Non-Goals

LOSS does not aim to:
- Guarantee bypassing AI detection
- Perfect grammar
- Full paraphrasing
- Semantic transformation

Slight imperfections are acceptable and intentional.

---

## Conformance

An implementation is considered LOSS-compatible if:

- It follows the pipeline order
- It implements all required stages
- It preserves meaning
- It produces human-readable output
- It supports deterministic behavior via a seed

Exact output matching is not required.

---

## Versioning

This specification describes LOSS version 1 behavior.

Future versions may add optional stages but should not remove or reorder core stages without a major version change.
