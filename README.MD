# LOSS

LOSS is a command line text sanitization tool written in Go.  
It is designed to clean and normalize text that comes from large language models so that it reads more like ordinary human writing.

The goal is not to hide AI usage, but to remove common stylistic artifacts such as boilerplate phrases, excessive polish, unnatural rhythm, emojis, and markdown formatting.

---

## Features

- Removes common LLM boilerplate phrases  
- Strips markdown formatting  
- Removes emojis  
- Normalizes punctuation and spacing  
- Flattens overly formal vocabulary  
- Lightly rephrases a subset of sentences  
- Randomized behavior with optional deterministic seed  
- Optional integration with a local OpenAI compatible model  

---

## How It Works

LOSS processes text in multiple stages:

1. Structural normalization  
2. Punctuation normalization  
3. Emoji removal  
4. Sentence rhythm randomization  
5. Vocabulary flattening  
6. Partial sentence rephrasing  
7. LLM phrase suppression  
8. Final cleanup and capitalization  

Each stage is intentionally simple and composable.

---

## Installation

You need Go 1.20 or newer.

Clone the repository and run:

```

go build -o loss main.go

```

---

## Usage

LOSS reads from standard input and writes to standard output.

Example:

```

type input.txt | loss

```

Or without building:

```

type input.txt | go run main.go

```

---

## Common Flags

```

--strip-markdown          Remove markdown formatting
--strip-emojis            Remove emojis
--normalize-punctuation   Normalize punctuation and spacing
--remove-llm-phrases      Remove common LLM phrases
--flatten-vocab           none | low | medium | high
--rephrase-ratio          Fraction of sentences to lightly rephrase
--seed                    Random seed for reproducible output

```

Example with options:

```

type input.txt | loss --flatten-vocab=high --rephrase-ratio=0.3 --seed=42

```

---

## Local Model Support

LOSS can optionally call a local OpenAI compatible API for sentence rephrasing and vocabulary flattening.

Supported via flags:

```

--use-local-model
--api-endpoint
--api-key
--model-name
--temperature

```

By default, LOSS does not require any model.

---

## Design Philosophy

- Simple transformations over heavy rewriting  
- Small randomness instead of full regeneration  
- Deterministic when needed  
- Readable output over aggressive sanitization  

If the output looks slightly imperfect, that is intentional.

---

## Disclaimer

LOSS does not guarantee that text will bypass detection systems.  
It is a text cleanup and normalization utility, not an obfuscation tool.

---

## License

MIT

