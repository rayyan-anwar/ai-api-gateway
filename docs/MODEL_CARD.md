# System Card

Author: Rayyan Anwar

**System:** an API gateway (routing, fallback, auth, rate limiting, metrics) plus a small built-in `local` provider with three classical text tools: frequency-based extractive summariser, lexicon sentiment with negation handling, frequency keywords. No machine-learned model is bundled.

**Intended use:** learning, prototyping and as a reference for gateway patterns in front of self-hosted or third-party AI APIs; local smoke tests without paid APIs.

**Out of scope / prohibited:** treating the local provider as a trustworthy NLP model; high-stakes decisions (medical, legal, financial, employment); using it as a security boundary without independent review; exposing it publicly with authentication disabled.

**Data:** sample requests and sentiment sentences are synthetic and written for this repository.

**Risks and limitations**
- Sentiment lexicon is tiny and English-only; sarcasm, domain terms and unlisted words are missed.
- Responses from upstream LLMs are passed through unmodified; the gateway does **not** verify factuality or filter harmful content.
- Prompts and responses may contain personal data; the gateway does not log message bodies by default, but upstream providers may retain them under their own terms.
- Token counts are estimated (about 4 characters per token) when an upstream does not report usage and are flagged as such; they are not billing-grade.
- Rate limiting is per process and in memory; it is not a distributed quota system.
