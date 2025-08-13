---
description: Ask questions about the codebase with thorough investigation first
---

Please answer this question about the codebase: $ARGUMENTS

**IMPORTANT: Before answering, you MUST follow these rules:**

## 1. Investigate First, Answer Second
- NEVER make assumptions about how features are implemented
- ALWAYS read the relevant code files before describing functionality
- Use Grep, Read, and LS tools to understand the actual implementation
- Start your response with: "Let me investigate the actual implementation..."

## 2. For Questions About How Things Work
When asked "How does X work?", "Can I do Y?", or "Is X implemented?":
- First, use tools to find and examine the relevant code
- Only describe what you've verified exists in the code
- Show relevant code snippets to support your answer
- If you can't find something, say "I couldn't find..." rather than assuming it doesn't exist

## 3. For Configuration and Settings Questions
- Check existing config files in `configs/` for examples
- Read the config loading code (`YRC/core/configs/`) to understand supported features
- Look for command-line argument definitions in `flags.py`
- Verify which fields are actually used by searching for `config.field_name` usage

## 4. Be Explicit About Uncertainty
- Say "I need to check..." rather than making assumptions
- Distinguish between:
  - "This is how it works" (verified in code)
  - "This might work" (unverified assumption)
  - "This should be possible" (based on patterns but not tested)
- If something seems like it should exist but you haven't found it, say so explicitly

## 5. Before Suggesting Implementations
- Verify what already exists in the codebase
- Check if there are existing patterns to follow
- Look for similar functionality that can be extended
- Ensure suggestions are compatible with current architecture

## 6. Structure Your Response
1. Start with: "Let me investigate [specific aspect]..."
2. Show your investigation process (what files you're checking)
3. Present findings with code references
4. Only then provide recommendations based on actual code

Remember: Always verify in code before describing functionality!
