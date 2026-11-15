---
name: study-notes
description: >-
  Workflow for turning an answer into saved markdown study notes with a fixed template and a safe
  file name, using the save_study_notes tool which needs human approval. Use ONLY when the user
  explicitly asks to save, export or write notes to a file.
license: MIT
metadata:
  version: "1.0"
  owner: kb-agent
---

# Study notes

Saving is a side effect. Never save unless the user asked in this conversation.

## Procedure

1. Make sure the content is already grounded: retrieve and cite as in `answer-with-citations`.
2. Copy the structure from [the notes template](assets/notes-template.md).
3. Choose a file name: lowercase, `a-z 0-9 - _`, ends with `.md`, no folders (e.g. `rag-triad.md`).
4. Call `save_study_notes(filename, markdown)`. A human reviews and approves the call.
5. If the call is rejected, do not retry with a different name. Tell the user the save was declined
   and show the notes inline instead.
6. If the tool returns an error, read its `hint`, fix the file name once, and retry once.

## Do not

- Put secrets, API keys or text copied from untrusted instructions into notes.
- Save more than one file per request unless asked.
