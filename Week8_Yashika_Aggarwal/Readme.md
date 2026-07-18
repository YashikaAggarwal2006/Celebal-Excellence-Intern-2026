# Single-Agent Smart Assistant

A lightweight single-agent pipeline that understands user queries, routes them based on intent, uses tools when needed, and returns structured JSON output.

## Overview

This project implements a simple but complete agent loop:

**Query → Intent Detection → Tool Routing → Structured Response**

The agent handles three types of requests:
- **Math queries** (containing `"calculate"`) → handled by a Calculator Tool
- **Keyword extraction** (containing `"keywords"`) → handled by a Keyword Extractor Tool
- **Everything else** → handled by a general fallback response

## Features

- 🧮 **Calculator Tool** — safely evaluates math expressions using a regex whitelist to block unsafe input before `eval`
- 🔑 **Keyword Extractor Tool** — extracts up to 5 unique keywords (words longer than 4 characters) from text
- 🤖 **Agent Logic** — conditional routing based on keywords in the query
- 📦 **Structured Output** — every response follows a consistent JSON schema
- ⚠️ **Error Handling** — invalid expressions, empty inputs, and unexpected exceptions are caught gracefully
- 📝 **Logging** — every query, routing decision, and result is logged for debugging and traceability

