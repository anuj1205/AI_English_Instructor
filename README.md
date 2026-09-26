# AI English Instructor

> Version 1.0 — Basic Voice-Based English Conversation Assistant

A basic local AI English instructor designed to help practice spoken English through voice interaction.

The first version focuses on the fundamental pipeline:

**Speech → Speech-to-Text → AI Response → Text-to-Speech**

The project runs locally using open-source/local AI tools.

---

## Version 1.0

This is the **first and most basic working version** of the project.

The goal of Version 1 is not to provide a complete English-learning platform, but to establish the core voice conversation pipeline and verify that all major components work together.

### Current capabilities

- Capture user's voice through microphone
- Convert speech into text using Whisper
- Detect the spoken language
- Send the text to a local LLM
- Generate an AI response
- Convert the AI response into speech using Piper
- Play the generated response

### Current conversation flow

```text
User speaks
     ↓
Microphone
     ↓
Audio recording
     ↓
Whisper
     ↓
Text
     ↓
Ollama + Qwen
     ↓
AI response
     ↓
Piper TTS
     ↓
Spoken response
