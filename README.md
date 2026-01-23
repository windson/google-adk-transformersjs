# ADK + Transformers.js Proof of Concept

This is a proof-of-concept demonstrating how to use Google's Agent Development Kit (ADK) with local Transformers.js models as the LLM backend.

## Overview

This PoC shows:
1. **Custom BaseLlm Implementation**: `TransformersLlm` class that wraps Transformers.js
2. **FunctionGemma for Tool Calling**: Uses FunctionGemma-270M for reliable function calling
3. **Stop Sequence Handling**: Proper implementation per FunctionGemma documentation
4. **ADK Agent with Tools**: Full ADK agent with function tools (calculator, time, weather)
5. **No Cloud Required**: 100% local inference, no API keys needed

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ADK Agent (LlmAgent)                     │
│  - Instruction: "You are a function calling model..."       │
│  - Tools: [calculator, get_current_time, get_weather]       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              TransformersLlm (extends BaseLlm)              │
│  - Converts LlmRequest → Transformers.js format             │
│  - Configures stop sequences for FunctionGemma              │
│  - Runs local inference                                     │
│  - Converts output → LlmResponse                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Transformers.js                          │
│  - @huggingface/transformers (Node.js)                      │
│  - Model: onnx-community/functiongemma-270m-it-ONNX         │
│  - Stop sequences: <start_function_response>, <end_of_turn> │
└─────────────────────────────────────────────────────────────┘
```

## FunctionGemma Stop Sequence Handling

Per the [FunctionGemma documentation](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices):

> `<start_function_response>` is an additional stop sequence for the inference engine.

This implementation properly configures stop sequences:

```typescript
// Stop tokens configured during initialization
const stopSequences = [
  '<start_function_response>',  // Model expects app to provide function result
  '<end_of_turn>',              // Normal end of turn
];

// Passed to generate() as eos_token_id
generateConfig.eos_token_id = [eosTokenId, ...stopTokenIds];
```

**Why this matters:**
- FunctionGemma outputs: `<start_function_call>call:func{args}<end_function_call><start_function_response>`
- The model stops at `<start_function_response>` waiting for the app to provide the result
- Without this, the model would continue generating (potentially hallucinating results)

## FunctionGemma Limitations

FunctionGemma is designed as a **base model for fine-tuning**:

1. **Single-turn only**: Not trained for multi-turn conversations
2. **No text responses after tools**: Base model doesn't generate natural language after receiving function results
3. **Needs fine-tuning**: For production, fine-tune on your specific use case

This PoC handles these limitations by:
- Using stop sequences to prevent infinite loops
- Returning tool results directly when function response is in history
- Optionally using SmolLM2 for natural language response generation

## Prerequisites

- Node.js 18+ (for Transformers.js)
- npm or yarn

## Installation

```bash
npm install
```

## Usage

### Build

```bash
npm run build
```

### Run the Agent

```bash
npm start -- "What is 5 + 3?"
npm start -- "What is 15% of 200?"
npm start -- "What time is it?"
npm start -- "What's the weather in Tokyo?"
```

### Interactive Mode

```bash
npm start
```

## Files

- `transformers-llm.ts` - Custom BaseLlm with stop sequence support
- `agent.ts` - ADK agent definition with tools
- `index.ts` - Main entry point with InMemoryRunner

## Example Output

```
[User]: What is 5 + 3?
[Agent]: Thinking...

[TransformersLlm] Stop sequence "<start_function_response>" -> token ID 50
[TransformersLlm] Stop sequence "<end_of_turn>" -> token ID 106
[TransformersLlm:functiongemma] Using stop tokens: 1, 50, 106
[TransformersLlm:functiongemma] Raw output: <start_function_call>call:calculator{expression:<escape>5 + 3<escape>}<end_function_call><start_function_response>

🔧 [Tool Call]: calculator
   Args: { "expression": "5 + 3" }

🔧 [Calculator] Evaluating: 5 + 3
✅ [Calculator] Result: 8

[Agent]: Tool result: {"status":"success","expression":"5 + 3","result":"8"}
[Tools Used]: Yes
```

## Limitations

1. **FunctionGemma Base Model**: Needs fine-tuning for multi-turn and text responses
2. **First Run**: Initial model download takes ~300MB
3. **Performance**: CPU inference is slower than GPU
4. **Node.js Cleanup**: May see mutex error on exit (known ONNX issue, doesn't affect functionality)

## Models Used

| Model | Purpose | Size |
|-------|---------|------|
| FunctionGemma-270M | Tool/function calling | ~300MB |
| SmolLM2-135M | Natural language responses (optional) | ~150MB |

## References

- [FunctionGemma Documentation](https://ai.google.dev/gemma/docs/functiongemma)
- [FunctionGemma Formatting Guide](https://ai.google.dev/gemma/docs/functiongemma/formatting-and-best-practices)
- [FunctionGemma ONNX on HuggingFace](https://huggingface.co/onnx-community/functiongemma-270m-it-ONNX)
- [Google ADK](https://github.com/google/adk-js)
- [Transformers.js](https://huggingface.co/docs/transformers.js)

## License

Apache 2.0
