# Transformers.js Guide

Transformers.js runs 🤗 Transformers directly in the browser or Node.js using ONNX Runtime. It's functionally equivalent to the Python library with a similar API.

## Installation

```bash
npm i @huggingface/transformers
```

## Basic Usage - Pipeline API

```typescript
import { pipeline } from "@huggingface/transformers";

// Sentiment analysis
const classifier = await pipeline("sentiment-analysis");
const result = await classifier("I love transformers!");
// [{ label: 'POSITIVE', score: 0.999817686 }]

// Use a specific model
const pipe = await pipeline("sentiment-analysis", "Xenova/bert-base-multilingual-uncased-sentiment");
```

## Device Options

```typescript
// CPU (default) - via WASM
const pipe = await pipeline("text-generation", "model-id");

// GPU - via WebGPU (experimental, ~70% browser support)
const pipe = await pipeline("text-generation", "model-id", { device: "webgpu" });
```

## Quantization (dtype)

Use quantized models for better performance and lower bandwidth:

| dtype | Description | Default For |
|-------|-------------|-------------|
| `"fp32"` | Full precision | WebGPU |
| `"fp16"` | Half precision | - |
| `"q8"` | 8-bit quantized | WASM |
| `"q4"` | 4-bit quantized | - |

```typescript
// 4-bit quantization with WebGPU
const generator = await pipeline(
  "text-generation",
  "onnx-community/Qwen2.5-0.5B-Instruct",
  { dtype: "q4", device: "webgpu" }
);
```

### Per-Module dtypes (for encoder-decoder models)

```typescript
import { Florence2ForConditionalGeneration } from "@huggingface/transformers";

const model = await Florence2ForConditionalGeneration.from_pretrained(
  "onnx-community/Florence-2-base-ft",
  {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      encoder_model: "q4",
      decoder_model_merged: "q4",
    },
    device: "webgpu",
  }
);
```

## Chat/Instruct Models

```typescript
const generator = await pipeline(
  "text-generation",
  "onnx-community/Qwen2.5-0.5B-Instruct",
  { dtype: "q4", device: "webgpu" }
);

const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "Tell me a joke." },
];

const output = await generator(messages, { max_new_tokens: 128 });
console.log(output[0].generated_text.at(-1).content);
```

## Supported Tasks

| Category | Tasks |
|----------|-------|
| NLP | text-classification, text-generation, summarization, translation, question-answering, fill-mask, token-classification, zero-shot-classification |
| Vision | image-classification, object-detection, image-segmentation, depth-estimation, background-removal |
| Audio | automatic-speech-recognition, audio-classification, text-to-speech |
| Multimodal | image-to-text, document-question-answering, zero-shot-image-classification |

## Best Practices

1. **Use ONNX models** - Look for models with `transformers.js` tag or from `onnx-community`
2. **Choose appropriate quantization** - Use `q4` or `q8` for browser, `fp16`/`fp32` for accuracy
3. **WebGPU for performance** - Enable with `device: "webgpu"` when supported
4. **Singleton pattern** - Reuse pipeline instances instead of recreating them
5. **Skip local model check** - In browser: `env.allowLocalModels = false`

## Environment Configuration

```typescript
import { env } from "@huggingface/transformers";

// Skip local model check (browser)
env.allowLocalModels = false;

// Use custom model path
env.localModelPath = "/models/";
```

## Resources

- [Documentation](https://huggingface.co/docs/transformers.js)
- [Models](https://huggingface.co/models?library=transformers.js)
- [Examples](https://github.com/huggingface/transformers.js-examples)
