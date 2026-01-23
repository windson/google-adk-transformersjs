# Transformers.js Compatible Models

This document lists models compatible with Transformers.js for browser/Node.js inference.

## Chat/Instruct Models

| Model | Size | Downloads | Description |
|-------|------|-----------|-------------|
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135M | 291K | Lightweight chat model |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | 53K | Mid-size chat model |
| [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://hf.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | 39K | Larger chat model |
| [onnx-community/Qwen2.5-0.5B-Instruct](https://hf.co/onnx-community/Qwen2.5-0.5B-Instruct) | 0.5B | 2K | Qwen chat model |
| [Xenova/Qwen1.5-0.5B-Chat](https://hf.co/Xenova/Qwen1.5-0.5B-Chat) | 0.5B | 1.8K | Qwen 1.5 chat |
| [onnx-community/Llama-3.2-1B-Instruct-ONNX](https://hf.co/onnx-community/Llama-3.2-1B-Instruct-ONNX) | 1B | 1.1K | Llama 3.2 instruct |
| [Xenova/TinyLlama-1.1B-Chat-v1.0](https://hf.co/Xenova/TinyLlama-1.1B-Chat-v1.0) | 1.1B | 785 | TinyLlama chat |
| [Xenova/Phi-3-mini-4k-instruct](https://hf.co/Xenova/Phi-3-mini-4k-instruct) | 3.8B | 606 | Microsoft Phi-3 |
| [onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX](https://hf.co/onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX) | 1.5B | 461 | DeepSeek R1 distilled |

## Function Calling Models

| Model | Size | Downloads | Description |
|-------|------|-----------|-------------|
| [onnx-community/functiongemma-270m-it-ONNX](https://hf.co/onnx-community/functiongemma-270m-it-ONNX) | 270M | 582 | Google FunctionGemma for tool use |

## Base/Text Generation Models

| Model | Size | Downloads | Description |
|-------|------|-----------|-------------|
| [HuggingFaceTB/SmolLM3-3B-Base](https://hf.co/HuggingFaceTB/SmolLM3-3B-Base) | 3B | 12.7K | SmolLM3 base model |
| [Xenova/distilgpt2](https://hf.co/Xenova/distilgpt2) | 82M | 6.7K | Distilled GPT-2 |
| [Xenova/gpt2](https://hf.co/Xenova/gpt2) | 124M | 3.4K | GPT-2 |
| [Xenova/llama2.c-stories15M](https://hf.co/Xenova/llama2.c-stories15M) | 15M | 2.1K | Tiny Llama for stories |
| [onnx-community/Qwen3-0.6B-ONNX](https://hf.co/onnx-community/Qwen3-0.6B-ONNX) | 0.6B | 1.3K | Qwen3 base |
| [onnx-community/gemma-3-270m-it-ONNX](https://hf.co/onnx-community/gemma-3-270m-it-ONNX) | 270M | 1.1K | Google Gemma 3 |
| [onnx-community/granite-4.0-350m-ONNX-web](https://hf.co/onnx-community/granite-4.0-350m-ONNX-web) | 350M | 2.4K | IBM Granite |
| [onnx-community/LFM2-1.2B-ONNX](https://hf.co/onnx-community/LFM2-1.2B-ONNX) | 1.2B | 397 | Liquid Foundation Model |
| [onnx-community/LFM2-350M-ONNX](https://hf.co/onnx-community/LFM2-350M-ONNX) | 350M | 393 | Liquid Foundation Model |

## Vision-Language Models

| Model | Size | Downloads | Description |
|-------|------|-----------|-------------|
| [onnx-community/Florence-2-large-ft](https://hf.co/onnx-community/Florence-2-large-ft) | Large | 1.6K | Microsoft Florence-2 |
| [onnx-community/Florence-2-base-ft](https://hf.co/onnx-community/Florence-2-base-ft) | Base | 873 | Microsoft Florence-2 |
| [onnx-community/FastVLM-0.5B-ONNX](https://hf.co/onnx-community/FastVLM-0.5B-ONNX) | 0.5B | 549 | Apple FastVLM |

## Embedding Models

| Model | Size | Downloads | Description |
|-------|------|-----------|-------------|
| [onnx-community/Qwen3-Embedding-0.6B-ONNX](https://hf.co/onnx-community/Qwen3-Embedding-0.6B-ONNX) | 0.6B | 903 | Qwen3 embeddings |
| [Maxi-Lein/Qwen3-Embedding-8B-onnx](https://hf.co/Maxi-Lein/Qwen3-Embedding-8B-onnx) | 8B | 17K | Large Qwen3 embeddings |

## Recommended Models by Use Case

### Lightweight Chat (Browser)
- **SmolLM2-135M-Instruct** - Best balance of size/quality
- **Qwen2.5-0.5B-Instruct** - Good quality, slightly larger

### Function Calling
- **functiongemma-270m-it-ONNX** - Purpose-built for tool use

### Vision + Language
- **Florence-2-base-ft** - Document understanding, OCR, captioning
- **FastVLM-0.5B-ONNX** - Fast vision-language tasks

### Edge/Mobile
- **SmolLM2-135M-Instruct** - Smallest viable chat model
- **functiongemma-270m-it-ONNX** - Local function calling

## Finding More Models

Browse models with the `transformers.js` library tag:
- [All transformers.js models](https://huggingface.co/models?library=transformers.js)
- [Text generation](https://huggingface.co/models?pipeline_tag=text-generation&library=transformers.js)
- [onnx-community](https://huggingface.co/onnx-community) - Official ONNX conversions

## Usage Example

```typescript
import { pipeline } from "@huggingface/transformers";

// Load any model from the table
const generator = await pipeline(
  "text-generation",
  "onnx-community/Qwen2.5-0.5B-Instruct",
  { dtype: "q4", device: "webgpu" }
);

const messages = [
  { role: "user", content: "Hello!" }
];

const output = await generator(messages, { max_new_tokens: 128 });
```
