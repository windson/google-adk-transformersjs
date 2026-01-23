# FunctionGemma Guide

FunctionGemma is a specialized Gemma 3 270M model tuned for function calling. It's designed as a base for fine-tuning into custom, fast, private, local agents.

## When to Use FunctionGemma

- **Defined API surface** - Apps with specific actions (smart home, media, navigation)
- **Fine-tuning required** - Need deterministic behavior over zero-shot prompting
- **Local-first deployment** - Edge devices requiring low latency and data privacy
- **Compound systems** - Lightweight edge model for local actions, offloading complex tasks to larger models

## Control Tokens

FunctionGemma uses special tokens within Gemma's turn structure (`<start_of_turn>role` / `<end_of_turn>`):

| Token Pair | Purpose |
|------------|---------|
| `<start_function_declaration>` / `<end_function_declaration>` | Define a tool |
| `<start_function_call>` / `<end_function_call>` | Model requests tool use |
| `<start_function_response>` / `<end_function_response>` | Provide tool result |

The `<escape>` token delimits all string values in structured data.

## TypeScript Usage (Transformers.js)

```typescript
import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";

const model_id = "onnx-community/functiongemma-270m-it-ONNX";
const tokenizer = await AutoTokenizer.from_pretrained(model_id);
const model = await AutoModelForCausalLM.from_pretrained(model_id);

const weather_function_schema = {
  type: "function",
  function: {
    name: "get_current_temperature",
    description: "Gets the current temperature for a given location.",
    parameters: {
      type: "object",
      properties: {
        location: { type: "string", description: "The city name, e.g. San Francisco" },
      },
      required: ["location"],
    },
  },
};

const messages = [
  { role: "developer", content: "You are a model that can do function calling with the following functions" },
  { role: "user", content: "What's the temperature in London?" },
];

const inputs = tokenizer.apply_chat_template(messages, {
  tools: [weather_function_schema],
  tokenize: true,
  add_generation_prompt: true,
  return_dict: true,
});

const output = await model.generate({ ...inputs, max_new_tokens: 512 });
const decoded = tokenizer.decode(output.slice(0, [inputs.input_ids.dims[1], null]), { skip_special_tokens: false });
// Output: <start_function_call>call:get_current_temperature{location:<escape>London<escape>}<end_function_call>
```

## Four-Step Function Calling Process

1. **Define Tools** - Create function schemas with name, description, and parameters
2. **Model's Turn** - Model generates `<start_function_call>call:fn_name{args}<end_function_call>`
3. **Developer's Turn** - Parse function call, execute it, return result via `<start_function_response>`
4. **Final Response** - Model uses tool output to generate natural language answer

## Prompt Structure

### Developer Turn (Tool Definition)
```
<start_of_turn>developer
You are a model that can do function calling with the following functions<start_function_declaration>declaration:get_weather{description:<escape>Gets weather for a location.<escape>,parameters:{...}}<end_function_declaration><end_of_turn>
```

### User Turn
```
<start_of_turn>user
What's the weather in Tokyo?<end_of_turn>
```

### Model Function Call
```
<start_function_call>call:get_weather{location:<escape>Tokyo<escape>}<end_function_call>
```

### Function Response (from your app)
```
<start_function_response>response:get_weather{temperature:15,weather:<escape>sunny<escape>}<end_function_response>
```

### Model Final Answer
```
The weather in Tokyo is sunny with 15 degrees.<end_of_turn>
```

## Supported Workflows

| Supported | Not Trained On |
|-----------|----------------|
| Single turn function calls | Multi-turn conversations |
| Parallel calls (independent) | Multi-step workflows (dependent) |

## Use Cases

- **Real-Time Data** - Weather APIs, search engines, stock prices
- **External Systems** - Send emails, manage calendars, control smart home
- **Mobile Actions** - Translate user inputs into Android OS system tool calls
- **Complex Workflows** - Chain multiple tool calls

## Best Practices

1. **Always include system prompt**: `"You are a model that can do function calling with the following functions"`

2. **Enrich tool descriptions** with semantic context:
   - Bad: `"Get the current temperature"`
   - Good: `"Get the current temperature. Can determine if weather is hot or cold."`

3. **Use `<start_function_response>` as stop sequence** in inference engine

4. **Parse function calls** - Extract function name and arguments from model output before executing

5. **Fine-tune for your domain** - Base model achieves ~58% on mobile actions, fine-tuned reaches ~85%

## On-Device Performance (S25 Ultra)

| Metric | Value |
|--------|-------|
| Prefill | ~1718 tokens/sec |
| Decode | ~126 tokens/sec |
| Time-to-first-token | 0.3s |
| Model Size | 288 MB |

## Resources

- [ONNX Model (TypeScript)](https://huggingface.co/onnx-community/functiongemma-270m-it-ONNX)
- [Original Model](https://huggingface.co/google/functiongemma-270m-it)
- [Documentation](https://ai.google.dev/gemma/docs/functiongemma)
- [Kaggle](https://www.kaggle.com/models/google/functiongemma)

## Related Steering Documents

- `#transformers-js-guide` - General Transformers.js usage
- `#transformers-js-models` - Compatible models table
