# Building a Next.js Chatbot with ADK + Transformers.js

This guide covers building a local AI chatbot using Next.js, shadcn/ui, Google ADK, and Transformers.js for 100% local inference.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Next.js Frontend                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ ChatContainer│  │ ChatInput   │  │ ChatMessage         │ │
│  │ (shadcn/ui) │  │             │  │ (tool calls, agents)│ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└────────────────────────────┬────────────────────────────────┘
                             │ POST /api/chat
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Next.js API Route                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              InMemoryRunner (ADK)                    │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │         Coordinator Agent (LlmAgent)        │    │   │
│  │  │  - Routes requests to appropriate tools     │    │   │
│  │  │  - Uses FunctionGemma for tool calling      │    │   │
│  │  │  - Tools: calculator, time, weather         │    │   │
│  │  └─────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              TransformersLlm (Custom BaseLlm)               │
│  - Bridges ADK with Transformers.js                         │
│  - Loads ONNX models locally                                │
│  - Parses FunctionGemma tool call format                    │
│  - Model: onnx-community/functiongemma-270m-it-ONNX         │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
chatbot-ui/
├── src/
│   ├── app/
│   │   ├── api/chat/route.ts    # API endpoint
│   │   ├── page.tsx             # Main page
│   │   ├── layout.tsx           # Root layout
│   │   └── globals.css          # Tailwind styles
│   ├── components/
│   │   ├── chat/
│   │   │   ├── chat-container.tsx
│   │   │   ├── chat-input.tsx
│   │   │   └── chat-message.tsx
│   │   └── ui/                  # shadcn components
│   └── lib/
│       ├── agent.ts             # ADK agent setup
│       ├── transformers-llm.ts  # Custom LLM bridge
│       └── utils.ts
├── package.json
└── next.config.ts
```

## Dependencies

```json
{
  "dependencies": {
    "@google/adk": "^0.2.4",
    "@huggingface/transformers": "^3.8.1",
    "next": "16.x",
    "react": "19.x",
    "zod": "^4.x",
    // shadcn/ui components
    "@radix-ui/react-avatar": "^1.x",
    "@radix-ui/react-scroll-area": "^1.x",
    "class-variance-authority": "^0.7.x",
    "clsx": "^2.x",
    "lucide-react": "^0.5x",
    "tailwind-merge": "^3.x"
  }
}
```

## Step 1: Custom BaseLlm for Transformers.js

Create `src/lib/transformers-llm.ts` to bridge ADK with local models:

```typescript
import { BaseLlm } from '@google/adk';
import type { LlmRequest, LlmResponse, BaseLlmConnection } from '@google/adk';

// Lazy import to avoid SSR issues
let transformersModule: typeof import('@huggingface/transformers') | null = null;

async function getTransformers() {
  if (!transformersModule) {
    transformersModule = await import('@huggingface/transformers');
  }
  return transformersModule;
}

// Model cache to avoid reloading
const modelCache = new Map();

export class TransformersLlm extends BaseLlm {
  private modelId: string;
  private toolSchemas: ToolSchema[] = [];
  
  constructor(params: { model: string; maxNewTokens?: number }) {
    super({ model: params.model });
    this.modelId = params.model;
  }

  setToolSchemas(schemas: ToolSchema[]): void {
    this.toolSchemas = schemas;
  }

  override async *generateContentAsync(
    llmRequest: LlmRequest
  ): AsyncGenerator<LlmResponse, void> {
    // Initialize model, format messages, generate, parse output
    // See full implementation in chatbot-ui/src/lib/transformers-llm.ts
  }
}
```

Key implementation details:
- Lazy load `@huggingface/transformers` to avoid SSR issues
- Cache models to avoid reloading on each request
- Parse FunctionGemma's special token format for tool calls
- Handle stop sequences (`<start_function_response>`, `<end_of_turn>`)

## Step 2: ADK Agent with Tools

Create `src/lib/agent.ts`:

```typescript
import { LlmAgent, FunctionTool } from '@google/adk';
import { z } from 'zod';
import { TransformersLlm } from './transformers-llm';

// Tool schemas for FunctionGemma
const toolSchemas = [
  {
    type: 'function',
    function: {
      name: 'calculator',
      description: 'Performs math calculations',
      parameters: {
        type: 'object',
        properties: {
          expression: { type: 'string', description: 'Math expression' },
        },
        required: ['expression'],
      },
    },
  },
  // Add more tool schemas...
];

// Create tools
const calculatorTool = new FunctionTool({
  name: 'calculator',
  description: 'Performs mathematical calculations',
  parameters: z.object({
    expression: z.string(),
  }),
  execute: ({ expression }) => {
    const result = new Function(`return (${expression})`)();
    return { status: 'success', result: String(result) };
  },
});

// Create LLM with tool schemas
const llm = new TransformersLlm({
  model: 'onnx-community/functiongemma-270m-it-ONNX',
  maxNewTokens: 128,
});
llm.setToolSchemas(toolSchemas);

// Create coordinator agent
export const coordinatorAgent = new LlmAgent({
  name: 'Coordinator',
  model: llm as any,
  instruction: `You are a helpful assistant with tools.
Use calculator for math, get_time for time, get_weather for weather.`,
  tools: [calculatorTool, timeTool, weatherTool],
});
```

## Step 3: API Route

Create `src/app/api/chat/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { InMemoryRunner } from '@google/adk';
import { coordinatorAgent } from '@/lib/agent';

const APP_NAME = 'MultiAgentChatbot';
let runner: InMemoryRunner | null = null;

function getRunner() {
  if (!runner) {
    runner = new InMemoryRunner({
      agent: coordinatorAgent,
      appName: APP_NAME,
    });
  }
  return runner;
}

export async function POST(request: NextRequest) {
  const { message } = await request.json();
  const currentRunner = getRunner();
  
  // Create fresh session per request
  const sessionId = `session-${Date.now()}`;
  await currentRunner.sessionService.createSession({
    appName: APP_NAME,
    userId: 'user',
    sessionId,
  });

  const toolCalls: ToolCall[] = [];
  let response = '';

  // Run agent and collect events
  for await (const event of currentRunner.runAsync({
    userId: 'user',
    sessionId,
    newMessage: { role: 'user', parts: [{ text: message }] },
  })) {
    if (event.content?.parts) {
      for (const part of event.content.parts) {
        if ('functionCall' in part) {
          toolCalls.push({ name: part.functionCall.name, args: part.functionCall.args });
        }
        if ('functionResponse' in part) {
          response = formatResponse(part.functionResponse.name, part.functionResponse.response);
        }
      }
    }
  }

  return NextResponse.json({ response, toolCalls });
}
```

## Step 4: Chat UI Components

### ChatContainer (main component)

```tsx
'use client';
import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';

export function ChatContainer() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (content: string) => {
    setMessages(prev => [...prev, { role: 'user', content }]);
    setIsLoading(true);

    const res = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message: content }),
    });
    const data = await res.json();

    setMessages(prev => [...prev, {
      role: 'assistant',
      content: data.response,
      toolCalls: data.toolCalls,
    }]);
    setIsLoading(false);
  };

  return (
    <Card>
      <ScrollArea>
        {messages.map(msg => <ChatMessage key={msg.id} message={msg} />)}
        <div ref={scrollRef} />
      </ScrollArea>
      <ChatInput onSend={handleSend} isLoading={isLoading} />
    </Card>
  );
}
```

### ChatMessage (displays tool calls)

```tsx
export function ChatMessage({ message }: { message: Message }) {
  return (
    <div className={cn('flex gap-3', message.role === 'user' ? 'flex-row-reverse' : '')}>
      <Avatar>{message.role === 'user' ? <User /> : <Bot />}</Avatar>
      <div>
        {/* Show tool calls */}
        {message.toolCalls?.map((tool, i) => (
          <Card key={i} className="bg-amber-50">
            <Badge>{tool.name}</Badge>
            <code>Args: {JSON.stringify(tool.args)}</code>
            {tool.result && <code>Result: {JSON.stringify(tool.result)}</code>}
          </Card>
        ))}
        {/* Main content */}
        <Card>{message.content}</Card>
      </div>
    </div>
  );
}
```

## FunctionGemma Token Format

FunctionGemma uses special tokens for tool calling:

```
Input:  <start_of_turn>developer
        You are a model that can do function calling...
        <start_function_declaration>declaration:calculator{...}<end_function_declaration>
        <end_of_turn>
        <start_of_turn>user
        What is 5 + 3?<end_of_turn>

Output: <start_function_call>call:calculator{expression:<escape>5+3<escape>}<end_function_call>
```

Parse with regex:
```typescript
const match = text.match(/<start_function_call>call:(\w+)\{([^}]*)\}<end_function_call>/);
const name = match[1];
const argsStr = match[2];
// Parse args: key:<escape>value<escape>
```

## Multi-Agent Patterns

### Coordinator/Dispatcher (used in this project)

```typescript
const coordinator = new LlmAgent({
  name: 'Coordinator',
  model: llm,
  instruction: 'Route to appropriate tool based on request',
  tools: [calculatorTool, timeTool, weatherTool],
});
```

### Sequential Pipeline (for multi-step tasks)

```typescript
import { SequentialAgent } from '@google/adk';

const pipeline = new SequentialAgent({
  name: 'Pipeline',
  subAgents: [
    new LlmAgent({ name: 'Analyzer', outputKey: 'analysis' }),
    new LlmAgent({ name: 'Responder', instruction: 'Use {analysis}' }),
  ],
});
```

## Best Practices

1. **Lazy load Transformers.js** - Avoid SSR issues with dynamic imports
2. **Cache models** - Don't reload on every request
3. **Fresh sessions** - Create new session per request for stateless API
4. **Format responses** - Convert tool results to natural language
5. **Show tool calls** - Display what tools were used for transparency
6. **Handle errors** - Graceful fallbacks for model/tool failures

## Related Steering Documents

- `#adk-typescript-guide` - ADK basics and built-in tools
- `#adk-multi-agent-systems` - Multi-agent patterns
- `#functiongemma-guide` - FunctionGemma token format
- `#transformers-js-guide` - Transformers.js usage
- `#transformers-js-models` - Compatible models
