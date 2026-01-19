# How to Build a 100% Local AI Agent (No Cloud, No Costs)

**Using Next.js 16, Transformers.js, and WebGPU**

> **Updated January 2026** - This guide has been revised based on real-world implementation experience with Next.js 16 and Transformers.js v3+.

Imagine an AI agent that runs entirely in your browser. It doesn't send your data to the cloud, it costs $0 to run, and it works offline. This guide shows you exactly how to build it.

## What We're Building

A chat interface powered by SmolLM2-135M that:
- ✅ Runs 100% in the browser (no server calls)
- ✅ Uses WebGPU for GPU acceleration (with WASM fallback)
- ✅ Streams responses in real-time
- ✅ Works offline after initial model download
- ✅ Costs nothing to run

## Technologies

| Technology | Purpose |
|------------|---------|
| **Next.js 16** | React framework with Turbopack |
| **Transformers.js v3+** | Run AI models in browser via ONNX Runtime |
| **WebGPU** | GPU acceleration for faster inference |
| **Web Workers** | Background processing (no UI freezing) |
| **shadcn/ui** | Beautiful, accessible UI components |

> **Note on Google ADK**: The original tutorial mentioned Google ADK for agent orchestration. In practice, for simple chat interfaces, ADK adds unnecessary complexity. We focus on the core: Transformers.js + Web Workers. ADK becomes valuable when you need multi-agent systems, tool calling, or complex planning.

---

## Phase 1: Project Setup

### 1. Create Next.js Project

```bash
npx create-next-app@latest local-agent
cd local-agent
```

Select: TypeScript ✅, ESLint ✅, Tailwind CSS ✅, App Router ✅

### 2. Install Dependencies

```bash
npm install @huggingface/transformers
```

### 3. Add shadcn/ui Components

```bash
npx shadcn@latest init
npx shadcn@latest add card button input scroll-area badge progress slider switch label tooltip
```

### 4. Configure Next.js for Transformers.js (Critical!)

Transformers.js uses ONNX Runtime which requires specific bundler configuration. Update `next.config.ts`:

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // External packages - don't bundle server-side ONNX
  serverExternalPackages: ["sharp"],

  // Turbopack (Next.js 16 default)
  turbopack: {},

  // Webpack configuration (fallback or when using --webpack flag)
  webpack: (config) => {
    // Prevent bundling node-specific modules
    config.resolve.alias = {
      ...config.resolve.alias,
      sharp$: false,
      "onnxruntime-node$": false,
    };

    // Enable WebAssembly support for ONNX Runtime
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
      layers: true,
    };

    // Handle .wasm files
    config.module.rules.push({
      test: /\.wasm$/,
      type: "asset/resource",
    });

    return config;
  },
};

export default nextConfig;
```

**Why this matters:**
- `onnxruntime-node$: false` prevents Node.js ONNX binaries from being bundled
- `asyncWebAssembly: true` enables WASM support for browser ONNX runtime
- Without this, you'll get cryptic build errors about missing binaries

---

## Phase 2: The Web Worker (AI Engine)

Running AI inference on the main thread freezes the UI. We use a Web Worker for background processing.

### Create `worker.js`

```javascript
// app/ai-tools/local-agent/worker.js
import { pipeline, TextStreamer, env } from "@huggingface/transformers";

// Fetch models from Hugging Face Hub
env.allowLocalModels = false;

// Track which device is actually being used
let actualDevice = "wasm";

// Check WebGPU availability (not all browsers support it)
async function checkWebGPUSupport() {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    console.log("[Worker] WebGPU not available");
    return false;
  }
  
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log("[Worker] WebGPU adapter not available");
      return false;
    }
    console.log("[Worker] ✅ WebGPU available");
    return true;
  } catch (error) {
    console.log("[Worker] WebGPU check failed:", error.message);
    return false;
  }
}

// Singleton pattern - load model once, reuse for all requests
class PipelineSingleton {
  static task = "text-generation";
  static model = "HuggingFaceTB/SmolLM2-135M-Instruct";
  static instance = null;
  static currentDevice = null;

  static async getInstance(useWebGPU = true, progress_callback = null) {
    let deviceToUse = useWebGPU ? "webgpu" : "wasm";
    
    // Verify WebGPU is actually available
    if (deviceToUse === "webgpu") {
      const hasWebGPU = await checkWebGPUSupport();
      if (!hasWebGPU) {
        console.log("[Worker] Falling back to WASM");
        deviceToUse = "wasm";
      }
    }
    
    // Recreate pipeline if device preference changed
    if (this.instance === null || this.currentDevice !== deviceToUse) {
      console.log(`[Worker] Creating pipeline with device: ${deviceToUse}`);
      this.currentDevice = deviceToUse;
      actualDevice = deviceToUse;
      
      try {
        this.instance = await pipeline(this.task, this.model, {
          device: deviceToUse,
          dtype: "q4", // 4-bit quantization (~100MB instead of ~500MB)
          progress_callback,
        });
      } catch (error) {
        // WebGPU can fail at runtime even if available
        if (deviceToUse === "webgpu") {
          console.warn("[Worker] WebGPU failed, falling back to WASM");
          actualDevice = "wasm";
          this.currentDevice = "wasm";
          this.instance = await pipeline(this.task, this.model, {
            device: "wasm",
            dtype: "q4",
            progress_callback,
          });
        } else {
          throw error;
        }
      }
    }
    return this.instance;
  }
}

// Handle messages from main thread
self.addEventListener("message", async (event) => {
  const { type, text, useWebGPU, generationConfig } = event.data;

  if (type === "generate") {
    try {
      self.postMessage({ status: "start", device: actualDevice });

      // Load model (shows download progress on first use)
      const generator = await PipelineSingleton.getInstance(useWebGPU, (progress) => {
        self.postMessage({ status: "progress", data: progress, device: actualDevice });
      });

      self.postMessage({ status: "generating", device: actualDevice });

      const genStartTime = Date.now();

      // TextStreamer enables real-time token streaming
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: (token) => {
          self.postMessage({ status: "stream", text: token, device: actualDevice });
        },
      });

      // Run inference
      const output = await generator(text, {
        max_new_tokens: generationConfig?.max_new_tokens || 128,
        temperature: generationConfig?.temperature || 0.7,
        top_p: generationConfig?.top_p || 0.9,
        do_sample: generationConfig?.do_sample !== false,
        streamer,
      });

      const genTime = Date.now() - genStartTime;

      self.postMessage({
        status: "complete",
        output: output[0]?.generated_text || "",
        device: actualDevice,
        generationTime: genTime,
      });
    } catch (error) {
      self.postMessage({
        status: "error",
        error: `${error.name}: ${error.message}`,
        device: actualDevice,
      });
    }
  }
});

// Global error handlers (important for debugging)
self.addEventListener("error", (event) => {
  self.postMessage({ status: "error", error: `Global error: ${event.error?.message}` });
});

self.addEventListener("unhandledrejection", (event) => {
  self.postMessage({ status: "error", error: `Unhandled rejection: ${event.reason}` });
});

console.log("[Worker] Ready");
```

**Key Implementation Details:**

1. **WebGPU Detection**: Always check if WebGPU is actually available, not just if the API exists
2. **Graceful Fallback**: If WebGPU fails at any point, fall back to WASM
3. **TextStreamer**: Use `TextStreamer` for real-time token streaming (better UX than waiting for full response)
4. **4-bit Quantization**: `dtype: "q4"` reduces model size from ~500MB to ~100MB
5. **Error Handlers**: Global error handlers catch unhandled exceptions in workers

---

## Phase 3: React Hook (Bridge to Worker)

Create a custom hook to manage worker communication:

```typescript
// app/ai-tools/local-agent/use-local-agent.ts
"use client";

import { useEffect, useRef, useState, useCallback } from "react";

export type AgentStatus = "idle" | "loading" | "generating" | "error";

interface GenerationConfig {
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  do_sample?: boolean;
}

export function useLocalAgent(config?: GenerationConfig) {
  const worker = useRef<Worker | null>(null);
  const [result, setResult] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [status, setStatus] = useState<AgentStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("");
  const [device, setDevice] = useState<"webgpu" | "wasm" | null>(null);
  const [generationTime, setGenerationTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useWebGPU, setUseWebGPU] = useState(true);

  useEffect(() => {
    if (!worker.current) {
      try {
        // Initialize worker with module type for ES imports
        worker.current = new Worker(
          new URL("./worker.js", import.meta.url),
          { type: "module" }
        );
      } catch (err) {
        setError(`Failed to initialize worker: ${err}`);
        return;
      }
    }

    const onMessage = (e: MessageEvent) => {
      const { status: msgStatus, data, output, error: workerError, 
              device: workerDevice, generationTime: genTime, text: streamText } = e.data;

      if (workerDevice) setDevice(workerDevice);

      switch (msgStatus) {
        case "start":
          setStatus("loading");
          setProgress(0);
          setProgressText("Initializing...");
          break;

        case "progress":
          // Handle various progress states from Transformers.js
          if (data?.status === "download") {
            const pct = data.progress ? Math.round(data.progress) : 0;
            setProgress(pct);
            if (data.loaded && data.total) {
              const loadedMB = (data.loaded / (1024 * 1024)).toFixed(1);
              const totalMB = (data.total / (1024 * 1024)).toFixed(1);
              setProgressText(`Downloading: ${data.file} (${loadedMB}MB / ${totalMB}MB)`);
            }
          } else if (data?.status === "ready") {
            setProgress(100);
            setProgressText("Model ready!");
          }
          break;

        case "generating":
          setStatus("generating");
          setStreamingText("");
          break;

        case "stream":
          // Accumulate streaming tokens
          if (streamText) {
            setStreamingText((prev) => prev + streamText);
          }
          break;

        case "complete":
          setStatus("idle");
          if (genTime) setGenerationTime(genTime);
          if (output) setResult(output);
          setStreamingText("");
          break;

        case "error":
          setStatus("error");
          setError(workerError || "An error occurred");
          break;
      }
    };

    worker.current.addEventListener("message", onMessage);
    worker.current.addEventListener("error", (e) => {
      setError(`Worker error: ${e.message}`);
      setStatus("error");
    });

    return () => {
      worker.current?.removeEventListener("message", onMessage);
    };
  }, []);

  const generate = useCallback((prompt: string) => {
    if (!worker.current || !prompt.trim()) return;

    // Format prompt for SmolLM2 (ChatML format)
    const formattedPrompt = `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;

    setError(null);
    setResult("");
    setStreamingText("");
    setGenerationTime(null);

    worker.current.postMessage({
      type: "generate",
      text: formattedPrompt,
      useWebGPU,
      generationConfig: config,
    });
  }, [useWebGPU, config]);

  const reset = useCallback(() => {
    setResult("");
    setStreamingText("");
    setStatus("idle");
    setProgress(0);
    setGenerationTime(null);
    setError(null);
  }, []);

  return {
    generate,
    result,
    streamingText,
    status,
    progress,
    progressText,
    device,
    generationTime,
    error,
    useWebGPU,
    setUseWebGPU,
    reset,
  };
}
```

**Key Points:**

1. **Worker Initialization**: Use `new URL("./worker.js", import.meta.url)` for proper module resolution
2. **ChatML Format**: SmolLM2 expects prompts in ChatML format (`<|im_start|>user\n...<|im_end|>`)
3. **Streaming State**: Track both `streamingText` (in-progress) and `result` (final)
4. **Device Toggle**: Allow users to switch between WebGPU and WASM

---

## Phase 4: Chat UI Component

```tsx
// app/ai-tools/local-agent/agent-chat.tsx
"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Bot, Send, Loader2, Zap, Cpu, Trash2 } from "lucide-react";
import { useLocalAgent } from "./use-local-agent";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export function AgentChat() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  const {
    generate, result, streamingText, status, progress, progressText,
    device, generationTime, error, useWebGPU, setUseWebGPU, reset,
  } = useLocalAgent({ max_new_tokens: 128, temperature: 0.7 });

  // Auto-scroll on new content
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, streamingText]);

  // Add assistant response when complete
  useEffect(() => {
    if (result && status === "idle") {
      setMessages((prev) => {
        if (prev[prev.length - 1]?.role === "assistant") return prev;
        return [...prev, { role: "assistant", content: result }];
      });
    }
  }, [result, status]);

  const handleSend = () => {
    if (!input.trim() || status === "loading" || status === "generating") return;
    setMessages((prev) => [...prev, { role: "user", content: input }]);
    generate(input);
    setInput("");
  };

  const isLoading = status === "loading" && progress < 100;
  const isGenerating = status === "generating";

  return (
    <div className="space-y-6">
      {/* Settings */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Switch
                checked={useWebGPU}
                onCheckedChange={setUseWebGPU}
                disabled={isLoading || isGenerating}
              />
              <Label>{useWebGPU ? "WebGPU (GPU)" : "WASM (CPU)"}</Label>
            </div>
            {device && (
              <Badge variant={device === "webgpu" ? "default" : "secondary"}>
                {device === "webgpu" ? <Zap className="h-3 w-3 mr-1" /> : <Cpu className="h-3 w-3 mr-1" />}
                {device.toUpperCase()}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Loading Progress */}
      {isLoading && (
        <Card className="border-blue-200 bg-blue-50">
          <CardContent className="pt-6">
            <div className="flex items-center gap-3 mb-3">
              <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
              <span className="text-sm font-medium">Loading AI Model...</span>
              <span className="ml-auto text-sm font-semibold">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
            {progressText && <p className="text-xs text-blue-600 mt-2">{progressText}</p>}
          </CardContent>
        </Card>
      )}

      {/* Error */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="pt-6">
            <p className="text-sm text-red-700">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Chat */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5" /> Local AI Chat
          </CardTitle>
          {messages.length > 0 && (
            <Button variant="ghost" size="sm" onClick={() => { setMessages([]); reset(); }}>
              <Trash2 className="h-4 w-4 mr-1" /> Clear
            </Button>
          )}
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[400px] pr-4 mb-4" ref={scrollRef}>
            {messages.length === 0 && !streamingText ? (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                <Bot className="h-12 w-12 mb-4 opacity-50" />
                <p>Start chatting with your local AI agent.</p>
                <p className="text-xs mt-1">100% private - runs in your browser.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className={`max-w-[85%] px-4 py-2 rounded-lg ${
                      msg.role === "user" ? "bg-primary text-primary-foreground" : "bg-muted"
                    }`}>
                      <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                ))}
                {streamingText && (
                  <div className="flex justify-start">
                    <div className="max-w-[85%] px-4 py-2 rounded-lg bg-muted">
                      <p className="text-sm whitespace-pre-wrap">{streamingText}</p>
                      <span className="inline-block w-2 h-4 bg-foreground/50 animate-pulse" />
                    </div>
                  </div>
                )}
                {isGenerating && !streamingText && (
                  <div className="flex justify-start">
                    <div className="px-4 py-2 rounded-lg bg-muted">
                      <Loader2 className="h-4 w-4 animate-spin" />
                    </div>
                  </div>
                )}
              </div>
            )}
          </ScrollArea>

          {generationTime && status === "idle" && (
            <p className="text-xs text-muted-foreground mb-2">
              Generated in {(generationTime / 1000).toFixed(1)}s
            </p>
          )}

          <div className="flex gap-2">
            <Input
              placeholder="Ask your local AI agent..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              disabled={isLoading || isGenerating}
            />
            <Button onClick={handleSend} disabled={isLoading || isGenerating || !input.trim()}>
              {isGenerating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

---

## Phase 5: Page Component

```tsx
// app/ai-tools/local-agent/page.tsx
import { Metadata } from "next";
import dynamic from "next/dynamic";
import { Loader2 } from "lucide-react";

// Dynamic import prevents SSR issues with Web Workers
const AgentChat = dynamic(
  () => import("./agent-chat").then((mod) => mod.AgentChat),
  {
    loading: () => (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    ),
  }
);

export const metadata: Metadata = {
  title: "Local AI Chat Agent - 100% Private Browser AI",
  description: "Chat with AI running entirely in your browser. No cloud, no costs, works offline.",
};

export default function LocalAgentPage() {
  return (
    <main className="container mx-auto px-4 py-8 max-w-4xl">
      <h1 className="text-3xl font-bold mb-2">Local AI Chat Agent</h1>
      <p className="text-muted-foreground mb-6">
        Chat with AI that runs 100% in your browser - no cloud, no costs, works offline.
      </p>
      <AgentChat />
    </main>
  );
}
```

---

## Run It

```bash
npm run dev
```

1. Open `http://localhost:3000/ai-tools/local-agent`
2. First message triggers model download (~100MB, cached after)
3. Watch the streaming response
4. Disconnect Wi-Fi - it still works!

---

## Lessons Learned (Real-World Implementation)

### What the Original Tutorial Got Right
- ✅ Web Workers are essential for non-blocking inference
- ✅ Singleton pattern prevents redundant model loading
- ✅ ChatML prompt format for SmolLM2

### What We Improved

| Original | Improved |
|----------|----------|
| Assumed WebGPU always works | Added runtime WebGPU detection + WASM fallback |
| Basic progress tracking | Detailed download progress with file names and sizes |
| `callback_function` for streaming | `TextStreamer` class (cleaner, official API) |
| No error handling | Global error handlers + graceful degradation |
| Fixed generation params | User-configurable temperature, max_tokens |
| No device indicator | Shows WebGPU/WASM status in UI |
| TypeScript worker | JavaScript worker (avoids TS compilation issues in workers) |

### What We Removed

- **Google ADK**: Adds complexity without benefit for simple chat. Use ADK when you need:
  - Multi-agent orchestration
  - Tool calling / function execution
  - Complex planning and memory
  - Integration with Google Cloud services

### Performance Tips

1. **Use 4-bit quantization** (`dtype: "q4"`) - 5x smaller download, minimal quality loss
2. **Prefer WebGPU** - 2-5x faster than WASM on supported browsers
3. **Cache the model** - Browser caches after first download
4. **Stream responses** - Better UX than waiting for full generation

### Browser Support

| Browser | WebGPU | WASM | Notes |
|---------|--------|------|-------|
| Chrome 113+ | ✅ | ✅ | Best support |
| Edge 113+ | ✅ | ✅ | Same as Chrome |
| Firefox | ❌ | ✅ | WebGPU behind flag |
| Safari | ❌ | ✅ | WebGPU in development |
| Mobile | ⚠️ | ✅ | WebGPU limited, WASM slower |

---

## Why This Matters

You've built an architecture that:

1. **Runs AI locally** - Complete privacy, no data leaves the device
2. **Costs $0** - No API keys, no usage limits, no cloud bills
3. **Works offline** - After initial download, no internet needed
4. **Uses modern web APIs** - WebGPU, Web Workers, ES Modules

This demonstrates that powerful AI applications don't require massive cloud infrastructure. The browser is becoming a capable AI runtime.

---

## Next Steps

- **Larger models**: Try `Qwen2.5-0.5B-Instruct` for better quality
- **Vision models**: Add image input with `FastVLM`
- **Conversation memory**: Implement chat history context
- **System prompts**: Customize agent personality
- **Tool calling**: Add function execution capabilities
- **ADK Integration**: See below for server-side ADK with Transformers.js

---

## Advanced: Google ADK + Transformers.js (Server-Side)

For more complex agent scenarios, you can use Google ADK with Transformers.js as the LLM backend. This gives you:

- **Multi-agent orchestration**: Coordinate multiple specialized agents
- **Tool calling**: ADK's built-in tool management
- **Session management**: Conversation history and state
- **Evaluation**: Built-in agent testing framework

### When to Use ADK vs Browser-Only

| Feature | Browser-Only (This Guide) | ADK + Transformers.js |
|---------|---------------------------|----------------------|
| Runtime | Browser | Node.js server |
| Privacy | 100% client-side | Server sees data |
| Complexity | Simple | More complex |
| Multi-agent | ❌ | ✅ |
| Tool orchestration | Manual | ADK handles it |
| Offline | ✅ | ❌ (needs server) |
| Best for | Privacy-first apps | Complex agent systems |

### ADK Integration PoC

We've created a proof-of-concept showing how to integrate ADK with Transformers.js:

```
examples/adk-transformers-poc/
├── README.md              # Full documentation
├── transformers-llm.ts    # Custom BaseLlm for Transformers.js
├── agent.ts               # ADK agent with tools
├── index.ts               # Main entry (uses full ADK)
├── standalone.ts          # Standalone version (no ADK dep)
└── package.json           # Dependencies
```

**Key concept**: Create a custom `BaseLlm` class that wraps Transformers.js:

```typescript
import { BaseLlm, LlmRequest, LlmResponse } from '@google/adk';
import { pipeline } from '@huggingface/transformers';

class TransformersLlm extends BaseLlm {
  private generator: TextGenerationPipeline | null = null;

  async *generateContentAsync(
    llmRequest: LlmRequest,
    stream: boolean = false
  ): AsyncGenerator<LlmResponse, void> {
    // Initialize model if needed
    if (!this.generator) {
      this.generator = await pipeline('text-generation', 'HuggingFaceTB/SmolLM2-135M-Instruct', {
        dtype: 'q4f16',
        device: 'auto',
      });
    }

    // Convert ADK request to prompt
    const prompt = this.formatPrompt(llmRequest);
    
    // Generate with Transformers.js
    const output = await this.generator(prompt, {
      max_new_tokens: 256,
      temperature: 0.7,
    });

    // Convert to ADK response
    yield {
      content: {
        role: 'model',
        parts: [{ text: output[0].generated_text }],
      },
      finishReason: 'STOP',
    };
  }
}
```

**Run the PoC:**

```bash
cd examples/adk-transformers-poc
npm install
npm start
```

See `examples/adk-transformers-poc/README.md` for full documentation.

---

## Resources

- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js)
- [WebGPU Status](https://github.com/nicholascelestin/webgpu-status)
- [SmolLM2 Model Card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

---

*Last Updated: January 2026*
