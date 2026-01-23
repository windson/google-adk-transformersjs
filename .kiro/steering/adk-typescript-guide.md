# ADK TypeScript Guide

Google's Agent Development Kit (ADK) for TypeScript - an open-source, code-first toolkit for building AI agents.

## Installation

```bash
npm install @google/adk
npm install @google/adk-devtools  # For development UI
```

**Requirements:** Node.js 20.12.7+, npm 9.2.0+

## Project Setup

```
my-agent/
├── agent.ts        # Main agent code
├── package.json
├── tsconfig.json
└── .env            # GEMINI_API_KEY="your-key"
```

## Basic Agent

```typescript
import { LlmAgent } from '@google/adk';

export const rootAgent = new LlmAgent({
  name: 'my_agent',
  model: 'gemini-2.5-flash',
  description: 'A helpful assistant',
  instruction: 'You are a helpful assistant that answers questions.',
});
```

## Function Tools

```typescript
import { FunctionTool, LlmAgent } from '@google/adk';
import { z } from 'zod';

const getWeatherTool = new FunctionTool({
  name: 'get_weather',
  description: 'Gets weather for a city',
  parameters: z.object({
    city: z.string().describe('City name'),
  }),
  execute: async ({ city }) => {
    return { weather: 'sunny', temperature: 72 };
  },
});

const agent = new LlmAgent({
  name: 'weather_agent',
  model: 'gemini-2.5-flash',
  instruction: 'Help users check weather using the get_weather tool.',
  tools: [getWeatherTool],
});
```

## Built-in Tools

ADK provides several built-in tool types:

| Tool | Import | Description |
|------|--------|-------------|
| `GOOGLE_SEARCH` | `@google/adk` | Google Search - auto-invoked by Gemini 2+ models |
| `FunctionTool` | `@google/adk` | Wrap custom functions as tools with Zod schemas |
| `AgentTool` | `@google/adk` | Wrap another agent as a callable tool |
| `LongRunningFunctionTool` | `@google/adk` | For async operations that take time |
| `MCPToolset` | `@google/adk` | Connect to MCP servers for external tools |
| `BuiltInCodeExecutor` | `@google/adk` | Execute code via Gemini 2.0+ built-in executor |

### GOOGLE_SEARCH

```typescript
import { LlmAgent, GOOGLE_SEARCH } from '@google/adk';

const searchAgent = new LlmAgent({
  name: 'search_assistant',
  model: 'gemini-2.5-flash',
  instruction: 'Answer questions using Google Search when needed.',
  tools: [GOOGLE_SEARCH],
});
```

### FunctionTool

```typescript
import { FunctionTool } from '@google/adk';
import { z } from 'zod';

const myTool = new FunctionTool({
  name: 'tool_name',
  description: 'What this tool does',
  parameters: z.object({
    param1: z.string().describe('Parameter description'),
  }),
  execute: async ({ param1 }, toolContext) => {
    return { result: 'value' };
  },
});
```

### AgentTool

Wrap an agent to use as a tool within another agent:

```typescript
import { LlmAgent, AgentTool } from '@google/adk';

const specialistAgent = new LlmAgent({
  name: 'specialist',
  model: 'gemini-2.5-flash',
  description: 'Handles specialized tasks',
});

const mainAgent = new LlmAgent({
  name: 'main',
  model: 'gemini-2.5-flash',
  tools: [new AgentTool({ agent: specialistAgent })],
});
```

### LongRunningFunctionTool

For operations that may take significant time:

```typescript
import { LongRunningFunctionTool } from '@google/adk';
import { z } from 'zod';

const longTask = new LongRunningFunctionTool({
  name: 'long_task',
  description: 'A task that takes time to complete',
  parameters: z.object({ taskId: z.string() }),
  execute: async ({ taskId }) => {
    // Long-running operation
    return { status: 'completed' };
  },
});
```

### MCPToolset

Connect to Model Context Protocol servers:

```typescript
import { MCPToolset } from '@google/adk';

const mcpToolset = new MCPToolset({
  type: 'StreamableHTTPConnectionParams',
  url: 'http://localhost:8788/mcp',
});

const tools = await mcpToolset.getTools();
// Use tools in your agent
```

### BuiltInCodeExecutor

Enable code execution via Gemini's built-in executor (Gemini 2.0+):

```typescript
import { LlmAgent, BuiltInCodeExecutor } from '@google/adk';

const agent = new LlmAgent({
  name: 'code_agent',
  model: 'gemini-2.5-flash',
  codeExecutor: new BuiltInCodeExecutor(),
});

## LlmAgent Configuration

```typescript
import { LlmAgent } from '@google/adk';
import { GenerateContentConfig } from '@google/genai';

const agent = new LlmAgent({
  // Required
  name: 'agent_name',           // Unique identifier
  model: 'gemini-2.5-flash',    // LLM model

  // Recommended
  description: 'What this agent does',  // For multi-agent routing
  instruction: 'Detailed behavior instructions',

  // Optional
  tools: [tool1, tool2],        // Function tools
  subAgents: [agentA, agentB],  // Child agents
  outputKey: 'result',          // Save output to state
  outputSchema: schema,         // Enforce JSON output structure

  // LLM tuning
  generateContentConfig: {
    temperature: 0.2,
    maxOutputTokens: 250,
  },
});
```

## State & Dynamic Instructions

Use `{var}` syntax in instructions to inject state values:

```typescript
const agent = new LlmAgent({
  name: 'greeter',
  model: 'gemini-2.5-flash',
  instruction: 'Greet the user. Their name is {userName}.',
  outputKey: 'greeting',  // Saves response to state['greeting']
});
```

## Structured Output

```typescript
import { Schema, Type } from '@google/genai';

const outputSchema: Schema = {
  type: Type.OBJECT,
  properties: {
    capital: { type: Type.STRING, description: 'Capital city' },
  },
  required: ['capital'],
};

const agent = new LlmAgent({
  name: 'capital_agent',
  model: 'gemini-2.5-flash',
  instruction: 'Return the capital as JSON: {"capital": "city_name"}',
  outputSchema,
  outputKey: 'found_capital',
});
```

## Running Agents

### Command Line
```bash
npx @google/adk-devtools run agent.ts
```

### Web UI (Development Only)
```bash
npx @google/adk-devtools web
# Opens http://localhost:8000
```

## Multi-Agent Systems

See `#adk-multi-agent-systems` for detailed patterns.

```typescript
import { LlmAgent, SequentialAgent, ParallelAgent } from '@google/adk';

// Sequential pipeline
const pipeline = new SequentialAgent({
  name: 'pipeline',
  subAgents: [
    new LlmAgent({ name: 'step1', outputKey: 'data' }),
    new LlmAgent({ name: 'step2', instruction: 'Process {data}' }),
  ],
});

// Parallel execution
const parallel = new ParallelAgent({
  name: 'gatherer',
  subAgents: [fetchWeather, fetchNews],
});

// Coordinator with delegation
const coordinator = new LlmAgent({
  name: 'coordinator',
  model: 'gemini-2.5-flash',
  instruction: 'Route billing to Billing agent, support to Support agent.',
  subAgents: [billingAgent, supportAgent],
});
```

## Best Practices

1. **Clear descriptions** - Essential for multi-agent routing
2. **Specific instructions** - Guide tool usage explicitly
3. **Use outputKey** - Pass data between agents via state
4. **Zod schemas** - Type-safe tool parameters
5. **Error handling** - Tools should return error info gracefully

## Resources

- [Documentation](https://google.github.io/adk-docs)
- [GitHub](https://github.com/google/adk-js)
- [Samples](https://github.com/google/adk-samples)
- [npm](https://www.npmjs.com/package/@google/adk)

## Related Steering Documents

- `#adk-multi-agent-systems` - Multi-agent patterns and workflows
- `#functiongemma-guide` - Local function calling with FunctionGemma
