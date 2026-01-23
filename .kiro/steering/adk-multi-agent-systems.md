# ADK Multi-Agent Systems Guide

This steering document provides guidance for building multi-agent systems using Google's Agent Development Kit (ADK) for TypeScript.

## Core Primitives

### Agent Hierarchy

- Use `subAgents` parameter to establish parent-child relationships
- Single parent rule: an agent can only have one parent
- Navigate with `agent.parentAgent` or `agent.findAgent(name)`

```typescript
import { LlmAgent } from '@google/adk';

const greeter = new LlmAgent({ name: 'Greeter', model: 'gemini-2.5-flash' });
const taskDoer = new LlmAgent({ name: 'TaskExecutor', model: 'gemini-2.5-flash' });

const coordinator = new LlmAgent({
  name: 'Coordinator',
  model: 'gemini-2.5-flash',
  description: 'I coordinate greetings and tasks.',
  subAgents: [greeter, taskDoer],
});
```

### Workflow Agents

| Agent | Behavior | Use Case |
|-------|----------|----------|
| `SequentialAgent` | Executes sub-agents in order | Pipelines, multi-step processes |
| `ParallelAgent` | Executes sub-agents concurrently | Independent parallel tasks |
| `LoopAgent` | Repeats until condition or max iterations | Iterative refinement |

### Communication Mechanisms

1. **Shared Session State** - Read/write via `context.state` and `outputKey`
2. **LLM-Driven Delegation** - Dynamic routing via `transfer_to_agent()` 
3. **Explicit Invocation** - Wrap agents as tools using `AgentTool`

## Common Patterns

### Coordinator/Dispatcher

Central LlmAgent routes requests to specialist sub-agents based on intent.

```typescript
import { LlmAgent } from '@google/adk';

const billingAgent = new LlmAgent({
  name: 'Billing',
  model: 'gemini-2.5-flash',
  description: 'Handles billing inquiries.',
});

const supportAgent = new LlmAgent({
  name: 'Support',
  model: 'gemini-2.5-flash',
  description: 'Handles technical support requests.',
});

const coordinator = new LlmAgent({
  name: 'HelpDeskCoordinator',
  model: 'gemini-2.5-flash',
  instruction: 'Route billing issues to Billing agent, technical problems to Support agent.',
  description: 'Main help desk router.',
  subAgents: [billingAgent, supportAgent],
});
```

### Sequential Pipeline

```typescript
import { SequentialAgent, LlmAgent } from '@google/adk';

const step1 = new LlmAgent({
  name: 'Step1_Fetch',
  model: 'gemini-2.5-flash',
  outputKey: 'data',  // Saves output to state['data']
});

const step2 = new LlmAgent({
  name: 'Step2_Process',
  model: 'gemini-2.5-flash',
  instruction: 'Process data from {data}.',  // Reads state['data']
});

const pipeline = new SequentialAgent({
  name: 'MyPipeline',
  subAgents: [step1, step2],
});
```

### Parallel Fan-Out/Gather

```typescript
import { SequentialAgent, ParallelAgent, LlmAgent } from '@google/adk';

const fetchWeather = new LlmAgent({
  name: 'WeatherFetcher',
  model: 'gemini-2.5-flash',
  outputKey: 'weather',
});

const fetchNews = new LlmAgent({
  name: 'NewsFetcher',
  model: 'gemini-2.5-flash',
  outputKey: 'news',
});

const gatherer = new ParallelAgent({
  name: 'InfoGatherer',
  subAgents: [fetchWeather, fetchNews],
});

const synthesizer = new LlmAgent({
  name: 'Synthesizer',
  model: 'gemini-2.5-flash',
  instruction: 'Combine results from {weather} and {news}.',
});

const workflow = new SequentialAgent({
  name: 'FetchAndSynthesize',
  subAgents: [gatherer, synthesizer],
});
```

### Generator-Critic

Generator produces output, Critic reviews it - both in a SequentialAgent.

```typescript
import { SequentialAgent, LlmAgent } from '@google/adk';

const generator = new LlmAgent({
  name: 'DraftWriter',
  model: 'gemini-2.5-flash',
  instruction: 'Write a short paragraph about subject X.',
  outputKey: 'draft_text',
});

const reviewer = new LlmAgent({
  name: 'FactChecker',
  model: 'gemini-2.5-flash',
  instruction: 'Review {draft_text} for accuracy. Output "valid" or "invalid" with reasons.',
  outputKey: 'review_status',
});

const reviewPipeline = new SequentialAgent({
  name: 'WriteAndReview',
  subAgents: [generator, reviewer],
});
```

### Iterative Refinement

LoopAgent with checker that sets `escalate: true` when quality threshold met.

```typescript
import { LoopAgent, LlmAgent } from '@google/adk';

const codeRefiner = new LlmAgent({
  name: 'CodeRefiner',
  model: 'gemini-2.5-flash',
  instruction: 'Refine code based on {requirements}. Save to state.',
  outputKey: 'current_code',
});

const qualityChecker = new LlmAgent({
  name: 'QualityChecker',
  model: 'gemini-2.5-flash',
  instruction: 'Evaluate {current_code}. Output "pass" or "fail".',
  outputKey: 'quality_status',
});

const refinementLoop = new LoopAgent({
  name: 'CodeRefinementLoop',
  maxIterations: 5,
  subAgents: [codeRefiner, qualityChecker],
});
```

### Agent as Tool (AgentTool)

```typescript
import { LlmAgent, AgentTool } from '@google/adk';

const webSearcher = new LlmAgent({
  name: 'WebSearch',
  model: 'gemini-2.5-flash',
  description: 'Performs web searches for facts.',
});

const summarizer = new LlmAgent({
  name: 'Summarizer',
  model: 'gemini-2.5-flash',
  description: 'Summarizes text.',
});

const researchAssistant = new LlmAgent({
  name: 'ResearchAssistant',
  model: 'gemini-2.5-flash',
  description: 'Finds and summarizes information.',
  tools: [new AgentTool({ agent: webSearcher }), new AgentTool({ agent: summarizer })],
});
```

## Best Practices

- Give sub-agents clear `description` fields for LLM-driven delegation
- Use `outputKey` to save agent outputs to state for downstream agents
- Use distinct state keys in ParallelAgent to avoid race conditions
- Set `maxIterations` on LoopAgent to prevent infinite loops
- Use `escalate: true` in EventActions to break out of loops early
- Provide specific `instruction` text explaining when to use each tool

## Related Steering Documents

- `#adk-typescript-guide` - Core ADK TypeScript setup and usage
