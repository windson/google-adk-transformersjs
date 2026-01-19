/**
 * ADK Multi-Agent System with Transformers.js backend
 * 
 * Architecture:
 * 1. FunctionGemma (270M) - Single-shot tool calling (not designed for multi-turn)
 * 2. SmolLM2 (135M) - Natural language response generation
 * 
 * FunctionGemma outputs: <start_function_call>call:name{args}<end_function_call>
 * We intercept this, execute the tool, then pass result to SmolLM2 for response.
 */

import { LlmAgent, FunctionTool } from '@google/adk';
import { z } from 'zod';
import { TransformersLlm } from './transformers-llm.js';

// ============================================================================
// Tool Schemas for FunctionGemma
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const toolSchemas: any[] = [
  {
    type: 'function',
    function: {
      name: 'calculator',
      description: 'Performs mathematical calculations. Use this for any math operations like addition, subtraction, multiplication, division, percentages.',
      parameters: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'The mathematical expression to evaluate, e.g., "5 + 3" or "15 * 200 / 100"',
          },
        },
        required: ['expression'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_current_time',
      description: 'Returns the current date and time. Use this when user asks about time, date, or what day it is.',
      parameters: {
        type: 'object',
        properties: {
          timezone: {
            type: 'string',
            description: 'Timezone like "America/New_York" or "UTC". Optional.',
          },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Gets the current weather for a location. Use this when user asks about weather, temperature, or climate.',
      parameters: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
            description: 'The city or location to get weather for',
          },
        },
        required: ['location'],
      },
    },
  },
];

// ============================================================================
// Tools (ADK FunctionTool implementations)
// ============================================================================

const calculatorTool = new FunctionTool({
  name: 'calculator',
  description: 'Performs mathematical calculations.',
  parameters: z.object({
    expression: z.string().describe('The mathematical expression to evaluate'),
  }),
  execute: ({ expression }: { expression: string }) => {
    console.log(`\n🔧 [Calculator] Evaluating: ${expression}`);
    try {
      const sanitized = expression.replace(/[^0-9+\-*/%().^\s]/g, '');
      const jsExpression = sanitized.replace(/\^/g, '**');
      const result = new Function(`"use strict"; return (${jsExpression})`)();
      
      if (typeof result !== 'number' || !isFinite(result)) {
        return { status: 'error', error: 'Invalid result' };
      }
      
      const formatted = Number.isInteger(result) 
        ? result.toLocaleString() 
        : result.toFixed(6).replace(/\.?0+$/, '');
      
      console.log(`✅ [Calculator] Result: ${formatted}`);
      return { status: 'success', expression, result: formatted };
    } catch (error) {
      return { status: 'error', error: error instanceof Error ? error.message : 'Calculation failed' };
    }
  },
});

const getCurrentTimeTool = new FunctionTool({
  name: 'get_current_time',
  description: 'Returns the current date and time.',
  parameters: z.object({
    timezone: z.string().optional().describe('Timezone like "America/New_York" or "UTC".'),
  }),
  execute: ({ timezone }: { timezone?: string }) => {
    console.log(`\n🔧 [Time] Getting current time${timezone ? ` for ${timezone}` : ''}`);
    try {
      const now = new Date();
      const options: Intl.DateTimeFormatOptions = {
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit', timeZoneName: 'short',
      };
      if (timezone) options.timeZone = timezone;
      
      const formatted = now.toLocaleString('en-US', options);
      console.log(`✅ [Time] Result: ${formatted}`);
      return { status: 'success', datetime: formatted, timestamp: now.toISOString() };
    } catch (error) {
      return { status: 'error', error: error instanceof Error ? error.message : 'Failed to get time' };
    }
  },
});

const getWeatherTool = new FunctionTool({
  name: 'get_weather',
  description: 'Gets the current weather for a location (mock).',
  parameters: z.object({
    location: z.string().describe('The city or location'),
  }),
  execute: ({ location }: { location: string }) => {
    console.log(`\n🔧 [Weather] Getting weather for: ${location}`);
    const conditions = ['sunny', 'cloudy', 'rainy', 'partly cloudy'];
    const condition = conditions[Math.floor(Math.random() * conditions.length)];
    const temp = Math.floor(Math.random() * 30) + 10;
    console.log(`✅ [Weather] Result: ${temp}°C, ${condition}`);
    return { status: 'success', location, temperature: `${temp}°C`, condition };
  },
});

// ============================================================================
// Tool Caller Agent (FunctionGemma) - Single tool call only
// ============================================================================

const toolCallerLlm = new TransformersLlm({
  model: 'onnx-community/functiongemma-270m-it-ONNX',
  maxNewTokens: 64, // Short - just need the function call
  temperature: 0.1, // Deterministic
});
toolCallerLlm.setToolSchemas(toolSchemas);

// Track tool calls to limit FunctionGemma to single-shot
let toolCallCount = 0;

/**
 * ToolCaller Agent using FunctionGemma
 * 
 * This agent is configured to call tools. FunctionGemma will output:
 * <start_function_call>call:function_name{args}<end_function_call>
 * 
 * We use afterToolCallback to stop after the first tool execution.
 */
const toolCallerAgent = new LlmAgent({
  name: 'ToolCaller',
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model: toolCallerLlm as any,
  description: 'Calls appropriate tools based on user request',
  instruction: `You are a model that can do function calling with the following functions.
Analyze the user request and call the appropriate tool.
Available tools: calculator, get_current_time, get_weather.
Call exactly ONE tool that best matches the user's request.`,
  tools: [calculatorTool, getCurrentTimeTool, getWeatherTool],
  // Use afterToolCallback to track and potentially limit tool calls
  afterToolCallback: ({ tool, args, response }) => {
    toolCallCount++;
    console.log(`\n📊 [ToolCaller] Tool call #${toolCallCount}: ${tool.name}`);
    console.log(`   Args: ${JSON.stringify(args)}`);
    console.log(`   Response: ${JSON.stringify(response)}`);
    // Return the response as-is (don't modify it)
    return response;
  },
});

// ============================================================================
// Response Generator Agent (SmolLM2)
// ============================================================================

const responseGeneratorLlm = new TransformersLlm({
  model: 'HuggingFaceTB/SmolLM2-135M-Instruct',
  maxNewTokens: 128,
  temperature: 0.7,
});

/**
 * ResponseGenerator Agent using SmolLM2
 * 
 * This agent generates natural language responses.
 * It receives the tool result via session state and formats a user-friendly response.
 */
const responseGeneratorAgent = new LlmAgent({
  name: 'ResponseGenerator',
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model: responseGeneratorLlm as any,
  description: 'Generates natural language responses from tool results',
  instruction: `You are a helpful assistant. Provide a clear, friendly response to the user.
Be concise and helpful. Answer directly based on the information available.`,
});

// ============================================================================
// Export - Use ToolCaller as the main agent for now
// Since FunctionGemma loops, we'll handle the response generation manually
// ============================================================================

export const localAgent = toolCallerAgent;

export { toolCallerAgent, responseGeneratorAgent, toolCallerLlm, responseGeneratorLlm };

export default localAgent;
