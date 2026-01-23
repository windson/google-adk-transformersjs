/**
 * Multi-Agent System with ADK - Coordinator/Dispatcher Pattern
 * 
 * Architecture (following ADK Multi-Agent best practices):
 * 
 * ┌─────────────────────────────────────────────────────────────────-┐
 * │              Coordinator Agent (Entry Point)                     │
 * │  - Receives ALL user requests                                    │
 * │  - Handles general conversation directly (greetings, chitchat)   │
 * │  - Delegates tool-based tasks to ToolRouter via subAgents        │
 * └─────────────────────────────────────────────────────────────────-┘
 *                              │
 *              ┌───────────────┴───────────────┐
 *              ▼                               ▼
 * ┌─────────────────────────┐     ┌─────────────────────────┐
 * │   Greeter Agent         │     │   ToolRouter Agent      │
 * │   (Handles greetings,   │     │   (FunctionGemma)       │
 * │    general chat)        │     │   - calculator          │
 * └─────────────────────────┘     │   - get_time            │
 *                                 │   - get_weather         │
 *                                 └─────────────────────────┘
 * 
 * Pattern: Coordinator/Dispatcher with LLM-Driven Delegation
 * - Coordinator uses subAgents for delegation
 * - Each specialist agent has a distinct description for routing
 * - Scalable: Add new tools to ToolRouter or new specialist agents as needed
 * 
 * Models:
 * - Coordinator: FunctionGemma (routes to appropriate agent)
 * - Greeter: FunctionGemma (handles conversation)
 * - ToolRouter: FunctionGemma (specialized for tool calling)
 */

import { LlmAgent, FunctionTool } from '@google/adk';
import { z } from 'zod';
import { TransformersLlm } from './transformers-llm';

// ============================================================================
// Model Configuration
// ============================================================================

// FunctionGemma for all agents - specialized for function calling
const MODEL_ID = 'onnx-community/functiongemma-270m-it-ONNX';

// ============================================================================
// Tool Functions
// ============================================================================

/**
 * Calculator - Evaluates mathematical expressions
 */
function evaluateExpression(expression: string): { status: string; expression: string; result?: string; error?: string } {
  console.log(`🔧 [Calculator] Evaluating: ${expression}`);
  try {
    const sanitized = expression.replace(/[^0-9+\-*/%().^\s]/g, '');
    const jsExpression = sanitized.replace(/\^/g, '**');
    const result = new Function(`"use strict"; return (${jsExpression})`)();
    
    if (typeof result !== 'number' || !isFinite(result)) {
      return { status: 'error', expression, error: 'Invalid mathematical result' };
    }
    
    const formatted = Number.isInteger(result) 
      ? result.toLocaleString() 
      : result.toFixed(6).replace(/\.?0+$/, '');
    
    console.log(`✅ [Calculator] Result: ${formatted}`);
    return { status: 'success', expression, result: formatted };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Calculation failed';
    console.log(`❌ [Calculator] Error: ${errorMsg}`);
    return { status: 'error', expression, error: errorMsg };
  }
}

/**
 * TimeKeeper - Returns current date and time
 */
function getCurrentTime(timezone?: string): { status: string; datetime?: string; timestamp?: string; error?: string } {
  console.log(`🔧 [TimeKeeper] Getting time${timezone ? ` for ${timezone}` : ''}`);
  try {
    const now = new Date();
    const options: Intl.DateTimeFormatOptions = {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      timeZoneName: 'short',
    };
    
    if (timezone) {
      options.timeZone = timezone;
    }
    
    const formatted = now.toLocaleString('en-US', options);
    console.log(`✅ [TimeKeeper] Result: ${formatted}`);
    return { status: 'success', datetime: formatted, timestamp: now.toISOString() };
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Failed to get time';
    console.log(`❌ [TimeKeeper] Error: ${errorMsg}`);
    return { status: 'error', error: errorMsg };
  }
}

/**
 * WMO Weather interpretation codes
 */
function getWeatherDescription(code: number): string {
  const weatherCodes: Record<number, string> = {
    0: 'clear sky', 1: 'mainly clear', 2: 'partly cloudy', 3: 'overcast',
    45: 'foggy', 48: 'depositing rime fog',
    51: 'light drizzle', 53: 'moderate drizzle', 55: 'dense drizzle',
    56: 'light freezing drizzle', 57: 'dense freezing drizzle',
    61: 'slight rain', 63: 'moderate rain', 65: 'heavy rain',
    66: 'light freezing rain', 67: 'heavy freezing rain',
    71: 'slight snow', 73: 'moderate snow', 75: 'heavy snow', 77: 'snow grains',
    80: 'slight rain showers', 81: 'moderate rain showers', 82: 'violent rain showers',
    85: 'slight snow showers', 86: 'heavy snow showers',
    95: 'thunderstorm', 96: 'thunderstorm with slight hail', 99: 'thunderstorm with heavy hail',
  };
  return weatherCodes[code] || 'unknown';
}

/**
 * WeatherAgent - Returns real weather information using Open-Meteo API
 */
async function getWeather(location: string): Promise<{ 
  status: string; location: string; temperature?: string; feelsLike?: string;
  condition?: string; humidity?: string; windSpeed?: string; error?: string;
}> {
  console.log(`🔧 [WeatherAgent] Getting weather for: ${location}`);
  
  try {
    // Step 1: Geocode the location
    const geoUrl = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(location)}&count=1&language=en&format=json`;
    const geoResponse = await fetch(geoUrl);
    if (!geoResponse.ok) throw new Error(`Geocoding failed: ${geoResponse.statusText}`);
    
    const geoData = await geoResponse.json();
    if (!geoData.results || geoData.results.length === 0) {
      return { status: 'error', location, error: `Location "${location}" not found` };
    }
    
    const { latitude, longitude, name, country } = geoData.results[0];
    const fullLocation = country ? `${name}, ${country}` : name;
    
    // Step 2: Get current weather data
    const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${latitude}&longitude=${longitude}&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m&timezone=auto`;
    const weatherResponse = await fetch(weatherUrl);
    if (!weatherResponse.ok) throw new Error(`Weather API failed: ${weatherResponse.statusText}`);
    
    const weatherData = await weatherResponse.json();
    const current = weatherData.current;
    
    const result = {
      status: 'success',
      location: fullLocation,
      temperature: `${Math.round(current.temperature_2m)}°C`,
      feelsLike: `${Math.round(current.apparent_temperature)}°C`,
      condition: getWeatherDescription(current.weather_code),
      humidity: `${current.relative_humidity_2m}%`,
      windSpeed: `${Math.round(current.wind_speed_10m)} km/h`,
    };
    
    console.log(`✅ [WeatherAgent] Result: ${result.temperature}, ${result.condition}`);
    return result;
  } catch (error) {
    const errorMsg = error instanceof Error ? error.message : 'Failed to get weather';
    console.log(`❌ [WeatherAgent] Error: ${errorMsg}`);
    return { status: 'error', location, error: errorMsg };
  }
}

// ============================================================================
// Tool Definitions
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const calculatorTool = new FunctionTool<any>({
  name: 'calculator',
  description: 'Performs mathematical calculations. Use for math, arithmetic, percentages, equations.',
  parameters: z.object({
    expression: z.string().describe('Mathematical expression to evaluate (e.g., "5+3", "15*200/100")'),
  }),
  execute: (input: unknown) => {
    const { expression } = input as { expression: string };
    return evaluateExpression(expression);
  },
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const timeTool = new FunctionTool<any>({
  name: 'get_time',
  description: 'Gets the current date and time. Use when asked about time, date, day, or "what time is it".',
  parameters: z.object({
    timezone: z.string().optional().describe('Optional timezone (e.g., "America/New_York")'),
  }),
  execute: (input: unknown) => {
    const { timezone } = input as { timezone?: string };
    return getCurrentTime(timezone);
  },
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const weatherTool = new FunctionTool<any>({
  name: 'get_weather',
  description: 'Gets current weather for a location. Use when asked about weather, temperature, or climate.',
  parameters: z.object({
    location: z.string().describe('City or location name (e.g., "Tokyo", "New York")'),
  }),
  execute: async (input: unknown) => {
    const { location } = input as { location: string };
    return await getWeather(location);
  },
});



// ============================================================================
// Tool Schemas for FunctionGemma
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const toolSchemas: any[] = [
  {
    type: 'function',
    function: {
      name: 'calculator',
      description: 'Performs mathematical calculations. Use for math, arithmetic, percentages.',
      parameters: {
        type: 'object',
        properties: {
          expression: { type: 'string', description: 'Math expression like "5+3" or "15*200/100"' },
        },
        required: ['expression'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_time',
      description: 'Gets current date and time. Use when asked about time or date.',
      parameters: {
        type: 'object',
        properties: {
          timezone: { type: 'string', description: 'Timezone like "America/New_York"' },
        },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Gets weather for a location. Use when asked about weather.',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'City name like "Tokyo"' },
        },
        required: ['location'],
      },
    },
  },
];

// ============================================================================
// Coordinator Agent - Main Entry Point
// ============================================================================

const coordinatorLlm = new TransformersLlm({
  model: MODEL_ID,
  maxNewTokens: 128,
  temperature: 0.1,
});
coordinatorLlm.setToolSchemas(toolSchemas);

/**
 * Coordinator Agent - The main entry point for all requests
 * 
 * This agent:
 * 1. Receives all user requests
 * 2. Routes to appropriate tool based on request type
 * 3. Handles greetings via greet tool
 * 4. Handles calculations via calculator tool
 * 5. Handles time queries via get_time tool
 * 6. Handles weather queries via get_weather tool
 * 
 * Scalability: To add new capabilities, simply:
 * 1. Create a new FunctionTool
 * 2. Add its schema to toolSchemas
 * 3. Add it to the tools array
 */
export const coordinatorAgent = new LlmAgent({
  name: 'Coordinator',
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  model: coordinatorLlm as any,
  description: 'Main coordinator that routes requests to appropriate tools',
  instruction: `You are a friendly AI assistant with access to tools.

WHEN TO USE TOOLS:
- "temperature", "weather", "cold", "hot", "forecast" + location → get_weather(location="<city>")
- "time", "date", "what day" → get_time()
- math, numbers, calculations → calculator(expression="<expr>")

WHEN TO JUST RESPOND (no tool):
- Greetings: "hi", "hello", "how are you"
- Thanks, goodbyes, general chat

Examples:
- "What's the weather in Tokyo?" → get_weather(location="Tokyo")
- "Temperature in Anantapur?" → get_weather(location="Anantapur")
- "I'm cold, what's it like in NYC?" → get_weather(location="NYC")
- "What time is it?" → get_time()
- "5 + 3" → calculator(expression="5+3")
- "Hi!" → Just say hello back

Be friendly and helpful!`,
  tools: [calculatorTool, timeTool, weatherTool],
});

// ============================================================================
// Exports
// ============================================================================

export { 
  coordinatorLlm,
  calculatorTool,
  timeTool,
  weatherTool,
  evaluateExpression,
  getCurrentTime,
  getWeather,
  toolSchemas,
};

export default coordinatorAgent;
