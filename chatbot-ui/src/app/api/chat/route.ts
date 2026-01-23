import { NextRequest, NextResponse } from 'next/server';
import { InMemoryRunner } from '@google/adk';
import { coordinatorAgent } from '@/lib/agent';

const APP_NAME = 'MultiAgentChatbot';

// Create runner instance with coordinator agent
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

/**
 * Format tool result into a natural language response
 */
function formatResponse(toolName: string, result: Record<string, unknown>, userMessage: string): string {
  if (result.status === 'error') {
    return `Sorry, I couldn't complete that request: ${result.error || 'Unknown error'}`;
  }

  switch (toolName) {
    case 'greet': {
      return result.response as string;
    }
    
    case 'calculator': {
      const expr = result.expression as string;
      const answer = result.result as string;
      if (userMessage.toLowerCase().includes('%') || userMessage.toLowerCase().includes('percent')) {
        return `${expr.replace(/\*/g, ' × ').replace(/\//g, ' ÷ ')} = ${answer}`;
      }
      return `The answer is ${answer}.`;
    }
    
    case 'get_time': {
      const datetime = result.datetime as string;
      return `It's ${datetime}.`;
    }
    
    case 'get_weather': {
      const location = result.location as string;
      const temp = result.temperature as string;
      const feelsLike = result.feelsLike as string;
      const condition = result.condition as string;
      const humidity = result.humidity as string;
      const windSpeed = result.windSpeed as string;
      
      let response = `The weather in ${location} is ${condition} with a temperature of ${temp}`;
      if (feelsLike && feelsLike !== temp) {
        response += ` (feels like ${feelsLike})`;
      }
      response += `. Humidity is ${humidity}`;
      if (windSpeed) {
        response += ` with winds at ${windSpeed}`;
      }
      response += '.';
      return response;
    }
    
    default:
      return JSON.stringify(result);
  }
}

export async function POST(request: NextRequest) {
  try {
    const { message } = await request.json();
    
    if (!message || typeof message !== 'string') {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    const currentRunner = getRunner();
    
    // Create a fresh session for each request
    const sessionId = `session-${Date.now()}`;
    await currentRunner.sessionService.createSession({
      appName: APP_NAME,
      userId: 'user',
      sessionId,
    });

    const toolCalls: Array<{ name: string; args: unknown; result?: unknown }> = [];
    const agents: string[] = [];
    let response = '';
    
    console.log(`\n${'='.repeat(60)}`);
    console.log(`[Chat] User message: ${message}`);
    console.log(`${'='.repeat(60)}`);

    // Run the coordinator agent
    for await (const event of currentRunner.runAsync({
      userId: 'user',
      sessionId,
      newMessage: {
        role: 'user',
        parts: [{ text: message }],
      },
    })) {
      const author = event.author || 'unknown';
      
      if (author !== 'user' && !agents.includes(author)) {
        agents.push(author);
      }
      
      if (event.content?.parts) {
        for (const part of event.content.parts) {
          if ('functionCall' in part && part.functionCall) {
            const funcName = part.functionCall.name || 'unknown';
            console.log(`[${author}] Tool call: ${funcName}(${JSON.stringify(part.functionCall.args)})`);
            toolCalls.push({
              name: funcName,
              args: part.functionCall.args,
            });
          }
          
          if ('functionResponse' in part && part.functionResponse) {
            const toolName = part.functionResponse.name || 'unknown';
            const toolResult = part.functionResponse.response as Record<string, unknown>;
            console.log(`[${author}] Tool response: ${toolName} -> ${JSON.stringify(toolResult)}`);
            
            const lastCall = toolCalls.find(tc => tc.name === toolName && !tc.result);
            if (lastCall) {
              lastCall.result = toolResult;
            }
            
            response = formatResponse(toolName, toolResult, message);
          }
          
          if ('text' in part && part.text && !response) {
            console.log(`[${author}] Text: ${part.text}`);
            if (!toolCalls.length) {
              response = part.text;
            }
          }
        }
      }
    }

    if (!response) {
      response = "Hello! I'm your AI assistant. I can help you with calculations, tell you the time, or check the weather. What would you like to know?";
    }

    console.log(`[Chat] Final response: ${response}`);
    console.log(`${'='.repeat(60)}\n`);

    return NextResponse.json({
      response,
      toolCalls,
      agents,
    });
  } catch (error) {
    console.error('Chat error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
