/**
 * ADK + Transformers.js Proof of Concept
 * 
 * This demonstrates running a Google ADK agent with a local Transformers.js
 * model instead of cloud-based LLMs.
 * 
 * Usage:
 *   npm start                          # Interactive mode
 *   npm start -- "What is 15% of 200?" # Single query
 */

import { InMemoryRunner } from '@google/adk';
import { localAgent } from './agent.js';
import * as readline from 'readline';

// ============================================================================
// Runner Setup
// ============================================================================

const APP_NAME = 'LocalAgentApp';
const USER_ID = 'local_user';
const SESSION_ID = 'session_1';

const runner = new InMemoryRunner({
  agent: localAgent,
  appName: APP_NAME,
});

// ============================================================================
// Chat Function
// ============================================================================

async function chat(message: string): Promise<string> {
  console.log('\n[User]:', message);
  console.log('[Agent]: Thinking...');
  
  let response = '';
  let toolCalled = false;
  
  try {
    // Run the agent using runAsync
    for await (const event of runner.runAsync({
      userId: USER_ID,
      sessionId: SESSION_ID,
      newMessage: {
        role: 'user',
        parts: [{ text: message }],
      },
    })) {
      // Log full event for debugging (optional - comment out for cleaner output)
      console.log('[Event]:', JSON.stringify({
        author: event.author,
        invocationId: event.invocationId?.slice(0, 8),
        hasParts: !!event.content?.parts?.length,
        partTypes: event.content?.parts?.map(p => {
          if ('text' in p) return 'text';
          if ('functionCall' in p) return 'functionCall';
          if ('functionResponse' in p) return 'functionResponse';
          return 'unknown';
        }),
      }));
      
      // Process events
      if (event.content?.parts) {
        for (const part of event.content.parts) {
          if ('text' in part && part.text) {
            response += part.text;
          }
          if ('functionCall' in part && part.functionCall) {
            toolCalled = true;
            console.log(`\n🔧 [Tool Call]: ${part.functionCall.name}`);
            console.log(`   Args: ${JSON.stringify(part.functionCall.args, null, 2)}`);
          }
          if ('functionResponse' in part && part.functionResponse) {
            console.log(`\n✅ [Tool Response]: ${part.functionResponse.name}`);
            console.log(`   Result: ${JSON.stringify(part.functionResponse.response, null, 2)}`);
          }
        }
      }
    }
  } catch (error) {
    console.error('[Error]:', error);
    response = `Error: ${error instanceof Error ? error.message : 'Unknown error'}`;
  }
  
  console.log('\n[Agent]:', response || '(no response)');
  console.log(`[Tools Used]: ${toolCalled ? 'Yes' : 'No'}`);
  return response;
}

// ============================================================================
// Interactive Mode
// ============================================================================

async function interactiveMode(): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('╔════════════════════════════════════════════════════════════════╗');
  console.log('║     ADK + Transformers.js Local Agent                          ║');
  console.log('║     Type your message and press Enter. Type "exit" to quit.    ║');
  console.log('╚════════════════════════════════════════════════════════════════╝');
  console.log('');
  console.log('Available tools: calculator, get_current_time, get_weather');
  console.log('Example queries:');
  console.log('  - "What is 15% of 200?"');
  console.log('  - "What time is it?"');
  console.log('  - "What\'s the weather in Tokyo?"');
  console.log('');

  // Create session
  await runner.sessionService.createSession({
    appName: APP_NAME,
    userId: USER_ID,
    sessionId: SESSION_ID,
  });

  const askQuestion = (): void => {
    rl.question('You: ', async (input) => {
      const trimmed = input.trim();
      
      if (trimmed.toLowerCase() === 'exit' || trimmed.toLowerCase() === 'quit') {
        console.log('Goodbye!');
        rl.close();
        process.exit(0);
      }
      
      if (trimmed) {
        await chat(trimmed);
      }
      
      askQuestion();
    });
  };

  askQuestion();
}

// ============================================================================
// Main
// ============================================================================

async function main(): Promise<void> {
  // Check for command line argument
  const args = process.argv.slice(2);
  
  if (args.length > 0) {
    // Single query mode - create session first
    await runner.sessionService.createSession({
      appName: APP_NAME,
      userId: USER_ID,
      sessionId: SESSION_ID,
    });
    
    const query = args.join(' ');
    await chat(query);
    
    // Force immediate exit to avoid ONNX Runtime cleanup crash
    // The mutex error is a known issue with ONNX Runtime in Node.js
    process.kill(process.pid, 'SIGKILL');
  } else {
    // Interactive mode
    await interactiveMode();
  }
}

main().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
