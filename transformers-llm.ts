/**
 * Custom BaseLlm implementation for Transformers.js
 * 
 * This class bridges Google ADK with local Transformers.js models,
 * enabling fully local LLM inference without cloud APIs.
 * 
 * Supports multiple model formats:
 * - FunctionGemma: Uses 'developer' role, specialized for function calling
 *   - Uses <start_function_response> as stop sequence (per FunctionGemma docs)
 * - SmolLM2: Uses ChatML format (<|im_start|>role\ncontent<|im_end|>)
 */

import { BaseLlm } from '@google/adk';
import type { LlmRequest, LlmResponse, BaseLlmConnection } from '@google/adk';
import { AutoModelForCausalLM, AutoTokenizer, Tensor } from '@huggingface/transformers';

// Model configuration
const DEFAULT_MODEL = 'onnx-community/functiongemma-270m-it-ONNX';

// Model type detection
type ModelType = 'functiongemma' | 'smollm' | 'generic';

interface TransformersLlmParams {
  model?: string;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  doSample?: boolean;
}

// Tool schema type for FunctionGemma
interface ToolSchema {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: {
      type: 'object';
      properties: Record<string, { type: string; description: string }>;
      required?: string[];
    };
  };
}

type TransformersModel = Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>>;
type TransformersTokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>;

/**
 * TransformersLlm - A custom BaseLlm implementation for local inference
 * 
 * This class allows ADK agents to use Transformers.js models instead of
 * cloud-based LLMs like Gemini or GPT.
 * 
 * FunctionGemma Stop Sequence Handling:
 * Per the FunctionGemma documentation, <start_function_response> is an
 * additional stop sequence. This means the model should stop generating
 * when it outputs this token, as it expects the application to provide
 * the function response.
 */
export class TransformersLlm extends BaseLlm {
  private transformersModel: TransformersModel | null = null;
  private tokenizer: TransformersTokenizer | null = null;
  private modelId: string;
  private modelType: ModelType;
  private maxNewTokens: number;
  private temperature: number;
  private topP: number;
  private doSample: boolean;
  private initPromise: Promise<void> | null = null;
  private toolSchemas: ToolSchema[] = [];
  
  // Stop sequence token IDs (populated during initialization)
  private stopTokenIds: number[] = [];

  // Static property required by ADK's LLM registry
  static override readonly supportedModels: (string | RegExp)[] = [
    /^transformers\/.*/,
    /^local\/.*/,
    'functiongemma',
    'functiongemma-270m',
    'smollm',
    'smollm2',
  ];

  constructor(params: TransformersLlmParams = {}) {
    super({ model: params.model || DEFAULT_MODEL });
    
    this.modelId = params.model || DEFAULT_MODEL;
    this.modelType = this.detectModelType(this.modelId);
    this.maxNewTokens = params.maxNewTokens || 256;
    this.temperature = params.temperature || 0.7;
    this.topP = params.topP || 0.9;
    this.doSample = params.doSample ?? true;
    
    console.log(`[TransformersLlm] Model type detected: ${this.modelType}`);
  }

  private detectModelType(modelId: string): ModelType {
    const lower = modelId.toLowerCase();
    if (lower.includes('functiongemma')) return 'functiongemma';
    if (lower.includes('smollm')) return 'smollm';
    return 'generic';
  }

  setToolSchemas(schemas: ToolSchema[]): void {
    this.toolSchemas = schemas;
  }

  /**
   * Initialize the model and tokenizer, and set up stop sequences
   */
  private async initialize(): Promise<void> {
    if (this.transformersModel && this.tokenizer) return;
    
    if (this.initPromise) {
      await this.initPromise;
      return;
    }

    this.initPromise = (async () => {
      console.log(`[TransformersLlm] Loading model: ${this.modelId}`);
      const startTime = Date.now();
      
      this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
      this.transformersModel = await AutoModelForCausalLM.from_pretrained(this.modelId);
      
      // Set up stop sequences for FunctionGemma
      // Per docs: "<start_function_response> is an additional stop sequence"
      if (this.modelType === 'functiongemma') {
        await this.setupStopSequences();
      }
      
      const loadTime = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`[TransformersLlm] Model loaded in ${loadTime}s`);
    })();

    await this.initPromise;
  }

  /**
   * Set up stop sequence token IDs for FunctionGemma
   * 
   * FunctionGemma should stop generating when it outputs:
   * - <start_function_response> (expects app to provide function result)
   * - <end_of_turn> (normal end of turn)
   */
  private async setupStopSequences(): Promise<void> {
    if (!this.tokenizer) return;
    
    const stopSequences = [
      '<start_function_response>',
      '<end_of_turn>',
    ];
    
    this.stopTokenIds = [];
    
    for (const seq of stopSequences) {
      try {
        // Encode the stop sequence to get its token ID(s)
        // The encode method returns either number[] or a Tensor
        const encoded = this.tokenizer.encode(seq, { add_special_tokens: false });
        
        let tokenData: number[] = [];
        if (Array.isArray(encoded)) {
          tokenData = encoded;
        } else if (encoded && typeof encoded === 'object' && 'data' in encoded) {
          // It's a Tensor
          tokenData = Array.from((encoded as Tensor).data as BigInt64Array).map(Number);
        }
        
        if (tokenData.length > 0) {
          // Use the first token of the sequence as stop token
          const tokenId = Number(tokenData[0]);
          this.stopTokenIds.push(tokenId);
          console.log(`[TransformersLlm] Stop sequence "${seq}" -> token ID ${tokenId}`);
        }
      } catch (e) {
        console.warn(`[TransformersLlm] Could not encode stop sequence "${seq}":`, e);
      }
    }
    
    console.log(`[TransformersLlm] Configured ${this.stopTokenIds.length} stop token(s)`);
  }

  /**
   * Convert ADK LlmRequest to chat format based on model type
   */
  private formatMessages(llmRequest: LlmRequest): Array<{ role: string; content: string }> {
    const messages: Array<{ role: string; content: string }> = [];
    
    const systemInstruction = llmRequest.config?.systemInstruction;
    if (systemInstruction) {
      let instructionText: string;
      if (typeof systemInstruction === 'string') {
        instructionText = systemInstruction;
      } else if (systemInstruction && 'parts' in systemInstruction && systemInstruction.parts) {
        instructionText = systemInstruction.parts
          .map((p) => ('text' in p ? p.text : '') || '')
          .join('') || '';
      } else {
        instructionText = '';
      }
      
      if (instructionText) {
        const systemRole = this.modelType === 'functiongemma' ? 'developer' : 'system';
        messages.push({ role: systemRole, content: instructionText });
      }
    }

    for (const content of llmRequest.contents) {
      const role = content.role === 'model' ? 'assistant' : (content.role || 'user');
      const parts = content.parts || [];
      const text = parts
        .map((part) => {
          if ('text' in part && part.text) return part.text;
          if ('functionCall' in part && part.functionCall) {
            if (this.modelType === 'functiongemma') {
              const args = Object.entries(part.functionCall.args || {})
                .map(([k, v]) => `${k}:<escape>${v}<escape>`)
                .join(',');
              return `<start_function_call>call:${part.functionCall.name}{${args}}<end_function_call>`;
            } else {
              return `Tool call: ${part.functionCall.name}(${JSON.stringify(part.functionCall.args)})`;
            }
          }
          if ('functionResponse' in part && part.functionResponse) {
            if (this.modelType === 'functiongemma') {
              return `<start_function_response>${JSON.stringify(part.functionResponse.response)}<end_function_response>`;
            } else {
              return `Tool result: ${JSON.stringify(part.functionResponse.response)}`;
            }
          }
          return '';
        })
        .filter(Boolean)
        .join('\n') || '';
      
      if (text) {
        messages.push({ role, content: text });
      }
    }

    return messages;
  }

  /**
   * Parse FunctionGemma function call output
   */
  private parseFunctionCall(text: string): { name: string; args: Record<string, unknown> } | null {
    const match = text.match(/<start_function_call>call:(\w+)\{([^}]*)\}<end_function_call>/);
    if (!match) return null;

    const name = match[1];
    const argsStr = match[2];
    const args: Record<string, unknown> = {};

    const argMatches = argsStr.matchAll(/(\w+):<escape>([^<]*)<escape>/g);
    for (const argMatch of argMatches) {
      args[argMatch[1]] = argMatch[2];
    }

    return { name, args };
  }

  /**
   * Check if conversation already has a function response
   * Used as fallback if stop sequences don't work
   */
  private hasFunctionResponse(llmRequest: LlmRequest): { hasResponse: boolean; lastResponse?: unknown } {
    for (const content of llmRequest.contents) {
      for (const part of content.parts || []) {
        if ('functionResponse' in part && part.functionResponse) {
          return { hasResponse: true, lastResponse: part.functionResponse.response };
        }
      }
    }
    return { hasResponse: false };
  }

  /**
   * Main generation method
   * 
   * For FunctionGemma, uses <start_function_response> as a stop sequence
   * per the official documentation.
   */
  override async *generateContentAsync(
    llmRequest: LlmRequest,
    _stream?: boolean
  ): AsyncGenerator<LlmResponse, void> {
    await this.initialize();
    
    if (!this.transformersModel || !this.tokenizer) {
      yield {
        errorCode: 'MODEL_NOT_LOADED',
        errorMessage: 'Failed to load Transformers.js model',
      };
      return;
    }

    // Fallback: If we already have a function response in the conversation,
    // FunctionGemma (base model) doesn't know how to generate text responses.
    // Return a summary instead of letting it loop.
    if (this.modelType === 'functiongemma') {
      const { hasResponse, lastResponse } = this.hasFunctionResponse(llmRequest);
      if (hasResponse) {
        console.log(`[TransformersLlm:functiongemma] Function response in history - returning summary`);
        const resultText = typeof lastResponse === 'object' 
          ? JSON.stringify(lastResponse)
          : String(lastResponse);
        
        yield {
          content: {
            role: 'model',
            parts: [{ text: `Tool result: ${resultText}` }],
          },
          finishReason: 'STOP' as LlmResponse['finishReason'],
        };
        return;
      }
    }

    const messages = this.formatMessages(llmRequest);
    console.log(`[TransformersLlm:${this.modelType}] Generating response...`);
    console.log(`[TransformersLlm:${this.modelType}] Messages:`, JSON.stringify(messages, null, 2));
    
    const startTime = Date.now();

    try {
      const maxTokens = llmRequest.config?.maxOutputTokens || this.maxNewTokens;

      const templateOptions: {
        tools?: ToolSchema[];
        tokenize: boolean;
        add_generation_prompt: boolean;
        return_dict: boolean;
      } = {
        tokenize: true,
        add_generation_prompt: true,
        return_dict: true,
      };
      
      if (this.modelType === 'functiongemma' && this.toolSchemas.length > 0) {
        templateOptions.tools = this.toolSchemas;
      }

      const inputs = this.tokenizer.apply_chat_template(messages, templateOptions) as { 
        input_ids: Tensor; 
        attention_mask: Tensor 
      };

      // Build generation config with stop sequences
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const generateConfig: any = {
        ...inputs,
        max_new_tokens: maxTokens,
      };

      // Add stop token IDs for FunctionGemma
      // This makes the model stop when it generates <start_function_response>
      if (this.modelType === 'functiongemma' && this.stopTokenIds.length > 0) {
        // Get the model's default EOS token ID
        const eosTokenId = this.tokenizer.eos_token_id;
        
        // Combine default EOS with our stop sequences
        const allStopTokens = eosTokenId 
          ? [eosTokenId, ...this.stopTokenIds]
          : this.stopTokenIds;
        
        generateConfig.eos_token_id = allStopTokens;
        console.log(`[TransformersLlm:functiongemma] Using stop tokens: ${allStopTokens.join(', ')}`);
      }

      const generateFn = this.transformersModel.generate.bind(this.transformersModel);
      const output = await generateFn(generateConfig);

      const generationTime = Date.now() - startTime;
      console.log(`[TransformersLlm:${this.modelType}] Generated in ${generationTime}ms`);

      const inputLength = inputs.input_ids.dims[1];
      const outputTensor = output as Tensor;
      const outputData = Array.from(outputTensor.data as BigInt64Array);
      const newTokenIds = outputData.slice(inputLength);
      const generatedText = this.tokenizer.decode(newTokenIds, { skip_special_tokens: false });

      console.log(`[TransformersLlm:${this.modelType}] Raw output:`, generatedText);

      // Parse output based on model type
      let functionCall: { name: string; args: Record<string, unknown> } | null = null;
      let textResponse = generatedText;
      
      if (this.modelType === 'functiongemma') {
        functionCall = this.parseFunctionCall(generatedText);
        if (!functionCall) {
          // Clean up any remaining special tokens
          textResponse = generatedText
            .replace(/<start_function_response>[\s\S]*$/, '') // Remove trailing function response start
            .replace(/<[^>]+>/g, '')
            .trim();
        }
      } else {
        textResponse = generatedText
          .replace(/<\|im_start\|>/g, '')
          .replace(/<\|im_end\|>/g, '')
          .replace(/<\|endoftext\|>/g, '')
          .replace(/^(assistant|user|system)\n?/i, '')
          .trim();
      }

      const response: LlmResponse = {
        content: {
          role: 'model',
          parts: functionCall 
            ? [{
                functionCall: {
                  name: functionCall.name,
                  args: functionCall.args,
                },
              }]
            : [{ text: textResponse }],
        },
        finishReason: 'STOP' as LlmResponse['finishReason'],
      };

      yield response;

    } catch (error) {
      console.error(`[TransformersLlm:${this.modelType}] Generation error:`, error);
      yield {
        errorCode: 'GENERATION_ERROR',
        errorMessage: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  override async connect(_llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    throw new Error('Live connections not supported for TransformersLlm. Use generateContentAsync instead.');
  }
}

export default TransformersLlm;
