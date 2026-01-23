/**
 * Custom BaseLlm implementation for Transformers.js
 * Bridges Google ADK with local Transformers.js models for fully local inference.
 * 
 * Supports:
 * - FunctionGemma: Specialized for function/tool calling
 * - SmolLM2: Natural language response generation
 */

import { BaseLlm } from '@google/adk';
import type { LlmRequest, LlmResponse, BaseLlmConnection } from '@google/adk';

// Lazy import for transformers to avoid SSR issues
let transformersModule: typeof import('@huggingface/transformers') | null = null;

async function getTransformers() {
  if (!transformersModule) {
    transformersModule = await import('@huggingface/transformers');
  }
  return transformersModule;
}

type ModelType = 'functiongemma' | 'smollm' | 'generic';

interface TransformersLlmParams {
  model: string;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  doSample?: boolean;
}

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

type TransformersModel = Awaited<ReturnType<typeof import('@huggingface/transformers').AutoModelForCausalLM.from_pretrained>>;
type TransformersTokenizer = Awaited<ReturnType<typeof import('@huggingface/transformers').AutoTokenizer.from_pretrained>>;
type Tensor = import('@huggingface/transformers').Tensor;

// Model cache to avoid reloading
const modelCache = new Map<string, { model: TransformersModel; tokenizer: TransformersTokenizer }>();

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
  private stopTokenIds: number[] = [];

  static override readonly supportedModels: (string | RegExp)[] = [
    /^onnx-community\/.*/,
    /^HuggingFaceTB\/.*/,
    'functiongemma',
    'smollm',
  ];

  constructor(params: TransformersLlmParams) {
    super({ model: params.model });
    this.modelId = params.model;
    this.modelType = this.detectModelType(this.modelId);
    this.maxNewTokens = params.maxNewTokens || 256;
    this.temperature = params.temperature || 0.7;
    this.topP = params.topP || 0.9;
    this.doSample = params.doSample ?? true;
    console.log(`[TransformersLlm] Created for model: ${this.modelId} (type: ${this.modelType})`);
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

  getModelType(): ModelType {
    return this.modelType;
  }

  private async initialize(): Promise<void> {
    if (this.transformersModel && this.tokenizer) return;
    if (this.initPromise) {
      await this.initPromise;
      return;
    }

    this.initPromise = (async () => {
      // Check cache first
      const cached = modelCache.get(this.modelId);
      if (cached) {
        console.log(`[TransformersLlm] Using cached model: ${this.modelId}`);
        this.transformersModel = cached.model;
        this.tokenizer = cached.tokenizer;
        if (this.modelType === 'functiongemma') {
          await this.setupStopSequences();
        }
        return;
      }

      console.log(`[TransformersLlm] Loading model: ${this.modelId}`);
      const startTime = Date.now();
      const { AutoModelForCausalLM, AutoTokenizer } = await getTransformers();
      
      this.tokenizer = await AutoTokenizer.from_pretrained(this.modelId);
      this.transformersModel = await AutoModelForCausalLM.from_pretrained(this.modelId);
      
      // Cache the model
      modelCache.set(this.modelId, { model: this.transformersModel, tokenizer: this.tokenizer });
      
      if (this.modelType === 'functiongemma') {
        await this.setupStopSequences();
      }
      
      const loadTime = ((Date.now() - startTime) / 1000).toFixed(1);
      console.log(`[TransformersLlm] Model loaded in ${loadTime}s`);
    })();

    await this.initPromise;
  }


  private async setupStopSequences(): Promise<void> {
    if (!this.tokenizer) return;
    const { Tensor } = await getTransformers();
    
    const stopSequences = ['<start_function_response>', '<end_of_turn>'];
    this.stopTokenIds = [];
    
    for (const seq of stopSequences) {
      try {
        const encoded = this.tokenizer.encode(seq, { add_special_tokens: false });
        let tokenData: number[] = [];
        if (Array.isArray(encoded)) {
          tokenData = encoded;
        } else if (encoded && typeof encoded === 'object' && 'data' in encoded) {
          tokenData = Array.from((encoded as Tensor).data as BigInt64Array).map(Number);
        }
        if (tokenData.length > 0) {
          this.stopTokenIds.push(Number(tokenData[0]));
        }
      } catch (e) {
        console.warn(`Could not encode stop sequence "${seq}":`, e);
      }
    }
  }

  private formatMessages(llmRequest: LlmRequest): Array<{ role: string; content: string }> {
    const messages: Array<{ role: string; content: string }> = [];
    
    const systemInstruction = llmRequest.config?.systemInstruction;
    if (systemInstruction) {
      let instructionText: string;
      if (typeof systemInstruction === 'string') {
        instructionText = systemInstruction;
      } else if (systemInstruction && 'parts' in systemInstruction && systemInstruction.parts) {
        instructionText = systemInstruction.parts.map((p) => ('text' in p ? p.text : '') || '').join('') || '';
      } else {
        instructionText = '';
      }
      
      if (instructionText) {
        // FunctionGemma uses 'developer' role, SmolLM uses 'system'
        const systemRole = this.modelType === 'functiongemma' ? 'developer' : 'system';
        messages.push({ role: systemRole, content: instructionText });
      }
    }

    for (const content of llmRequest.contents) {
      const role = content.role === 'model' ? 'assistant' : (content.role || 'user');
      const parts = content.parts || [];
      const text = parts.map((part) => {
        if ('text' in part && part.text) return part.text;
        if ('functionCall' in part && part.functionCall) {
          if (this.modelType === 'functiongemma') {
            const args = Object.entries(part.functionCall.args || {})
              .map(([k, v]) => `${k}:<escape>${v}<escape>`).join(',');
            return `<start_function_call>call:${part.functionCall.name}{${args}}<end_function_call>`;
          }
          return `Tool call: ${part.functionCall.name}(${JSON.stringify(part.functionCall.args)})`;
        }
        if ('functionResponse' in part && part.functionResponse) {
          if (this.modelType === 'functiongemma') {
            return `<start_function_response>${JSON.stringify(part.functionResponse.response)}<end_function_response>`;
          }
          return `Tool result: ${JSON.stringify(part.functionResponse.response)}`;
        }
        return '';
      }).filter(Boolean).join('\n') || '';
      
      if (text) messages.push({ role, content: text });
    }
    return messages;
  }

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

  override async *generateContentAsync(llmRequest: LlmRequest, _stream?: boolean): AsyncGenerator<LlmResponse, void> {
    await this.initialize();
    
    if (!this.transformersModel || !this.tokenizer) {
      yield { errorCode: 'MODEL_NOT_LOADED', errorMessage: 'Failed to load model' };
      return;
    }

    // For FunctionGemma: if we already have a function response, don't generate more
    if (this.modelType === 'functiongemma') {
      const { hasResponse, lastResponse } = this.hasFunctionResponse(llmRequest);
      if (hasResponse) {
        const resultText = typeof lastResponse === 'object' ? JSON.stringify(lastResponse) : String(lastResponse);
        yield {
          content: { role: 'model', parts: [{ text: `Tool result: ${resultText}` }] },
          finishReason: 'STOP' as LlmResponse['finishReason'],
        };
        return;
      }
    }

    const messages = this.formatMessages(llmRequest);
    console.log(`[TransformersLlm:${this.modelType}] Generating with ${messages.length} messages`);
    
    try {
      const maxTokens = llmRequest.config?.maxOutputTokens || this.maxNewTokens;
      
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const templateOptions: any = {
        tokenize: true, 
        add_generation_prompt: true, 
        return_dict: true,
      };
      
      if (this.modelType === 'functiongemma' && this.toolSchemas.length > 0) {
        templateOptions.tools = this.toolSchemas;
      }

      const inputs = this.tokenizer.apply_chat_template(messages, templateOptions) as { input_ids: Tensor; attention_mask: Tensor };
      
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const generateConfig: any = { ...inputs, max_new_tokens: maxTokens };

      if (this.modelType === 'functiongemma' && this.stopTokenIds.length > 0) {
        const eosTokenId = this.tokenizer.eos_token_id;
        generateConfig.eos_token_id = eosTokenId ? [eosTokenId, ...this.stopTokenIds] : this.stopTokenIds;
      }

      const startTime = Date.now();
      const output = await this.transformersModel.generate(generateConfig);
      const genTime = Date.now() - startTime;
      
      const inputLength = inputs.input_ids.dims[1];
      const outputData = Array.from((output as Tensor).data as BigInt64Array);
      const newTokenIds = outputData.slice(inputLength);
      const generatedText = this.tokenizer.decode(newTokenIds, { skip_special_tokens: false });

      console.log(`[TransformersLlm:${this.modelType}] Generated ${newTokenIds.length} tokens in ${genTime}ms`);

      let functionCall: { name: string; args: Record<string, unknown> } | null = null;
      let textResponse = generatedText;
      
      if (this.modelType === 'functiongemma') {
        functionCall = this.parseFunctionCall(generatedText);
        if (!functionCall) {
          textResponse = generatedText.replace(/<start_function_response>[\s\S]*$/, '').replace(/<[^>]+>/g, '').trim();
        }
      } else {
        // SmolLM2 cleanup
        textResponse = generatedText
          .replace(/<\|im_start\|>/g, '')
          .replace(/<\|im_end\|>/g, '')
          .replace(/<\|endoftext\|>/g, '')
          .replace(/^(assistant|user|system)\n?/i, '')
          .trim();
      }

      yield {
        content: {
          role: 'model',
          parts: functionCall ? [{ functionCall: { name: functionCall.name, args: functionCall.args } }] : [{ text: textResponse }],
        },
        finishReason: 'STOP' as LlmResponse['finishReason'],
      };
    } catch (error) {
      console.error(`[TransformersLlm:${this.modelType}] Error:`, error);
      yield { errorCode: 'GENERATION_ERROR', errorMessage: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  /**
   * Generate text directly without ADK framework.
   * Used by SmolLM2 for conversational responses.
   */
  async generateText(systemPrompt: string, userMessage: string): Promise<string> {
    await this.initialize();
    
    if (!this.transformersModel || !this.tokenizer) {
      throw new Error('Model not loaded');
    }

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMessage },
    ];

    console.log(`[TransformersLlm:${this.modelType}] generateText for: "${userMessage.substring(0, 50)}..."`);

    const { Tensor } = await getTransformers();
    
    const inputs = this.tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_dict: true,
    }) as { input_ids: Tensor; attention_mask: Tensor };

    const startTime = Date.now();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const generateConfig: any = {
      ...inputs,
      max_new_tokens: this.maxNewTokens,
      temperature: this.temperature,
      top_p: this.topP,
      do_sample: this.doSample,
    };
    const output = await this.transformersModel.generate(generateConfig);
    const genTime = Date.now() - startTime;

    const inputLength = inputs.input_ids.dims[1];
    const outputData = Array.from((output as Tensor).data as BigInt64Array);
    const newTokenIds = outputData.slice(inputLength);
    let generatedText = this.tokenizer.decode(newTokenIds, { skip_special_tokens: false });

    // Clean up SmolLM2 special tokens
    generatedText = generatedText
      .replace(/<\|im_start\|>/g, '')
      .replace(/<\|im_end\|>/g, '')
      .replace(/<\|endoftext\|>/g, '')
      .replace(/^(assistant|user|system)\n?/i, '')
      .trim();

    console.log(`[TransformersLlm:${this.modelType}] Generated ${newTokenIds.length} tokens in ${genTime}ms`);
    
    return generatedText;
  }

  override async connect(_llmRequest: LlmRequest): Promise<BaseLlmConnection> {
    throw new Error('Live connections not supported. Use generateContentAsync instead.');
  }
}

export default TransformersLlm;
