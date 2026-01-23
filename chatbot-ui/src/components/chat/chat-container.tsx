'use client';

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { ChatMessage, Message, ToolCall } from './chat-message';
import { ChatInput } from './chat-input';
import { Bot, Calculator, Clock, Cloud, Cpu } from 'lucide-react';

export function ChatContainer() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const scrollAnchorRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change or loading state changes
  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSend = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setLoadingStatus('Analyzing request...');

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: content }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to get response');
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        toolCalls: data.toolCalls as ToolCall[],
        agents: data.agents as string[],
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setLoadingStatus('');
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto h-[calc(100vh-12rem)] sm:h-[calc(100vh-14rem)] md:h-[650px] min-h-[400px] flex flex-col">
      <CardHeader className="border-b shrink-0 p-3 sm:p-4 md:pb-3">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <Bot className="h-5 w-5 sm:h-6 sm:w-6" />
            <CardTitle className="text-base sm:text-lg">Multi-Agent Chatbot</CardTitle>
          </div>
          <div className="flex flex-wrap gap-1">
            <Badge variant="outline" className="text-[10px] sm:text-xs px-1.5 sm:px-2">
              <Cpu className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-0.5 sm:mr-1" /> FunctionGemma
            </Badge>
            <Badge variant="secondary" className="text-[10px] sm:text-xs px-1.5 sm:px-2">
              <Calculator className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-0.5 sm:mr-1" /> Calculator
            </Badge>
            <Badge variant="secondary" className="text-[10px] sm:text-xs px-1.5 sm:px-2">
              <Clock className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-0.5 sm:mr-1" /> Time
            </Badge>
            <Badge variant="secondary" className="text-[10px] sm:text-xs px-1.5 sm:px-2">
              <Cloud className="h-2.5 w-2.5 sm:h-3 sm:w-3 mr-0.5 sm:mr-1" /> Weather
            </Badge>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="flex-1 p-0 overflow-hidden">
        <ScrollArea className="h-full">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-4 sm:p-8">
              <Bot className="h-10 w-10 sm:h-12 sm:w-12 mb-3 sm:mb-4 opacity-50" />
              <p className="text-center font-medium mb-2 text-sm sm:text-base">
                Multi-Agent AI Assistant
              </p>
              <p className="text-center text-xs sm:text-sm mb-3 sm:mb-4">
                Powered by FunctionGemma
              </p>
              <div className="text-xs sm:text-sm space-y-1 text-center">
                <p>Try: &quot;What is 15% of 200?&quot;</p>
                <p>Try: &quot;What time is it?&quot;</p>
                <p>Try: &quot;Weather in Tokyo?&quot;</p>
              </div>
            </div>
          ) : (
            <div className="flex flex-col">
              {messages.map((message) => (
                <ChatMessage key={message.id} message={message} />
              ))}
              {isLoading && (
                <div className="flex gap-2 sm:gap-3 p-3 sm:p-4">
                  <div className="h-7 w-7 sm:h-8 sm:w-8 rounded-full bg-muted flex items-center justify-center shrink-0">
                    <Bot className="h-3.5 w-3.5 sm:h-4 sm:w-4 animate-pulse" />
                  </div>
                  <div className="flex flex-col gap-1">
                    <span className="text-xs sm:text-sm text-muted-foreground">{loadingStatus || 'Processing...'}</span>
                    <div className="flex gap-1">
                      <span className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-1.5 h-1.5 sm:w-2 sm:h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              )}
              {/* Scroll anchor - invisible element at the bottom for auto-scroll */}
              <div ref={scrollAnchorRef} className="h-px" />
            </div>
          )}
        </ScrollArea>
      </CardContent>
      
      <ChatInput onSend={handleSend} isLoading={isLoading} />
    </Card>
  );
}
