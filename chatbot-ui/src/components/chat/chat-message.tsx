'use client';

import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Bot, User, Wrench, Cpu } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  result?: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  toolCalls?: ToolCall[];
  agents?: string[];
  timestamp: Date;
}

interface ChatMessageProps {
  message: Message;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  // Get unique agents involved (filter out duplicates and generic names)
  const uniqueAgents = message.agents 
    ? [...new Set(message.agents)].filter(a => a !== 'model' && a !== 'unknown')
    : [];

  return (
    <div className={cn('flex gap-2 sm:gap-3 p-2 sm:p-4', isUser ? 'flex-row-reverse' : 'flex-row')}>
      <Avatar className="h-7 w-7 sm:h-8 sm:w-8 shrink-0">
        <AvatarFallback className={cn(isUser ? 'bg-primary text-primary-foreground' : 'bg-muted')}>
          {isUser ? <User className="h-3.5 w-3.5 sm:h-4 sm:w-4" /> : <Bot className="h-3.5 w-3.5 sm:h-4 sm:w-4" />}
        </AvatarFallback>
      </Avatar>
      
      <div className={cn('flex flex-col gap-1.5 sm:gap-2 min-w-0 flex-1', isUser ? 'items-end' : 'items-start')}>
        {/* Show agents involved */}
        {!isUser && uniqueAgents.length > 0 && (
          <div className="flex gap-1 flex-wrap">
            {uniqueAgents.map((agent, idx) => (
              <Badge key={idx} variant="outline" className="text-[10px] sm:text-xs py-0 px-1.5">
                <Cpu className="h-2 w-2 sm:h-2.5 sm:w-2.5 mr-0.5 sm:mr-1" />
                {agent}
              </Badge>
            ))}
          </div>
        )}

        {/* Tool calls */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="flex flex-col gap-1.5 sm:gap-2 w-full max-w-full">
            {message.toolCalls.map((tool, idx) => (
              <Card key={idx} className="px-2 sm:px-3 py-1.5 sm:py-2 bg-amber-50 dark:bg-amber-950 border-amber-200 dark:border-amber-800 overflow-hidden">
                <div className="flex items-center gap-1.5 sm:gap-2 mb-1">
                  <Wrench className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-amber-600 shrink-0" />
                  <Badge variant="outline" className="text-[10px] sm:text-xs bg-amber-100 dark:bg-amber-900 px-1.5">
                    {tool.name}
                  </Badge>
                </div>
                <div className="text-[10px] sm:text-xs text-muted-foreground font-mono overflow-x-auto">
                  <div className="break-all">Args: {JSON.stringify(tool.args)}</div>
                  {tool.result && (
                    <div className="mt-0.5 sm:mt-1 text-green-600 dark:text-green-400 break-all">
                      Result: {JSON.stringify(tool.result)}
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* Main message content */}
        <Card className={cn('px-3 sm:px-4 py-1.5 sm:py-2 max-w-full', isUser ? 'bg-primary text-primary-foreground' : 'bg-muted')}>
          <p className="text-xs sm:text-sm whitespace-pre-wrap break-words">{message.content}</p>
        </Card>
        
        <span className="text-[10px] sm:text-xs text-muted-foreground">
          {message.timestamp.toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}
