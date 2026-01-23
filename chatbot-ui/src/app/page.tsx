import { ChatContainer } from '@/components/chat/chat-container';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-muted p-2 sm:p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-4 sm:mb-6 md:mb-8">
          <h1 className="text-xl sm:text-2xl md:text-3xl font-bold mb-1 sm:mb-2">Local AI Chatbot</h1>
          <p className="text-xs sm:text-sm text-muted-foreground px-2">
            Powered by Google ADK + Transformers.js • 100% Local Inference
          </p>
        </div>
        <ChatContainer />
      </div>
    </main>
  );
}
