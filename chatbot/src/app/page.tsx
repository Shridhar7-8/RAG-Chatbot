import { ChatInterface } from '@/components/chat/chat-interface'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 p-4">
      <ChatInterface />
    </main>
  )
}

