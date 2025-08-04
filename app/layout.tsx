import './globals.css'
import { Inter } from 'next/font/google'
import { ChatProvider } from './context/ChatContext'
import { AuthProvider } from './context/AuthContext'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'MechAgent RAG Chatbot',
  description: 'A plug-and-play SaaS-style RAG chatbot for product manuals',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>
          <ChatProvider>
            {children}
          </ChatProvider>
        </AuthProvider>
      </body>
    </html>
  )
}