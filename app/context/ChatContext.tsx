'use client'

import React, { createContext, useContext, useState, ReactNode } from 'react'

interface Message {
  id: string
  text: string
  sender: 'user' | 'bot'
  timestamp: Date
}

interface Conversation {
  id: string
  agent_name: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

interface ChatContextType {
  messages: Message[]
  conversations: Conversation[]
  currentConversationId: string | null
  addMessage: (text: string, sender: 'user' | 'bot') => void
  clearMessages: () => void
  loadConversation: (conversationId: string) => Promise<void>
  loadAgentConversations: (agentName: string) => Promise<void>
  createNewConversation: () => void
  deleteConversation: (conversationId: string) => Promise<void>
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

const ChatContext = createContext<ChatContextType | undefined>(undefined)

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const addMessage = (text: string, sender: 'user' | 'bot') => {
    const newMessage: Message = {
      id: Date.now().toString(),
      text,
      sender,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, newMessage])
  }

  const clearMessages = () => {
    setMessages([])
    setCurrentConversationId(null)
  }

  const createNewConversation = () => {
    setMessages([])
    setCurrentConversationId(null)
  }

  const loadConversation = async (conversationId: string) => {
    try {
      setIsLoading(true)
      const response = await fetch(`/api/conversations/${conversationId}`)
      if (response.ok) {
        const history = await response.json()
        const loadedMessages = history.messages.map((msg: any) => ({
          id: msg.id,
          text: msg.text,
          sender: msg.sender,
          timestamp: new Date(msg.timestamp)
        }))
        setMessages(loadedMessages)
        setCurrentConversationId(conversationId)
      }
    } catch (error) {
      console.error('Error loading conversation:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadAgentConversations = async (agentName: string) => {
    try {
      const response = await fetch(`/api/agents/${agentName}/conversations`)
      if (response.ok) {
        const agentConversations = await response.json()
        setConversations(agentConversations)
      }
    } catch (error) {
      console.error('Error loading agent conversations:', error)
    }
  }

  const deleteConversation = async (conversationId: string) => {
    try {
      const response = await fetch(`/api/conversations/${conversationId}`, {
        method: 'DELETE'
      })
      if (response.ok) {
        setConversations(prev => prev.filter(conv => conv.id !== conversationId))
        if (currentConversationId === conversationId) {
          clearMessages()
        }
      }
    } catch (error) {
      console.error('Error deleting conversation:', error)
    }
  }

  return (
    <ChatContext.Provider value={{
      messages,
      conversations,
      currentConversationId,
      addMessage,
      clearMessages,
      loadConversation,
      loadAgentConversations,
      createNewConversation,
      deleteConversation,
      isLoading,
      setIsLoading
    }}>
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const context = useContext(ChatContext)
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider')
  }
  return context
}