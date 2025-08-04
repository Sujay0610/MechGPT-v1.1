'use client'

import React, { createContext, useContext, useState, ReactNode } from 'react'
import axios from 'axios'

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
  setCurrentConversationId: (conversationId: string | null) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

const ChatContext = createContext<ChatContextType | undefined>(undefined)

export function ChatProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  // Helper function to get auth headers
  const getAuthHeaders = () => {
    const token = localStorage.getItem('auth_token')
    return token ? { Authorization: `Bearer ${token}` } : {}
  }

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
      const response = await axios.get(`/api/conversations/${conversationId}`, {
        headers: getAuthHeaders()
      })
      if (response.data) {
        const history = response.data
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
      const response = await axios.get(`/api/agents/${agentName}/conversations`, {
        headers: getAuthHeaders()
      })
      if (response.data) {
        setConversations(response.data)
      }
    } catch (error) {
      console.error('Error loading agent conversations:', error)
    }
  }

  const deleteConversation = async (conversationId: string) => {
    try {
      const response = await axios.delete(`/api/conversations/${conversationId}`, {
        headers: getAuthHeaders()
      })
      // Axios throws on error status codes, so if we reach here, it was successful
      setConversations(prev => prev.filter(conv => conv.id !== conversationId))
      if (currentConversationId === conversationId) {
        clearMessages()
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
      setCurrentConversationId,
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