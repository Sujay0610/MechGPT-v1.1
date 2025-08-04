'use client'

import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import { useChat } from './context/ChatContext'
import { useAuth } from './context/AuthContext'
import ProtectedRoute from './components/ProtectedRoute'
import MainLayout from './components/MainLayout'

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

interface Agent {
  id: string
  name: string
  description: string
  extra_instructions: string
  created_at: string
  updated_at: string
}

interface AgentStats {
  agent_name: string
  total_chunks: number
  total_files: number
  files: string[]
  created_at: string
  updated_at: string
  description: string
  extra_instructions: string
}

function ChatPage() {
  const {
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
  } = useChat()
  
  const { user } = useAuth()
  
  // Helper function to get auth headers
  const getAuthHeaders = () => {
    const token = localStorage.getItem('auth_token')
    return token ? { Authorization: `Bearer ${token}` } : {}
  }
  
  const [agents, setAgents] = useState<Agent[]>([])
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null)
  const [agentStats, setAgentStats] = useState<Record<string, AgentStats>>({})
  const [input, setInput] = useState('')
  const [agentsLoading, setAgentsLoading] = useState(true)
  const [showCreateForm, setShowCreateForm] = useState(false)
  const [newAgentName, setNewAgentName] = useState('')
  const [newAgentDescription, setNewAgentDescription] = useState('')
  const [newAgentInstructions, setNewAgentInstructions] = useState('')
  const [creating, setCreating] = useState(false)
  const [showUploadForm, setShowUploadForm] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [uploadMode, setUploadMode] = useState<'files' | 'text' | 'links'>('files')
  const [textContent, setTextContent] = useState('')
  const [textTitle, setTextTitle] = useState('')
  const [linkUrls, setLinkUrls] = useState('')
  const [showConversations, setShowConversations] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [agentToDelete, setAgentToDelete] = useState<Agent | null>(null)
  const [deleting, setDeleting] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchAgents()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const fetchAgents = async () => {
    try {
      const response = await axios.get('/api/agents', {
        headers: getAuthHeaders()
      })
      setAgents(response.data)
      
      // Fetch stats for each agent
      const statsPromises = response.data.map(async (agent: Agent) => {
        try {
          const statsResponse = await axios.get(`/api/agents/${agent.name}/stats`, {
            headers: getAuthHeaders()
          })
          return { [agent.name]: statsResponse.data }
        } catch (error) {
          return { [agent.name]: { 
            agent_name: agent.name,
            total_chunks: 0, 
            total_files: 0, 
            files: [],
            created_at: '',
            updated_at: '',
            description: '',
            extra_instructions: ''
          } }
        }
      })
      
      const statsResults = await Promise.all(statsPromises)
      const combinedStats = statsResults.reduce((acc, stat) => ({ ...acc, ...stat }), {})
      setAgentStats(combinedStats)
      
      // Auto-select first agent if available
      if (response.data.length > 0 && !selectedAgent) {
        setSelectedAgent(response.data[0])
      }
    } catch (error) {
      console.error('Error fetching agents:', error)
    } finally {
      setAgentsLoading(false)
    }
  }

  const createAgent = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newAgentName.trim()) return

    setCreating(true)
    try {
      await axios.post('/api/agents', {
        name: newAgentName.trim(),
        description: newAgentDescription.trim() || 'No description provided',
        extra_instructions: newAgentInstructions.trim() || ''
      }, {
        headers: getAuthHeaders()
      })
      
      setNewAgentName('')
      setNewAgentDescription('')
      setNewAgentInstructions('')
      setShowCreateForm(false)
      await fetchAgents()
    } catch (error: any) {
      console.error('Error creating agent:', error)
      alert(error.response?.data?.detail || 'Failed to create agent')
    } finally {
      setCreating(false)
    }
  }

  const handleDeleteAgent = (agent: Agent, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent agent selection when clicking delete
    setAgentToDelete(agent)
    setShowDeleteConfirm(true)
  }

  const confirmDeleteAgent = async () => {
    if (!agentToDelete) return

    setDeleting(true)
    try {
      await axios.delete(`/api/agents/${agentToDelete.name}`, {
        headers: getAuthHeaders()
      })
      
      // If the deleted agent was selected, clear selection
      if (selectedAgent?.id === agentToDelete.id) {
        setSelectedAgent(null)
        clearMessages()
      }
      
      setShowDeleteConfirm(false)
      setAgentToDelete(null)
      await fetchAgents()
    } catch (error: any) {
      console.error('Error deleting agent:', error)
      alert(error.response?.data?.detail || 'Failed to delete agent')
    } finally {
      setDeleting(false)
    }
  }

  const cancelDeleteAgent = () => {
    setShowDeleteConfirm(false)
    setAgentToDelete(null)
  }

  const selectAgent = async (agent: Agent) => {
    // Prevent agent selection during upload
    if (uploadingAgent) {
      alert('Please wait for the current upload to complete before selecting a different agent.')
      return
    }
    
    setSelectedAgent(agent)
    clearMessages() // Clear messages when switching agents
    setCurrentConversationId(null) // Always start with new chat
    setShowConversations(false) // Always start in chat mode, not history
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    const pdfFiles = files.filter(file => file.type === 'application/pdf')
    setSelectedFiles(pdfFiles)
  }

  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files)
    const pdfFiles = files.filter(file => file.type === 'application/pdf')
    setSelectedFiles(pdfFiles)
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const [uploadProgress, setUploadProgress] = useState('')
  const [uploadResult, setUploadResult] = useState('')
  const [showUploadStatus, setShowUploadStatus] = useState(false)
  const [uploadingAgent, setUploadingAgent] = useState<string | null>(null)

  const pollJobStatus = async (jobId: string, agentName: string) => {
    const maxAttempts = 120 // 10 minutes with 5-second intervals
    let attempts = 0
    
    const poll = async (): Promise<void> => {
      try {
        const response = await axios.get(`/api/agents/${agentName}/upload/status/${jobId}`, {
          headers: getAuthHeaders()
        })
        const status = response.data
        
        // Update progress
        const progressText = `${status.message}\n\nProgress: ${status.progress}/${status.total_files} files\nProcessed: ${status.processed_files?.length || 0}\nSkipped: ${status.skipped_files?.length || 0}\nFailed: ${status.failed_files?.length || 0}`
        setUploadProgress(progressText)
        
        if (status.status === 'completed') {
          setUploadResult(status.message)
          setSelectedFiles([])
          setShowUploadForm(false)
          await fetchAgents() // Refresh agents
          setUploading(false)
          setUploadingAgent(null)
          setShowUploadStatus(false)
          return
        }
        
        if (status.status === 'failed') {
          setUploadResult(`Processing failed: ${status.message}`)
          setUploading(false)
          setUploadingAgent(null)
          return
        }
        
        // Continue polling if still processing
        if (status.status === 'processing' && attempts < maxAttempts) {
          attempts++
          setTimeout(poll, 5000) // Poll every 5 seconds
        } else if (attempts >= maxAttempts) {
          setUploadResult('Status check timed out. Files may still be processing.')
          setUploading(false)
          setUploadingAgent(null)
        }
        
      } catch (error: any) {
        console.error('Status polling error:', error)
        setUploadResult(`Status check failed: ${error.message}`)
        setUploading(false)
        setUploadingAgent(null)
      }
    }
    
    poll()
  }

  const uploadFiles = async () => {
    if (!selectedAgent || selectedFiles.length === 0) return

    setUploading(true)
    setUploadingAgent(selectedAgent.name)
    setUploadProgress('Starting upload...')
    setUploadResult('')
    setShowUploadStatus(true)
    
    // Minimize the upload dialog
    setShowUploadForm(false)
    
    try {
      const formData = new FormData()
      selectedFiles.forEach(file => {
        formData.append('files', file)
      })

      setUploadProgress(`Uploading ${selectedFiles.length} file(s) and starting background processing...`)
      
      const response = await axios.post(`/api/agents/${selectedAgent.name}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          ...getAuthHeaders()
        }
      })
      
      const result = response.data
      
      if (result.job_id) {
        setUploadProgress('Files uploaded successfully. Starting processing...')
        // Start polling for job status
        pollJobStatus(result.job_id, selectedAgent.name)
      } else {
        setUploadResult(result.message || 'Files uploaded successfully!')
        setSelectedFiles([])
        await fetchAgents()
        setUploading(false)
        setUploadingAgent(null)
        setShowUploadStatus(false)
      }
      
    } catch (error: any) {
      console.error('Upload error:', error)
      
      let errorMessage = 'Failed to upload files'
      
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.message) {
        errorMessage = `Upload failed: ${error.message}`
      }
      
      setUploadResult(errorMessage)
      setUploading(false)
      setUploadingAgent(null)
      setUploadProgress('')
    }
  }

  const uploadText = async () => {
    if (!selectedAgent || !textContent.trim()) return

    setUploading(true)
    setUploadProgress('Processing text content...')
    setUploadResult('')
    
    try {
      const response = await axios.post(`/api/agents/${selectedAgent.name}/text`, {
        content: textContent.trim(),
        title: textTitle.trim() || 'Text Content'
      }, {
        headers: getAuthHeaders()
      })
      
      const result = response.data
      setUploadResult(result.message || 'Text content processed successfully!')
      setTextContent('')
      setTextTitle('')
      await fetchAgents()
      setUploading(false)
      
    } catch (error: any) {
      console.error('Text upload error:', error)
      
      let errorMessage = 'Failed to process text content'
      
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.message) {
        errorMessage = `Text processing failed: ${error.message}`
      }
      
      setUploadResult(errorMessage)
      setUploading(false)
      setUploadProgress('')
    }
  }

  const uploadLinks = async () => {
    if (!selectedAgent || !linkUrls.trim()) return

    setUploading(true)
    setUploadProgress('Crawling and processing links...')
    setUploadResult('')
    
    try {
      const urls = linkUrls.split('\n').map(url => url.trim()).filter(url => url)
      
      const response = await axios.post(`/api/agents/${selectedAgent.name}/crawl`, {
        urls: urls
      }, {
        headers: getAuthHeaders()
      })
      
      const result = response.data
      setUploadResult(result.message || 'Links crawled and processed successfully!')
      setLinkUrls('')
      await fetchAgents()
      setUploading(false)
      
    } catch (error: any) {
      console.error('Link crawl error:', error)
      
      let errorMessage = 'Failed to crawl and process links'
      
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail
      } else if (error.message) {
        errorMessage = `Link crawling failed: ${error.message}`
      }
      
      setUploadResult(errorMessage)
      setUploading(false)
      setUploadProgress('')
    }
  }

  const handleUpload = () => {
    switch (uploadMode) {
      case 'files':
        uploadFiles()
        break
      case 'text':
        uploadText()
        break
      case 'links':
        uploadLinks()
        break
    }
  }

  const removeFile = (index: number) => {
    setSelectedFiles(files => files.filter((_, i) => i !== index))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || !selectedAgent) return

    const userMessage = input.trim()
    setInput('')
    addMessage(userMessage, 'user')
    setIsLoading(true)

    try {
      const requestBody: any = {
        message: userMessage
      }
      
      if (currentConversationId) {
        requestBody.conversation_id = currentConversationId
      }
      
      const response = await axios.post(`/api/agents/${selectedAgent.name}/chat`, requestBody, {
        headers: getAuthHeaders()
      })
      
      addMessage(response.data.response, 'bot')
      
      // If this was a new conversation, set the conversation ID and load the updated conversations list
      if (!currentConversationId && response.data.conversation_id) {
        setCurrentConversationId(response.data.conversation_id)
        await loadAgentConversations(selectedAgent.name)
      }
    } catch (error: any) {
      console.error('Chat error:', error)
      addMessage('Sorry, I encountered an error. Please try again.', 'bot')
    } finally {
      setIsLoading(false)
    }
  }



  return (
    <div className="page-container flex h-[calc(100vh-8rem)] max-w-7xl mx-auto">
      {/* Upload Status Indicator */}
      {showUploadStatus && uploadingAgent && (
        <div className="fixed top-20 right-4 z-40 w-80 bg-white border border-blue-200 rounded-lg shadow-lg">
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-blue-900">Upload Progress</h3>
              <div className="flex items-center space-x-2">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                <span className="text-xs text-blue-600">{uploadingAgent}</span>
              </div>
            </div>
            {uploadProgress && (
              <div className="text-xs text-gray-600 whitespace-pre-line bg-gray-50 p-2 rounded border max-h-32 overflow-y-auto">
                {uploadProgress}
              </div>
            )}
            {uploadResult && (
              <div className={`text-xs mt-2 p-2 rounded border ${
                uploadResult.includes('failed') || uploadResult.includes('error') 
                  ? 'bg-red-50 border-red-200 text-red-800' 
                  : 'bg-green-50 border-green-200 text-green-800'
              }`}>
                {uploadResult}
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Sidebar - Agents List */}
      <div className="w-72 bg-white border-r border-gray-200 flex flex-col shadow-lg">
        {/* Sidebar Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-lg font-semibold text-gray-900">
              {selectedAgent && showConversations ? 'Conversations' : 'Agents'}
            </h2>
            <div className="flex space-x-2">
              {selectedAgent && showConversations && (
                <button
                  onClick={createNewConversation}
                  className="animated-button bg-blue-600 text-white px-3 py-1 rounded-lg text-sm hover:bg-blue-700 shadow-sm"
                >
                  💬 New Chat
                </button>
              )}
              {!(selectedAgent && showConversations) && (
                <button
                  onClick={() => setShowCreateForm(true)}
                  className="animated-button bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700"
                >
                  + Agent
                </button>
              )}
            </div>
          </div>
          <p className="text-sm text-gray-600">
            {selectedAgent && showConversations 
              ? `Chat history for ${selectedAgent.name}` 
              : 'Select an agent to chat with'}
          </p>
        </div>

        {/* Agents List / Conversations List */}
        <div className="flex-1 overflow-y-auto">
          {selectedAgent && showConversations ? (
            // Conversations List
            <div className="p-2">
              {conversations.length === 0 ? (
                <div className="p-4 text-center">
                  <div className="text-gray-400 text-3xl mb-2">💬</div>
                  <p className="text-sm text-gray-600 mb-3">No conversations yet</p>
                  <button
                    onClick={createNewConversation}
                    className="animated-button bg-blue-600 text-white px-3 py-2 rounded-lg text-sm hover:bg-blue-700 shadow-sm"
                  >
                    💬 Start First Chat
                  </button>
                </div>
              ) : (
                conversations.map((conversation) => {
                  const isSelected = currentConversationId === conversation.id
                  return (
                    <div
                      key={conversation.id}
                      className={`sidebar-item p-3 rounded-lg mb-2 cursor-pointer transition-colors ${
                        isSelected ? 'bg-green-50 border border-green-200' : 'hover:bg-gray-50'
                      }`}
                      onClick={() => loadConversation(conversation.id)}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h3 className={`font-medium text-sm ${
                          isSelected ? 'text-green-900' : 'text-gray-900'
                        }`}>{conversation.title}</h3>
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            deleteConversation(conversation.id)
                          }}
                          className="text-red-500 hover:text-red-700 text-xs"
                        >
                          ✕
                        </button>
                      </div>
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>{conversation.message_count} messages</span>
                        <span>{new Date(conversation.updated_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                  )
                })
              )}
            </div>
          ) : (
            // Agents List
            agentsLoading ? (
              <div className="flex justify-center items-center h-32">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              </div>
            ) : agents.length === 0 ? (
              <div className="p-4 text-center">
                <div className="text-gray-400 text-3xl mb-2">🤖</div>
                <p className="text-sm text-gray-600 mb-3">No agents yet</p>
                <button
                  onClick={() => setShowCreateForm(true)}
                  className="animated-button bg-blue-600 text-white px-3 py-2 rounded text-sm hover:bg-blue-700"
                >
                  Create First Agent
                </button>
              </div>
            ) : (
              <div className="p-2">
                {agents.map((agent) => {
                  const stats = agentStats[agent.name] || { total_chunks: 0, total_files: 0, last_updated: 'Never' }
                  const isSelected = selectedAgent?.id === agent.id
                  return (
                    <div
                      key={agent.id}
                      className={`sidebar-item p-3 rounded-lg mb-2 transition-colors ${
                        uploadingAgent && uploadingAgent !== agent.name 
                          ? 'opacity-50 cursor-not-allowed bg-gray-100' 
                          : isSelected 
                            ? 'bg-blue-50 border border-blue-200 cursor-pointer' 
                            : 'hover:bg-gray-50 cursor-pointer'
                      }`}
                      onClick={() => selectAgent(agent)}
                    >
                      <div className="flex justify-between items-start mb-2">
                        <h3 className={`font-medium ${
                          isSelected ? 'text-blue-900' : 'text-gray-900'
                        }`}>{agent.name}</h3>
                        <button
                          onClick={(e) => handleDeleteAgent(agent, e)}
                          className="text-red-500 hover:text-red-700 p-1 rounded transition-colors"
                          title="Delete agent"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                      <p className="text-xs text-gray-600 mb-2">{agent.description}</p>
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>{stats.total_files} files</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            )
          )}
        </div>
      </div>

      {/* Main Chat Area - Widget Style */}
      <div className="flex-1 flex flex-col chat-widget m-4">
        {selectedAgent ? (
          <>
            {/* Chat Header - Widget Style */}
            <div className="chat-widget-header">
              <div className="flex justify-between items-center">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
                    <span className="text-lg">🤖</span>
                  </div>
                  <div>
                    <h1 className="text-lg font-semibold">{selectedAgent.name}</h1>
                    <p className="text-sm text-blue-100 opacity-90">{selectedAgent.description}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="text-xs text-blue-100 bg-white bg-opacity-20 px-2 py-1 rounded-full">
                    {agentStats[selectedAgent.name]?.total_files || 0} files
                  </div>
                  <button
                    onClick={() => setShowUploadForm(true)}
                    disabled={uploadingAgent && uploadingAgent !== selectedAgent.name}
                    className={`animated-button px-3 py-1.5 rounded-lg text-sm ${
                      uploadingAgent && uploadingAgent !== selectedAgent.name
                        ? 'bg-gray-400 text-gray-600 cursor-not-allowed opacity-50'
                        : 'bg-white bg-opacity-20 hover:bg-opacity-30 text-white'
                    }`}
                  >
                    📁 Upload
                  </button>
                  <button
                    onClick={() => {
                      // Start a new conversation
                      clearMessages()
                      setCurrentConversationId(null)
                      setShowConversations(false)
                    }}
                    disabled={uploadingAgent && uploadingAgent !== selectedAgent.name}
                    className={`animated-button px-3 py-1.5 rounded-lg text-sm ${
                      uploadingAgent && uploadingAgent !== selectedAgent.name
                        ? 'bg-gray-400 text-gray-600 cursor-not-allowed opacity-50'
                        : 'bg-white bg-opacity-20 hover:bg-opacity-30 text-white'
                    }`}
                  >
                    💬 New Chat
                  </button>
                  <button
                    onClick={async () => {
                      if (!showConversations && selectedAgent) {
                        // Load conversations immediately when showing history
                        await loadAgentConversations(selectedAgent.name)
                      } else if (showConversations) {
                        // When going back from history to chat, clear current conversation
                        // to start fresh chat mode
                        setCurrentConversationId(null)
                        clearMessages()
                      }
                      setShowConversations(!showConversations)
                    }}
                    disabled={uploadingAgent && uploadingAgent !== selectedAgent.name}
                    className={`animated-button px-3 py-1.5 rounded-lg text-sm ${
                      uploadingAgent && uploadingAgent !== selectedAgent.name
                        ? 'bg-gray-400 text-gray-600 cursor-not-allowed opacity-50'
                        : 'bg-white bg-opacity-20 hover:bg-opacity-30 text-white'
                    }`}
                  >
                    {showConversations ? '← Back' : '📜 History'}
                  </button>
                </div>
              </div>
            </div>

            {/* Chat Messages */}
            <div className="chat-widget-messages">
              {messages.length === 0 && (
                <div className="text-center py-12">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <span className="text-2xl">💬</span>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Chat with {selectedAgent.name}</h3>
                  <p className="text-gray-600 text-sm mb-4">
                    Ask questions about your uploaded documents and get instant help.
                  </p>
                  {(agentStats[selectedAgent.name]?.total_files || 0) === 0 && (
                    <div className="bg-orange-50 border border-orange-200 rounded-lg p-3 text-orange-700 text-sm">
                      📁 No files uploaded yet. Upload documents to get started!
                    </div>
                  )}
                </div>
              )}
              
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`message-bubble ${
                      message.sender === 'user' ? 'user-message' : 'bot-message'
                    }`}
                  >
                    {message.sender === 'user' ? (
                      <p className="text-sm">{message.text}</p>
                    ) : (
                      <div className="text-sm prose prose-sm max-w-none">
                        <ReactMarkdown
                          components={{
                            p: ({children}) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
                            ul: ({children}) => <ul className="list-disc list-inside mb-3 space-y-1 pl-2">{children}</ul>,
                            ol: ({children}) => <ol className="list-decimal list-inside mb-3 space-y-1 pl-2">{children}</ol>,
                            li: ({children}) => <li className="text-sm leading-relaxed">{children}</li>,
                            strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
                            em: ({children}) => <em className="italic text-gray-700">{children}</em>,
                            code: ({children}) => <code className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs font-mono border">{children}</code>,
                            pre: ({children}) => <pre className="bg-gray-50 p-3 rounded-lg border overflow-x-auto mb-3">{children}</pre>,
                            blockquote: ({children}) => <blockquote className="border-l-4 border-blue-200 pl-4 py-2 bg-blue-50 rounded-r mb-3 italic">{children}</blockquote>,
                            h1: ({children}) => <h1 className="text-lg font-bold mb-3 text-gray-900 border-b border-gray-200 pb-1">{children}</h1>,
                            h2: ({children}) => <h2 className="text-base font-bold mb-2 text-gray-900">{children}</h2>,
                            h3: ({children}) => <h3 className="text-sm font-bold mb-2 text-gray-800">{children}</h3>,
                            a: ({href, children}) => (
                              <a 
                                href={href} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="text-blue-600 hover:text-blue-800 underline decoration-blue-300 hover:decoration-blue-500 transition-colors duration-200 font-medium"
                              >
                                {children}
                                <span className="inline-block ml-1 text-xs">↗</span>
                              </a>
                            ),
                          }}
                        >
                          {message.text}
                        </ReactMarkdown>
                      </div>
                    )}
                    <p className="text-xs opacity-70 mt-1">
                      {typeof window !== 'undefined' ? message.timestamp.toLocaleTimeString() : ''}
                    </p>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="message-bubble bot-message">
                    <div className="flex items-center space-x-3 animate-pulse">
                      <div className="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                      <span className="text-sm text-gray-600 font-medium">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>

            {/* Chat Input - Widget Style */}
            <div className="chat-widget-input">
              <form onSubmit={handleSubmit} className="flex space-x-3">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder={uploadingAgent && uploadingAgent !== selectedAgent.name 
                      ? `Upload in progress for ${uploadingAgent}. Please wait...` 
                      : `💬 Ask ${selectedAgent.name} anything...`}
                    className="animated-input w-full border border-gray-300 rounded-full px-5 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent shadow-sm transition-all duration-200 hover:shadow-md bg-white"
                    disabled={isLoading || (uploadingAgent && uploadingAgent !== selectedAgent.name)}
                  />
                  <button
                    type="submit"
                    disabled={!input.trim() || isLoading || (uploadingAgent && uploadingAgent !== selectedAgent.name)}
                    className="animated-button absolute right-2 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-blue-500 to-blue-600 text-white w-8 h-8 rounded-full hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-lg flex items-center justify-center"
                  >
                    {isLoading ? (
                      <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></div>
                    ) : (
                      <span className="text-sm">➤</span>
                    )}
                  </button>
                </div>
              </form>
              <div className="flex items-center justify-center mt-2">
                <span className="text-xs text-gray-400">Powered by <span className="font-medium text-blue-600">MechGPT</span></span>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
            <div className="text-center p-8 bg-white rounded-2xl shadow-xl border border-gray-200 max-w-md mx-4">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl text-white">🤖</span>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Welcome to MechGPT</h3>
              <p className="text-gray-600 mb-6 text-sm leading-relaxed">
                Your AI-powered technical assistant. Select an agent from the sidebar to start getting help with manuals, troubleshooting, and technical documentation.
              </p>
              {agents.length === 0 && (
                <button
                    onClick={() => setShowCreateForm(true)}
                    className="animated-button bg-gradient-to-r from-blue-500 to-blue-600 text-white px-6 py-3 rounded-full hover:from-blue-600 hover:to-blue-700 shadow-md hover:shadow-lg font-medium"
                  >
                    🚀 Create Your First Agent
                  </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Create Agent Modal */}
      {showCreateForm && (
          <div className="modal-overlay fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="modal-content bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold mb-4">Create New Agent</h2>
            <form onSubmit={createAgent}>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Agent Name *
                </label>
                <input
                  type="text"
                  value={newAgentName}
                  onChange={(e) => setNewAgentName(e.target.value)}
                  className="animated-input w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter agent name"
                  required
                />
              </div>
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                </label>
                <textarea
                  value={newAgentDescription}
                  onChange={(e) => setNewAgentDescription(e.target.value)}
                  className="animated-input w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter agent description"
                  rows={3}
                />
              </div>
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Extra Instructions
                </label>
                <textarea
                  value={newAgentInstructions}
                  onChange={(e) => setNewAgentInstructions(e.target.value)}
                  className="animated-input w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Enter custom instructions for this agent (optional)"
                  rows={4}
                />
              </div>
              <div className="flex justify-end space-x-3">
                <button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="px-4 py-2 text-gray-600 hover:text-gray-800"
                  disabled={creating}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creating || !newAgentName.trim()}
                  className="animated-button bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {creating ? 'Creating...' : 'Create Agent'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Upload Files Modal */}
      {showUploadForm && (
          <div className="modal-overlay fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="modal-content bg-white rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <h2 className="text-xl font-bold mb-4">Upload Content to {selectedAgent?.name || 'Agent'}</h2>
            
            {/* Upload Mode Tabs */}
            <div className="flex border-b mb-4">
              <button
                onClick={() => setUploadMode('files')}
                className={`px-4 py-2 font-medium ${uploadMode === 'files' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                Files
              </button>
              <button
                onClick={() => setUploadMode('text')}
                className={`px-4 py-2 font-medium ${uploadMode === 'text' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                Text/Markdown
              </button>
              <button
                onClick={() => setUploadMode('links')}
                className={`px-4 py-2 font-medium ${uploadMode === 'links' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-500 hover:text-gray-700'}`}
              >
                Links
              </button>
            </div>

            {/* Files Upload */}
            {uploadMode === 'files' && (
              <>
                {/* File Drop Zone */}
                <div
                  onDrop={handleFileDrop}
                  onDragOver={handleDragOver}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-4 hover:border-blue-400 transition-colors"
                >
                  <div className="text-gray-400 text-4xl mb-4">📄</div>
                  <p className="text-gray-600 mb-2">Drag and drop PDF files here, or</p>
                  <label className="animated-button bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 cursor-pointer inline-block">
                    Choose Files
                    <input
                      type="file"
                      multiple
                      accept=".pdf"
                      onChange={handleFileSelect}
                      className="hidden"
                    />
                  </label>
                  <p className="text-sm text-gray-500 mt-2">Only PDF files are supported</p>
                </div>

                {/* Selected Files List */}
                {selectedFiles.length > 0 && (
                  <div className="mb-4">
                    <h3 className="font-medium mb-2">Selected Files ({selectedFiles.length})</h3>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {selectedFiles.map((file, index) => (
                        <div key={index} className="flex items-center justify-between bg-gray-50 p-2 rounded">
                          <div className="flex items-center space-x-2">
                            <span className="text-red-500">📄</span>
                            <span className="text-sm">{file.name}</span>
                            <span className="text-xs text-gray-500">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                          </div>
                          <button
                            onClick={() => removeFile(index)}
                            className="text-red-500 hover:text-red-700 text-sm"
                            disabled={uploading}
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Text/Markdown Upload */}
            {uploadMode === 'text' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Title (optional)
                  </label>
                  <input
                    type="text"
                    value={textTitle}
                    onChange={(e) => setTextTitle(e.target.value)}
                    placeholder="Enter a title for this content"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Content *
                  </label>
                  <textarea
                    value={textContent}
                    onChange={(e) => setTextContent(e.target.value)}
                    placeholder="Enter your text or markdown content here..."
                    rows={12}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 resize-vertical"
                  />
                  <p className="text-sm text-gray-500 mt-1">Supports plain text and Markdown formatting</p>
                </div>
              </div>
            )}

            {/* Links Upload */}
            {uploadMode === 'links' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    URLs to Crawl *
                  </label>
                  <textarea
                    value={linkUrls}
                    onChange={(e) => setLinkUrls(e.target.value)}
                    placeholder="Enter URLs, one per line:\nhttps://example.com/page1\nhttps://example.com/page2"
                    rows={8}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 resize-vertical"
                  />
                  <p className="text-sm text-gray-500 mt-1">Enter one URL per line. The content will be crawled and processed automatically.</p>
                </div>
              </div>
            )}

            {/* Upload Progress */}
            {uploading && (
              <div className="mb-4">
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span className="text-sm font-medium text-blue-800">
                      {uploadMode === 'files' && `Processing ${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''}...`}
                      {uploadMode === 'text' && 'Processing text content...'}
                      {uploadMode === 'links' && 'Crawling and processing links...'}
                    </span>
                  </div>
                  {uploadProgress && (
                    <div className="text-xs text-blue-600 whitespace-pre-line">
                      {uploadProgress}
                    </div>
                  )}
                  {!uploadProgress && uploadMode === 'files' && (
                    <div className="text-xs text-blue-600">
                      <div>• Uploading files to server</div>
                      <div>• Parsing PDFs with LlamaParse</div>
                      <div>• Creating text chunks and embeddings</div>
                      <div>• Indexing in knowledge base</div>
                      <div className="mt-1 font-medium">This may take several minutes for large files or batches.</div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Upload Result */}
            {uploadResult && (
              <div className="mb-4">
                <div className={`border rounded-lg p-4 ${
                  uploadResult.includes('failed') || uploadResult.includes('error') 
                    ? 'bg-red-50 border-red-200 text-red-800' 
                    : 'bg-green-50 border-green-200 text-green-800'
                }`}>
                  <div className="text-sm whitespace-pre-line">{uploadResult}</div>
                </div>
              </div>
            )}

            {/* Modal Actions */}
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={() => {
                  setShowUploadForm(false)
                  setSelectedFiles([])
                  setTextContent('')
                  setTextTitle('')
                  setLinkUrls('')
                  setUploadResult('')
                  setUploadProgress('')
                  setUploadMode('files')
                }}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
                disabled={uploading}
              >
                Cancel
              </button>
              <button
                onClick={handleUpload}
                disabled={(
                  (uploadMode === 'files' && selectedFiles.length === 0) ||
                  (uploadMode === 'text' && !textContent.trim()) ||
                  (uploadMode === 'links' && !linkUrls.trim())
                ) || uploading}
                className="animated-button bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploadMode === 'files' && (uploading ? 'Uploading...' : `Upload ${selectedFiles.length} File${selectedFiles.length !== 1 ? 's' : ''}`)}
                {uploadMode === 'text' && (uploading ? 'Processing...' : 'Process Text')}
                {uploadMode === 'links' && (uploading ? 'Crawling...' : 'Crawl & Process Links')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Agent Confirmation Modal */}
      {showDeleteConfirm && (
          <div className="modal-overlay fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="modal-content bg-white rounded-lg p-6 w-full max-w-md">
            <h2 className="text-xl font-bold mb-4 text-red-600">Delete Agent</h2>
            <div className="mb-6">
              <p className="text-gray-700 mb-2">
                Are you sure you want to delete the agent <strong>"{agentToDelete?.name}"</strong>?
              </p>
              <p className="text-sm text-red-600">
                This action cannot be undone. All uploaded files, conversations, and knowledge base data for this agent will be permanently deleted.
              </p>
            </div>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={cancelDeleteAgent}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
                disabled={deleting}
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteAgent}
                disabled={deleting}
                className="animated-button bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {deleting ? 'Deleting...' : 'Delete Agent'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function ProtectedChatPage() {
  return (
    <ProtectedRoute>
      <MainLayout>
        <ChatPage />
      </MainLayout>
    </ProtectedRoute>
  )
}