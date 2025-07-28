import os
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from models.schemas import Conversation, ConversationMessage, ConversationHistory

class ConversationService:
    def __init__(self):
        self.conversations_file = Path("./data/conversations.json")
        self.messages_file = Path("./data/messages.json")
        self.conversations_file.parent.mkdir(exist_ok=True)
        
        print("Conversation service initialized")
    
    def _load_conversations(self) -> Dict[str, Any]:
        """Load conversations metadata from file"""
        if self.conversations_file.exists():
            try:
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading conversations: {e}")
        return {}
    
    def _save_conversations(self, conversations: Dict[str, Any]):
        """Save conversations metadata to file"""
        try:
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving conversations: {e}")
    
    def _load_messages(self) -> Dict[str, Any]:
        """Load messages from file"""
        if self.messages_file.exists():
            try:
                with open(self.messages_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading messages: {e}")
        return {}
    
    def _save_messages(self, messages: Dict[str, Any]):
        """Save messages to file"""
        try:
            with open(self.messages_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving messages: {e}")
    
    def _generate_conversation_title(self, first_message: str) -> str:
        """Generate a title for the conversation based on the first message"""
        # Take first 50 characters and clean up
        title = first_message.strip()[:50]
        if len(first_message) > 50:
            title += "..."
        return title
    
    async def create_conversation(self, agent_name: str, first_message: str) -> str:
        """Create a new conversation and return its ID"""
        try:
            conversations = self._load_conversations()
            
            conversation_id = str(uuid.uuid4())
            title = self._generate_conversation_title(first_message)
            
            conversation_data = {
                "id": conversation_id,
                "agent_name": agent_name,
                "title": title,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "message_count": 0
            }
            
            conversations[conversation_id] = conversation_data
            self._save_conversations(conversations)
            
            print(f"Created conversation '{conversation_id}' for agent '{agent_name}'")
            return conversation_id
            
        except Exception as e:
            print(f"Error creating conversation: {e}")
            raise
    
    async def add_message(self, conversation_id: str, text: str, sender: str, agent_name: str) -> str:
        """Add a message to a conversation"""
        try:
            conversations = self._load_conversations()
            messages = self._load_messages()
            
            if conversation_id not in conversations:
                raise ValueError(f"Conversation '{conversation_id}' not found")
            
            message_id = str(uuid.uuid4())
            message_data = {
                "id": message_id,
                "text": text,
                "sender": sender,
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent_name,
                "conversation_id": conversation_id
            }
            
            # Initialize conversation messages if not exists
            if conversation_id not in messages:
                messages[conversation_id] = []
            
            messages[conversation_id].append(message_data)
            self._save_messages(messages)
            
            # Update conversation metadata
            conversations[conversation_id]["updated_at"] = datetime.now().isoformat()
            conversations[conversation_id]["message_count"] = len(messages[conversation_id])
            self._save_conversations(conversations)
            
            return message_id
            
        except Exception as e:
            print(f"Error adding message: {e}")
            raise
    
    async def get_conversation_history(self, conversation_id: str) -> Optional[ConversationHistory]:
        """Get full conversation history"""
        try:
            conversations = self._load_conversations()
            messages = self._load_messages()
            
            if conversation_id not in conversations:
                return None
            
            conversation_data = conversations[conversation_id]
            conversation_messages = messages.get(conversation_id, [])
            
            conversation = Conversation(**conversation_data)
            message_objects = [ConversationMessage(**msg) for msg in conversation_messages]
            
            return ConversationHistory(
                conversation=conversation,
                messages=message_objects
            )
            
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return None
    
    async def get_agent_conversations(self, agent_name: str) -> List[Conversation]:
        """Get all conversations for a specific agent"""
        try:
            conversations = self._load_conversations()
            
            agent_conversations = []
            for conv_data in conversations.values():
                if conv_data.get("agent_name") == agent_name:
                    agent_conversations.append(Conversation(**conv_data))
            
            # Sort by updated_at descending (most recent first)
            agent_conversations.sort(key=lambda x: x.updated_at, reverse=True)
            
            return agent_conversations
            
        except Exception as e:
            print(f"Error getting agent conversations: {e}")
            return []
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        try:
            conversations = self._load_conversations()
            messages = self._load_messages()
            
            if conversation_id not in conversations:
                return False
            
            # Remove conversation
            del conversations[conversation_id]
            self._save_conversations(conversations)
            
            # Remove messages
            if conversation_id in messages:
                del messages[conversation_id]
                self._save_messages(messages)
            
            print(f"Deleted conversation '{conversation_id}'")
            return True
            
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False
    
    async def get_conversation_messages(self, conversation_id: str) -> List[ConversationMessage]:
        """Get messages for a specific conversation"""
        try:
            messages = self._load_messages()
            conversation_messages = messages.get(conversation_id, [])
            
            return [ConversationMessage(**msg) for msg in conversation_messages]
            
        except Exception as e:
            print(f"Error getting conversation messages: {e}")
            return []