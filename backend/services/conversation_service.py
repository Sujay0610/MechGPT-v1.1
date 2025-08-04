import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from config.supabase_client import get_supabase_client
from models.schemas import Conversation, ConversationMessage, ConversationHistory

class ConversationService:
    def __init__(self):
        self.supabase = get_supabase_client()
        print("ConversationService initialized with Supabase")
    
    def _get_conversations_for_user(self, user_id: str = None) -> List[Dict[str, Any]]:
        """Get conversations from Supabase, optionally filtered by user_id"""
        try:
            query = self.supabase.table("conversations").select("*")
            if user_id:
                query = query.eq("user_id", user_id)
            result = query.order("updated_at", desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting conversations: {e}")
            return []
    
    def _generate_conversation_title(self, first_message: str) -> str:
        """Generate a title for the conversation based on the first message"""
        # Take first 50 characters and clean up
        title = first_message.strip()[:50]
        if len(first_message) > 50:
            title += "..."
        return title
    
    async def create_conversation(self, agent_name: str, first_message: str, user_id: str = None) -> str:
        """Create a new conversation and return its ID"""
        try:
            conversation_id = str(uuid.uuid4())
            title = self._generate_conversation_title(first_message)
            
            conversation_data = {
                "id": conversation_id,
                "user_id": user_id,
                "agent_name": agent_name,
                "title": title,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("conversations").insert(conversation_data).execute()
            
            if result.data:
                print(f"Created conversation '{conversation_id}' for agent '{agent_name}'")
                return conversation_id
            else:
                raise Exception("Failed to create conversation in database")
            
        except Exception as e:
            print(f"Error creating conversation: {e}")
            raise
    
    async def add_message(self, conversation_id: str, text: str, sender: str, agent_name: str = None, user_id: str = None) -> str:
        """Add a message to a conversation"""
        try:
            # Check if conversation exists
            conv_result = self.supabase.table("conversations").select("id").eq("id", conversation_id).execute()
            if not conv_result.data:
                raise ValueError(f"Conversation '{conversation_id}' not found")
            
            message_id = str(uuid.uuid4())
            message_data = {
                "id": message_id,
                "conversation_id": conversation_id,
                "text": text,
                "sender": sender,
                "agent_name": agent_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Insert message
            result = self.supabase.table("messages").insert(message_data).execute()
            
            if result.data:
                # Update conversation's updated_at timestamp
                self.supabase.table("conversations").update({
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", conversation_id).execute()
                
                return message_id
            else:
                raise Exception("Failed to add message to database")
            
        except Exception as e:
            print(f"Error adding message: {e}")
            raise
    
    async def get_conversation_history(self, conversation_id: str, user_id: str = None) -> Optional[ConversationHistory]:
        """Get full conversation history"""
        try:
            # Get conversation data
            conv_query = self.supabase.table("conversations").select("*").eq("id", conversation_id)
            
            conv_result = conv_query.execute()
            if not conv_result.data:
                return None
            
            conversation_data = conv_result.data[0]
            
            # Get messages
            msg_query = self.supabase.table("messages").select("*").eq("conversation_id", conversation_id)
            
            msg_result = msg_query.order("timestamp").execute()
            conversation_messages = msg_result.data if msg_result.data else []
            
            # Convert to schema objects
            conversation = Conversation(**conversation_data)
            message_objects = []
            for msg in conversation_messages:
                # Map database fields to schema fields
                msg_data = {
                    "id": msg["id"],
                    "text": msg["text"],
                    "sender": msg["sender"],
                    "timestamp": msg["timestamp"],
                    "agent_name": msg["agent_name"],
                    "conversation_id": msg["conversation_id"]
                }
                message_objects.append(ConversationMessage(**msg_data))
            
            return ConversationHistory(
                conversation=conversation,
                messages=message_objects
            )
            
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return None
    
    async def get_agent_conversations(self, agent_name: str, user_id: str = None) -> List[Conversation]:
        """Get all conversations for a specific agent"""
        try:
            query = self.supabase.table("conversations").select("*").eq("agent_name", agent_name)
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.order("updated_at", desc=True).execute()
            
            agent_conversations = []
            for conv_data in result.data if result.data else []:
                agent_conversations.append(Conversation(**conv_data))
            
            return agent_conversations
            
        except Exception as e:
            print(f"Error getting agent conversations: {e}")
            return []
    
    async def delete_conversation(self, conversation_id: str, user_id: str = None) -> bool:
        """Delete a conversation and all its messages"""
        try:
            # Check if conversation exists
            conv_query = self.supabase.table("conversations").select("id").eq("id", conversation_id)
            
            conv_result = conv_query.execute()
            if not conv_result.data:
                return False
            
            # Delete all messages first
            msg_query = self.supabase.table("messages").delete().eq("conversation_id", conversation_id)
            msg_query.execute()
            
            # Delete conversation
            conv_delete_query = self.supabase.table("conversations").delete().eq("id", conversation_id)
            conv_delete_query.execute()
            
            print(f"Deleted conversation '{conversation_id}'")
            return True
            
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False
    
    async def get_conversation_messages(self, conversation_id: str, user_id: str = None) -> List[ConversationMessage]:
        """Get all messages for a conversation"""
        try:
            query = self.supabase.table("messages").select("*").eq("conversation_id", conversation_id)
            
            result = query.order("timestamp").execute()
            conversation_messages = result.data if result.data else []
            
            message_objects = []
            for msg in conversation_messages:
                # Map database fields to schema fields
                msg_data = {
                    "id": msg["id"],
                    "text": msg["text"],
                    "sender": msg["sender"],
                    "timestamp": msg["timestamp"],
                    "agent_name": msg["agent_name"],
                    "conversation_id": msg["conversation_id"]
                }
                message_objects.append(ConversationMessage(**msg_data))
            
            return message_objects
            
        except Exception as e:
            print(f"Error getting conversation messages: {e}")
            return []
    
    async def delete_agent_conversations(self, agent_name: str, user_id: str = None) -> int:
        """Delete all conversations for a specific agent"""
        try:
            # Get all conversations for this agent
            query = self.supabase.table("conversations").select("id").eq("agent_name", agent_name)
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            conversation_ids = [conv["id"] for conv in result.data] if result.data else []
            
            deleted_count = 0
            for conversation_id in conversation_ids:
                # Delete all messages for this conversation
                self.supabase.table("messages").delete().eq("conversation_id", conversation_id).execute()
                
                # Delete the conversation
                self.supabase.table("conversations").delete().eq("id", conversation_id).execute()
                deleted_count += 1
            
            if deleted_count > 0:
                print(f"Deleted {deleted_count} conversations for agent '{agent_name}'")
            
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting agent conversations: {e}")
            return 0