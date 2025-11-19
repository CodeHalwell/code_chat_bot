"""Database management for chat history and application data."""
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime
import json
import os

try:
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

from ..models import ChatMessage


class BaseDatabaseProvider(ABC):
    """Base class for database providers."""

    @abstractmethod
    def save_chat_session(self, session_id: str, messages: List[ChatMessage], metadata: Dict[str, Any]) -> bool:
        """Save a chat session to the database."""
        pass

    @abstractmethod
    def load_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session from the database."""
        pass

    @abstractmethod
    def list_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all chat sessions."""
        pass

    @abstractmethod
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        pass


class JSONDatabaseProvider(BaseDatabaseProvider):
    """JSON file-based database provider (fallback)."""

    def __init__(self, data_dir: str = "previous_chats"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def save_chat_session(self, session_id: str, messages: List[ChatMessage], metadata: Dict[str, Any]) -> bool:
        """Save a chat session to a JSON file."""
        try:
            file_path = os.path.join(self.data_dir, f"{session_id}.json")
            data = {
                "session_id": session_id,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving chat session: {e}")
            return False

    def load_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session from a JSON file."""
        try:
            file_path = os.path.join(self.data_dir, f"{session_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading chat session: {e}")
        return None

    def list_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all chat sessions from JSON files."""
        sessions = []
        try:
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)), reverse=True)

            for filename in files[:limit]:
                session_id = filename[:-5]  # Remove .json extension
                session = self.load_chat_session(session_id)
                if session:
                    sessions.append({
                        "session_id": session_id,
                        "timestamp": session.get("timestamp"),
                        "message_count": len(session.get("messages", []))
                    })
        except Exception as e:
            print(f"Error listing chat sessions: {e}")
        return sessions

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session JSON file."""
        try:
            file_path = os.path.join(self.data_dir, f"{session_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            print(f"Error deleting chat session: {e}")
        return False


class MongoDBProvider(BaseDatabaseProvider):
    """MongoDB database provider."""

    def __init__(self, connection_string: str, database_name: str = "chatbot_db"):
        if not PYMONGO_AVAILABLE:
            raise ImportError("pymongo is required for MongoDB support")

        self.client = pymongo.MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db["chat_sessions"]

    def save_chat_session(self, session_id: str, messages: List[ChatMessage], metadata: Dict[str, Any]) -> bool:
        """Save a chat session to MongoDB."""
        try:
            document = {
                "session_id": session_id,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "metadata": metadata,
                "timestamp": datetime.now()
            }
            self.collection.update_one(
                {"session_id": session_id},
                {"$set": document},
                upsert=True
            )
            return True
        except Exception as e:
            print(f"Error saving chat session to MongoDB: {e}")
            return False

    def load_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session from MongoDB."""
        try:
            result = self.collection.find_one({"session_id": session_id}, {"_id": 0})
            if result and "timestamp" in result:
                result["timestamp"] = result["timestamp"].isoformat()
            return result
        except Exception as e:
            print(f"Error loading chat session from MongoDB: {e}")
            return None

    def list_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all chat sessions from MongoDB."""
        try:
            sessions = list(
                self.collection.find(
                    {},
                    {"session_id": 1, "timestamp": 1, "_id": 0}
                ).sort("timestamp", -1).limit(limit)
            )
            for session in sessions:
                if "timestamp" in session:
                    session["timestamp"] = session["timestamp"].isoformat()
                # Get message count
                full_session = self.collection.find_one({"session_id": session["session_id"]})
                session["message_count"] = len(full_session.get("messages", [])) if full_session else 0
            return sessions
        except Exception as e:
            print(f"Error listing chat sessions from MongoDB: {e}")
            return []

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session from MongoDB."""
        try:
            result = self.collection.delete_one({"session_id": session_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting chat session from MongoDB: {e}")
            return False


class PostgreSQLProvider(BaseDatabaseProvider):
    """PostgreSQL database provider."""

    def __init__(self, connection_params: Dict[str, str]):
        if not PSYCOPG2_AVAILABLE:
            raise ImportError("psycopg2-binary is required for PostgreSQL support")

        self.connection_params = connection_params
        self._create_tables()

    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(**self.connection_params)

    def _create_tables(self):
        """Create necessary tables if they don't exist."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    messages JSONB NOT NULL,
                    metadata JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error creating PostgreSQL tables: {e}")

    def save_chat_session(self, session_id: str, messages: List[ChatMessage], metadata: Dict[str, Any]) -> bool:
        """Save a chat session to PostgreSQL."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            messages_json = json.dumps([{"role": msg.role, "content": msg.content} for msg in messages])
            metadata_json = json.dumps(metadata)

            cursor.execute("""
                INSERT INTO chat_sessions (session_id, messages, metadata, timestamp)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (session_id)
                DO UPDATE SET messages = %s, metadata = %s, timestamp = %s
            """, (session_id, messages_json, metadata_json, datetime.now(),
                  messages_json, metadata_json, datetime.now()))

            conn.commit()
            cursor.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving chat session to PostgreSQL: {e}")
            return False

    def load_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session from PostgreSQL."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT * FROM chat_sessions WHERE session_id = %s",
                (session_id,)
            )
            result = cursor.fetchone()
            cursor.close()
            conn.close()

            if result:
                return {
                    "session_id": result["session_id"],
                    "messages": result["messages"],
                    "metadata": result["metadata"],
                    "timestamp": result["timestamp"].isoformat()
                }
        except Exception as e:
            print(f"Error loading chat session from PostgreSQL: {e}")
        return None

    def list_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all chat sessions from PostgreSQL."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT session_id, timestamp,
                       jsonb_array_length(messages) as message_count
                FROM chat_sessions
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return [
                {
                    "session_id": row["session_id"],
                    "timestamp": row["timestamp"].isoformat(),
                    "message_count": row["message_count"]
                }
                for row in results
            ]
        except Exception as e:
            print(f"Error listing chat sessions from PostgreSQL: {e}")
            return []

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session from PostgreSQL."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_sessions WHERE session_id = %s", (session_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            cursor.close()
            conn.close()
            return deleted
        except Exception as e:
            print(f"Error deleting chat session from PostgreSQL: {e}")
            return False


class FirebaseProvider(BaseDatabaseProvider):
    """Firebase Firestore database provider."""

    def __init__(self, credentials_path: str, collection_name: str = "chat_sessions"):
        if not FIREBASE_AVAILABLE:
            raise ImportError("firebase-admin is required for Firebase support")

        if not firebase_admin._apps:
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self.collection = self.db.collection(collection_name)

    def save_chat_session(self, session_id: str, messages: List[ChatMessage], metadata: Dict[str, Any]) -> bool:
        """Save a chat session to Firebase."""
        try:
            doc = {
                "session_id": session_id,
                "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                "metadata": metadata,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            self.collection.document(session_id).set(doc)
            return True
        except Exception as e:
            print(f"Error saving chat session to Firebase: {e}")
            return False

    def load_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a chat session from Firebase."""
        try:
            doc = self.collection.document(session_id).get()
            if doc.exists:
                data = doc.to_dict()
                if "timestamp" in data and data["timestamp"]:
                    data["timestamp"] = data["timestamp"].isoformat()
                return data
        except Exception as e:
            print(f"Error loading chat session from Firebase: {e}")
        return None

    def list_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all chat sessions from Firebase."""
        try:
            docs = self.collection.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
            sessions = []
            for doc in docs:
                data = doc.to_dict()
                sessions.append({
                    "session_id": data.get("session_id"),
                    "timestamp": data["timestamp"].isoformat() if data.get("timestamp") else None,
                    "message_count": len(data.get("messages", []))
                })
            return sessions
        except Exception as e:
            print(f"Error listing chat sessions from Firebase: {e}")
            return []

    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session from Firebase."""
        try:
            self.collection.document(session_id).delete()
            return True
        except Exception as e:
            print(f"Error deleting chat session from Firebase: {e}")
            return False


def get_database_provider(
    provider_type: str = "json",
    **kwargs
) -> BaseDatabaseProvider:
    """
    Factory function to get the appropriate database provider.

    Args:
        provider_type: Type of database ('json', 'mongodb', 'postgresql', 'firebase')
        **kwargs: Provider-specific configuration

    Returns:
        Database provider instance
    """
    if provider_type == "json":
        return JSONDatabaseProvider(kwargs.get("data_dir", "previous_chats"))
    elif provider_type == "mongodb":
        if not PYMONGO_AVAILABLE:
            print("MongoDB not available, falling back to JSON")
            return JSONDatabaseProvider()
        return MongoDBProvider(
            kwargs.get("connection_string", "mongodb://localhost:27017/"),
            kwargs.get("database_name", "chatbot_db")
        )
    elif provider_type == "postgresql":
        if not PSYCOPG2_AVAILABLE:
            print("PostgreSQL not available, falling back to JSON")
            return JSONDatabaseProvider()
        return PostgreSQLProvider(kwargs.get("connection_params", {}))
    elif provider_type == "firebase":
        if not FIREBASE_AVAILABLE:
            print("Firebase not available, falling back to JSON")
            return JSONDatabaseProvider()
        return FirebaseProvider(
            kwargs.get("credentials_path"),
            kwargs.get("collection_name", "chat_sessions")
        )
    else:
        raise ValueError(f"Unsupported database provider: {provider_type}")
