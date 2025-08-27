# Document management module for handling multiple document sets
from typing import Dict, List, Optional, Any
from langchain.schema import Document
from langchain_community.vectorstores import FAISS  

import logging
import os
import json
import pickle
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class DocumentSet:
    """Represents a collection of documents with metadata."""
    
    def __init__(self, name: str, description: str = "", source_type: str = "unknown"):
        self.name = name
        self.description = description
        self.source_type = source_type
        self.documents: List[Document] = []
        self.metadata: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.document_count = 0
        self.total_tokens = 0
    
    def add_documents(self, documents: List[Document]):
        """Add documents to this set."""
        self.documents.extend(documents)
        self.document_count = len(self.documents)
        self.updated_at = datetime.now()
        logger.info(f"Added {len(documents)} documents to set '{self.name}'")
    
    def remove_documents(self, indices: List[int]):
        """Remove documents by indices."""
        # Sort indices in descending order to avoid index shifting
        indices.sort(reverse=True)
        for idx in indices:
            if 0 <= idx < len(self.documents):
                del self.documents[idx]
        
        self.document_count = len(self.documents)
        self.updated_at = datetime.now()
        logger.info(f"Removed {len(indices)} documents from set '{self.name}'")
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get a summary of this document set."""
        return {
            "name": self.name,
            "description": self.description,
            "source_type": self.source_type,
            "document_count": self.document_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_tokens": self.total_tokens,
            "metadata": self.metadata
        }
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Document]:
        """Simple text-based search within documents."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            if query_lower in doc.page_content.lower():
                results.append(doc)
                if len(results) >= max_results:
                    break
        
        return results

class DocumentManager:
    """Manages multiple document sets and their FAISS indices."""
    
    def __init__(self, storage_dir: str = "document_sets"):
        self.storage_dir = storage_dir
        self.document_sets: Dict[str, DocumentSet] = {}
        self.faiss_indices: Dict[str, FAISS] = {}
        self.active_set: Optional[str] = None
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing document sets
        self._load_document_sets()
    
    def create_document_set(self, name: str, description: str = "", 
                           source_type: str = "unknown") -> str:
        """Create a new document set."""
        if name in self.document_sets:
            raise ValueError(f"Document set '{name}' already exists")
        
        doc_set = DocumentSet(name, description, source_type)
        self.document_sets[name] = doc_set
        
        # Save to disk
        self._save_document_set(name)
        logger.info(f"Created new document set: '{name}'")
        
        return name
    
    def delete_document_set(self, name: str) -> bool:
        """Delete a document set and its associated files."""
        if name not in self.document_sets:
            return False
        
        try:
            # Remove from memory
            del self.document_sets[name]
            
            # Remove FAISS index if exists
            if name in self.faiss_indices:
                del self.faiss_indices[name]
            
            # Remove from disk
            set_dir = os.path.join(self.storage_dir, name)
            if os.path.exists(set_dir):
                import shutil
                shutil.rmtree(set_dir)
            
            # Update active set if necessary
            if self.active_set == name:
                self.active_set = None
                if self.document_sets:
                    self.active_set = next(iter(self.document_sets.keys()))
            
            logger.info(f"Deleted document set: '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document set '{name}': {e}")
            return False
    
    def add_documents_to_set(self, set_name: str, documents: List[Document], 
                            embeddings_model) -> bool:
        """Add documents to a specific set and update its FAISS index."""
        if set_name not in self.document_sets:
            raise ValueError(f"Document set '{set_name}' does not exist")
        
        try:
            doc_set = self.document_sets[set_name]
            doc_set.add_documents(documents)
            
            # Update or create FAISS index
            if set_name in self.faiss_indices:
                # Add to existing index
                self.faiss_indices[set_name] = self._add_to_faiss_index(
                    self.faiss_indices[set_name], documents, embeddings_model
                )
            else:
                # Create new index
                self.faiss_indices[set_name] = self._create_faiss_index(
                    documents, embeddings_model, set_name
                )
            
            # Save updated set
            self._save_document_set(set_name)
            self._save_faiss_index(set_name)
            
            logger.info(f"Added {len(documents)} documents to set '{set_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to set '{set_name}': {e}")
            return False
    
    def switch_document_set(self, set_name: str) -> bool:
        """Switch to a different document set as the active set."""
        if set_name not in self.document_sets:
            logger.error(f"Cannot switch to non-existent document set: '{set_name}'")
            return False
        
        self.active_set = set_name
        logger.info(f"Switched to document set: '{set_name}'")
        return True
    
    def get_active_set(self) -> Optional[DocumentSet]:
        """Get the currently active document set."""
        if self.active_set and self.active_set in self.document_sets:
            return self.document_sets[self.active_set]
        return None
    
    def get_active_faiss_index(self) -> Optional[FAISS]:
        """Get the FAISS index for the active document set."""
        if self.active_set and self.active_set in self.faiss_indices:
            return self.faiss_indices[self.active_set]
        return None
    
    def list_document_sets(self) -> List[Dict[str, Any]]:
        """List all available document sets with summaries."""
        return [doc_set.get_document_summary() 
                for doc_set in self.document_sets.values()]
    
    def get_document_set_info(self, set_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document set."""
        if set_name in self.document_sets:
            return self.document_sets[set_name].get_document_summary()
        return None
    
    def _create_faiss_index(self, documents: List[Document], embeddings_model, 
                           set_name: str) -> FAISS:
        """Create a new FAISS index for documents."""
        try:
            from core.retriever import build_faiss_index
            index = build_faiss_index(embeddings_model, documents, set_name)
            logger.info(f"Created FAISS index for set '{set_name}'")
            return index
        except Exception as e:
            logger.error(f"Error creating FAISS index for set '{set_name}': {e}")
            raise
    
    def _add_to_faiss_index(self, existing_index: FAISS, new_documents: List[Document], 
                           embeddings_model) -> FAISS:
        """Add new documents to an existing FAISS index."""
        try:
            from core.retriever import add_documents_to_index
            updated_index = add_documents_to_index(existing_index, new_documents, embeddings_model)
            logger.info(f"Updated FAISS index with {len(new_documents)} new documents")
            return updated_index
        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}")
            raise
    
    def _save_document_set(self, set_name: str):
        """Save document set metadata to disk."""
        try:
            set_dir = os.path.join(self.storage_dir, set_name)
            os.makedirs(set_dir, exist_ok=True)
            
            doc_set = self.document_sets[set_name]
            metadata_file = os.path.join(set_dir, "metadata.json")
            
            with open(metadata_file, 'w') as f:
                json.dump(doc_set.get_document_summary(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving document set '{set_name}': {e}")
    
    def _save_faiss_index(self, set_name: str):
        """Save FAISS index to disk."""
        try:
            if set_name in self.faiss_indices:
                set_dir = os.path.join(self.storage_dir, set_name)
                os.makedirs(set_dir, exist_ok=True)
                
                index_path = os.path.join(set_dir, "faiss_index")
                self.faiss_indices[set_name].save_local(index_path)
                
        except Exception as e:
            logger.error(f"Error saving FAISS index for set '{set_name}': {e}")
    
    def _load_document_sets(self):
        """Load existing document sets from disk."""
        try:
            if not os.path.exists(self.storage_dir):
                return
            
            for item in os.listdir(self.storage_dir):
                item_path = os.path.join(self.storage_dir, item)
                if os.path.isdir(item_path):
                    metadata_file = os.path.join(item_path, "metadata.json")
                    
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Recreate document set
                        doc_set = DocumentSet(
                            metadata["name"],
                            metadata["description"],
                            metadata["source_type"]
                        )
                        doc_set.created_at = datetime.fromisoformat(metadata["created_at"])
                        doc_set.updated_at = datetime.fromisoformat(metadata["updated_at"])
                        doc_set.document_count = metadata["document_count"]
                        doc_set.total_tokens = metadata["total_tokens"]
                        doc_set.metadata = metadata["metadata"]
                        
                        self.document_sets[item] = doc_set
                        
                        # Try to load FAISS index
                        index_path = os.path.join(item_path, "faiss_index")
                        if os.path.exists(index_path):
                            try:
                                from core.embeddings import get_embedding_model
                                embeddings = get_embedding_model()
                                self.faiss_indices[item] = FAISS.load_local(index_path, embeddings)
                                logger.info(f"Loaded FAISS index for set '{item}'")
                            except Exception as e:
                                logger.warning(f"Could not load FAISS index for set '{item}': {e}")
            
            # Set first available set as active
            if self.document_sets and not self.active_set:
                self.active_set = next(iter(self.document_sets.keys()))
                
        except Exception as e:
            logger.error(f"Error loading document sets: {e}")
    
    def export_document_set(self, set_name: str, export_path: str) -> bool:
        """Export a document set to a specified location."""
        if set_name not in self.document_sets:
            return False
        
        try:
            doc_set = self.document_sets[set_name]
            
            # Create export data
            export_data = {
                "document_set": doc_set.get_document_summary(),
                "documents": [{"content": doc.page_content, "metadata": doc.metadata} 
                             for doc in doc_set.documents]
            }
            
            # Save export file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported document set '{set_name}' to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting document set '{set_name}': {e}")
            return False
    
    def import_document_set(self, import_path: str, set_name: str = None) -> str:
        """Import a document set from a file."""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            # Generate set name if not provided
            if not set_name:
                base_name = os.path.splitext(os.path.basename(import_path))[0]
                set_name = f"imported_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create document set
            doc_set = DocumentSet(
                set_name,
                import_data["document_set"]["description"],
                import_data["document_set"]["source_type"]
            )
            
            # Add documents
            for doc_data in import_data["documents"]:
                doc = Document(
                    page_content=doc_data["content"],
                    metadata=doc_data["metadata"]
                )
                doc_set.documents.append(doc)
            
            doc_set.document_count = len(doc_set.documents)
            self.document_sets[set_name] = doc_set
            
            # Save to disk
            self._save_document_set(set_name)
            
            logger.info(f"Imported document set '{set_name}' with {doc_set.document_count} documents")
            return set_name
            
        except Exception as e:
            logger.error(f"Error importing document set: {e}")
            raise
