import gradio as gr
import os
import tempfile
import io
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re

# Cloud storage imports
try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    print("⚠️ Dropbox SDK not installed. Install with: pip install dropbox")

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.http import MediaIoBaseDownload
    import pickle
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    print("⚠️ Google Drive API not installed. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# Try to import langdetect, but make it optional
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("⚠️ langdetect not installed. Multilingual features will be disabled.")

class CloudStorageManager:
    def __init__(self):
        self.dropbox_client = None
        self.gdrive_service = None
        self.temp_dir = tempfile.mkdtemp()
        
    def setup_dropbox(self, access_token):
        """Setup Dropbox client with access token"""
        if not DROPBOX_AVAILABLE:
            return "❌ Dropbox SDK not installed. Install with: pip install dropbox"
        
        if not access_token.strip():
            return "❌ Please provide Dropbox access token"
        
        try:
            self.dropbox_client = dropbox.Dropbox(access_token)
            # Test the connection
            account_info = self.dropbox_client.users_get_current_account()
            return f"✅ Connected to Dropbox as {account_info.name.display_name}"
        except AuthError:
            return "❌ Invalid Dropbox access token"
        except Exception as e:
            return f"❌ Dropbox connection error: {str(e)}"
    
    def setup_gdrive(self, credentials_file):
        """Setup Google Drive service with credentials file"""
        if not GDRIVE_AVAILABLE:
            return "❌ Google Drive API not installed. Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        
        if not credentials_file:
            return "❌ Please upload Google Drive credentials JSON file"
        
        try:
            import json
            from google.auth.transport.requests import Request
            
            # Read the uploaded credentials file
            if hasattr(credentials_file, 'name'):
                credentials_path = credentials_file.name
            else:
                credentials_path = credentials_file
            
            # Load credentials from JSON file
            with open(credentials_path, 'r') as f:
                credentials_info = json.load(f)
            
            # For service account credentials
            if 'type' in credentials_info and credentials_info['type'] == 'service_account':
                from google.oauth2 import service_account
                
                creds = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                
                self.gdrive_service = build('drive', 'v3', credentials=creds)
                return "✅ Connected to Google Drive successfully with service account!"
            
            # For OAuth2 credentials - use the out-of-band flow
            else:
                flow = Flow.from_client_secrets_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                
                # Use the out-of-band (OOB) redirect URI
                flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                
                # Generate authorization URL
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true',
                    prompt='consent'
                )
                
                # Store the flow for later use
                self._temp_flow = flow
                
                return f"""📋 **Google Drive Setup - Step 1 Complete**

**Next Steps:**
1. **Click this authorization link:** {auth_url}

2. **Sign in** to your Google account and **allow permissions**

3. **Copy the authorization code** that appears on the page

4. **Paste the code** in the "Authorization Code" field below and click "Complete Setup"

**Troubleshooting:**
- If you get a redirect error, make sure you've added `urn:ietf:wg:oauth:2.0:oob` as a redirect URI in your Google Cloud Console
- Or simply copy the code from the browser address bar after authorization"""
            
        except FileNotFoundError:
            return "❌ Credentials file not found. Please upload a valid JSON file."
        except json.JSONDecodeError:
            return "❌ Invalid JSON file. Please upload a valid Google credentials file."
        except Exception as e:
            return f"❌ Google Drive setup error: {str(e)}"
    
    def setup_gdrive_with_code(self, credentials_file, auth_code):
        """Complete Google Drive setup with authorization code"""
        if not GDRIVE_AVAILABLE:
            return "❌ Google Drive API not installed."
        
        if not auth_code or not auth_code.strip():
            return "❌ Please provide the authorization code from Google"
        
        try:
            # Use the stored flow from the previous step
            if not hasattr(self, '_temp_flow'):
                return "❌ Please run the initial setup first by clicking 'Setup Google Drive'"
            
            flow = self._temp_flow
            
            # Exchange authorization code for access token
            flow.fetch_token(code=auth_code.strip())
            creds = flow.credentials
            
            # Build the service
            self.gdrive_service = build('drive', 'v3', credentials=creds)
            
            # Test the connection by listing files
            try:
                results = self.gdrive_service.files().list(pageSize=1).execute()
                files = results.get('files', [])
                
                # Clean up the temporary flow
                delattr(self, '_temp_flow')
                
                return "✅ Google Drive connected successfully! You can now list and process PDF files."
                
            except Exception as test_error:
                return f"❌ Connection test failed: {str(test_error)}"
            
        except Exception as e:
            error_msg = str(e)
            if "invalid_grant" in error_msg:
                return "❌ Invalid or expired authorization code. Please try the setup process again."
            elif "invalid_request" in error_msg:
                return "❌ Invalid authorization code format. Please copy the entire code."
            else:
                return f"❌ Error completing Google Drive setup: {error_msg}"
    
    def list_dropbox_files(self, folder_path="", file_extension=".pdf"):
        """List PDF files in Dropbox folder"""
        if not self.dropbox_client:
            return []
        
        try:
            files = []
            result = self.dropbox_client.files_list_folder(folder_path)
            
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    if entry.name.lower().endswith(file_extension.lower()):
                        files.append({
                            'name': entry.name,
                            'path': entry.path_lower,
                            'size': entry.size,
                            'modified': entry.client_modified.strftime('%Y-%m-%d %H:%M:%S') if entry.client_modified else 'Unknown',
                            'id': entry.path_lower  # Use path as ID for consistency
                        })
            
            return files
        except Exception as e:
            print(f"Error listing Dropbox files: {e}")
            return []
    
    def list_gdrive_files(self, folder_id=None, file_extension=".pdf"):
        """List PDF files in Google Drive folder"""
        if not self.gdrive_service:
            return []
        
        try:
            query = f"mimeType='application/pdf'"
            if folder_id:
                query += f" and '{folder_id}' in parents"
            
            results = self.gdrive_service.files().list(
                q=query,
                fields="files(id, name, size, modifiedTime)"
            ).execute()
            
            files = []
            for file in results.get('files', []):
                files.append({
                    'id': file['id'],
                    'name': file['name'],
                    'size': int(file.get('size', 0)),
                    'modified': file.get('modifiedTime', 'Unknown')
                })
            
            return files
        except Exception as e:
            print(f"Error listing Google Drive files: {e}")
            return []
    
    def download_dropbox_file(self, file_path):
        """Download file from Dropbox to temporary location"""
        if not self.dropbox_client:
            return None
        
        try:
            local_path = os.path.join(self.temp_dir, os.path.basename(file_path))
            with open(local_path, 'wb') as f:
                metadata, response = self.dropbox_client.files_download(file_path)
                f.write(response.content)
            return local_path
        except Exception as e:
            print(f"Error downloading from Dropbox: {e}")
            return None
    
    def download_gdrive_file(self, file_id, file_name):
        """Download file from Google Drive to temporary location"""
        if not self.gdrive_service:
            return None
        
        try:
            request = self.gdrive_service.files().get_media(fileId=file_id)
            local_path = os.path.join(self.temp_dir, file_name)
            
            with io.FileIO(local_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
            
            return local_path
        except Exception as e:
            print(f"Error downloading from Google Drive: {e}")
            return None
    
    def cleanup_temp_files(self):
        """Clean up temporary downloaded files"""
        try:
            for file_path in Path(self.temp_dir).glob("*"):
                file_path.unlink()
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

class ConfigurablePDFQASystem:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.retriever = None
        self.embeddings = None
        self.vector_store = None
        self.documents = None
        self.processed_files = []  # Track processed files
        self.multilingual_mode = False  # Default to English-only
        self.cloud_manager = CloudStorageManager()
        
        # Language detection and mapping (only used in multilingual mode)
        self.language_map = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'zh': 'Chinese',
            'it': 'Italian',
            'nl': 'Dutch',
            'tr': 'Turkish'
        }
    
    def setup_dropbox_connection(self, access_token):
        """Setup Dropbox connection and return status"""
        return self.cloud_manager.setup_dropbox(access_token)
    
    def setup_gdrive_connection(self, credentials_file):
        """Setup Google Drive connection and return status"""
        return self.cloud_manager.setup_gdrive(credentials_file)
    
    def complete_gdrive_setup(self, credentials_file, auth_code):
        """Complete Google Drive setup with authorization code"""
        return self.cloud_manager.setup_gdrive_with_code(credentials_file, auth_code)
    
    def get_dropbox_files(self, folder_path=""):
        """Get list of PDF files from Dropbox"""
        files = self.cloud_manager.list_dropbox_files(folder_path)
        if not files:
            return "No PDF files found in Dropbox folder", []
        
        return f"Found {len(files)} PDF files in Dropbox", files
    
    def get_gdrive_files(self, folder_id=""):
        """Get list of PDF files from Google Drive"""
        files = self.cloud_manager.list_gdrive_files(folder_id if folder_id else None)
        if not files:
            return "No PDF files found in Google Drive", []
        
        return f"Found {len(files)} PDF files in Google Drive", files
    
    def process_cloud_files(self, selected_file_names, all_files, source_type, chunk_size, chunk_overlap):
        """Process selected files from cloud storage"""
        if not selected_file_names or not all_files:
            return "❌ No files selected!"
        
        # Find selected files by matching names
        selected_files = []
        for file in all_files:
            if file['name'] in selected_file_names:
                selected_files.append(file)
        
        if not selected_files:
            return "❌ Selected files not found in the file list!"
        
        try:
            # Load embeddings first
            embed_status = self.load_embeddings()
            if "❌" in embed_status:
                return embed_status
            
            all_documents = []
            file_info = []
            total_pages = 0
            
            # Download and process each selected file
            for i, file_info_item in enumerate(selected_files):
                print(f"Processing file {i+1}/{len(selected_files)}: {file_info_item['name']}")
                
                # Download file based on source type
                if source_type == "Dropbox":
                    local_path = self.cloud_manager.download_dropbox_file(file_info_item['path'])
                elif source_type == "Google Drive":
                    local_path = self.cloud_manager.download_gdrive_file(file_info_item['id'], file_info_item['name'])
                else:
                    continue
                
                if not local_path or not os.path.exists(local_path):
                    print(f"Failed to download: {file_info_item['name']}")
                    continue
                
                # Load the PDF using PyMuPDFLoader
                loader = PyMuPDFLoader(local_path)
                documents = loader.load()
                
                # Add source file information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = file_info_item['name']
                    doc.metadata['file_index'] = i
                    doc.metadata['source_type'] = source_type
                
                all_documents.extend(documents)
                file_info.append({
                    'name': file_info_item['name'],
                    'pages': len(documents),
                    'source': source_type
                })
                total_pages += len(documents)
            
            if not all_documents:
                return "❌ No documents could be processed from cloud storage"
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            self.documents = text_splitter.split_documents(all_documents)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            )
            
            self.processed_files = file_info
            chunks_count = len(self.documents)
            
            # Clean up downloaded files
            self.cloud_manager.cleanup_temp_files()
            
            # Success message with cloud source info
            files_summary = ", ".join([f"{info['name']} ({info['source']})" for info in file_info])
            return f"✅ Successfully processed {len(selected_files)} files from {source_type}: {files_summary}\n📊 {total_pages} pages → {chunks_count} searchable chunks"
            
        except Exception as e:
            return f"❌ Error processing cloud files: {str(e)}"
    
    def set_language_mode(self, multilingual_enabled):
        """Set the language mode (English-only or Multilingual)"""
        self.multilingual_mode = multilingual_enabled and LANGDETECT_AVAILABLE
        if multilingual_enabled and not LANGDETECT_AVAILABLE:
            return "⚠️ Multilingual mode requires 'langdetect' package. Install with: pip install langdetect"
        
        mode_text = "Multilingual" if self.multilingual_mode else "English-only"
        return f"✅ Language mode set to: {mode_text}"
    
    def detect_language(self, text):
        """Detect the language of input text and check for Hinglish patterns"""
        if not self.multilingual_mode:
            return 'en', 'English'
        
        try:
            # First check for Hinglish patterns (mix of Hindi and English)
            if self.is_hinglish(text):
                return 'hinglish', 'Hinglish'
            
            # Use langdetect for other languages
            detected_lang = detect(text)
            language_name = self.language_map.get(detected_lang, detected_lang.upper())
            
            return detected_lang, language_name
        except:
            # Default to English if detection fails
            return 'en', 'English'
    
    def is_hinglish(self, text):
        """Check if text contains Hinglish patterns (mix of Hindi and English characters)"""
        if not self.multilingual_mode:
            return False
            
        # Check for presence of both Devanagari and Latin scripts
        has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        
        # Check for common Hinglish words/patterns
        hinglish_patterns = [
            r'\b(kya|hai|aur|ka|ki|ke|main|mein|se|ko|me|hoon|hun|tum|ap|aap)\b',
            r'\b(bharat|india|delhi|mumbai|kolkata|chennai)\b',
            r'\b(rajdhani|rajniti|sarkar|desh|log|sab)\b'
        ]
        
        has_hinglish_words = any(re.search(pattern, text.lower()) for pattern in hinglish_patterns)
        
        return (has_devanagari and has_latin) or has_hinglish_words
    
    def get_language_instruction(self, lang_code, lang_name):
        """Get appropriate language instruction for the model"""
        if not self.multilingual_mode:
            return "Please respond in English."
            
        if lang_code == 'hinglish':
            return "Please respond in Hinglish (a mix of Hindi and English, using Roman script for Hindi words). Use simple Hindi words written in English letters mixed with English words."
        elif lang_code == 'hi':
            return "कृपया हिंदी में उत्तर दें।"
        elif lang_code == 'es':
            return "Por favor responde en español."
        elif lang_code == 'fr':
            return "Veuillez répondre en français."
        elif lang_code == 'de':
            return "Bitte antworten Sie auf Deutsch."
        elif lang_code == 'pt':
            return "Por favor, responda em português."
        elif lang_code == 'ru':
            return "Пожалуйста, отвечайте на русском языке."
        elif lang_code == 'ja':
            return "日本語で回答してください。"
        elif lang_code == 'ko':
            return "한국어로 답변해 주세요."
        elif lang_code == 'ar':
            return "يرجى الإجابة باللغة العربية."
        elif lang_code == 'zh':
            return "请用中文回答。"
        elif lang_code == 'it':
            return "Si prega di rispondere in italiano."
        elif lang_code == 'nl':
            return "Gelieve in het Nederlands te antwoorden."
        elif lang_code == 'tr':
            return "Lütfen Türkçe cevap verin."
        else:
            return "Please respond in English."
        
    def unload_current_model(self):
        """Properly unload the current model from GPU memory"""
        if self.model is not None:
            print(f"Unloading model: {self.current_model_name}")
            # Move model to CPU and delete references
            self.model.cpu()
            del self.model
            self.model = None
            
            # Delete tokenizer reference
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            print("✅ Model unloaded and GPU memory cleared")
    
    def load_model(self, model_name):
        """Load the specified Qwen model and tokenizer with proper GPU memory management"""
        if self.current_model_name != model_name:
            # Unload current model first to free GPU memory
            if self.current_model_name is not None:
                self.unload_current_model()
            
            print(f"Loading model: {model_name}")
            try:
                # Show GPU memory before loading
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(f"GPU memory before loading: {memory_before:.2f} GB")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype="auto", 
                    device_map="auto"
                )
                self.current_model_name = model_name
                
                # Show GPU memory after loading
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_used = memory_after - memory_before
                    lang_status = "Multilingual" if self.multilingual_mode else "English-only"
                    return f"✅ Model {model_name} loaded successfully!\n💾 GPU Memory: {memory_after:.2f} GB (+{memory_used:.2f} GB)\n🌐 Language mode: {lang_status}"
                else:
                    lang_status = "Multilingual" if self.multilingual_mode else "English-only"
                    return f"✅ Model {model_name} loaded successfully! (CPU mode)\n🌐 Language mode: {lang_status}"
                    
            except Exception as e:
                return f"❌ Error loading model: {str(e)}"
        
        lang_status = "Multilingual" if self.multilingual_mode else "English-only"
        return f"✅ Model {model_name} already loaded!\n🌐 Language mode: {lang_status}"
    
    def load_embeddings(self):
        """Load embedding model for vector search"""
        if self.embeddings is None:
            try:
                if self.multilingual_mode:
                    # Use multilingual embedding model for better cross-language performance
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                    )
                    return "✅ Multilingual embedding model loaded!"
                else:
                    # Use English-focused model for better English performance
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    return "✅ English embedding model loaded!"
            except Exception as e:
                # Fallback to English model if multilingual fails
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    return "✅ Embedding model loaded (fallback to English)!"
                except Exception as e2:
                    return f"❌ Error loading embeddings: {str(e2)}"
        
        model_type = "Multilingual" if self.multilingual_mode else "English"
        return f"✅ {model_type} embedding model already loaded!"
    
    def process_multiple_pdfs(self, pdf_files, chunk_size, chunk_overlap):
        """Process multiple uploaded PDFs with chunking and create vector store"""
        if not pdf_files or len(pdf_files) == 0:
            return "❌ No PDF files uploaded!"
        
        try:
            # Load embeddings first
            embed_status = self.load_embeddings()
            if "❌" in embed_status:
                return embed_status
            
            all_documents = []
            file_info = []
            total_pages = 0
            
            # Process each PDF file
            for i, pdf_file in enumerate(pdf_files):
                # Get the file path from the uploaded file
                if hasattr(pdf_file, 'name'):
                    file_path = pdf_file.name
                    file_name = os.path.basename(file_path)
                else:
                    file_path = pdf_file
                    file_name = os.path.basename(pdf_file)
                
                print(f"Processing file {i+1}/{len(pdf_files)}: {file_name}")
                
                # Load the PDF using PyMuPDFLoader
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
                
                # Add source file information to metadata
                for doc in documents:
                    doc.metadata['source_file'] = file_name
                    doc.metadata['file_index'] = i
                    doc.metadata['source_type'] = 'Local Upload'
                
                all_documents.extend(documents)
                file_info.append({
                    'name': file_name,
                    'pages': len(documents)
                })
                total_pages += len(documents)
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            self.documents = text_splitter.split_documents(all_documents)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            )
            
            self.processed_files = file_info
            chunks_count = len(self.documents)
            
            # Simple success message
            files_summary = ", ".join([info['name'] for info in file_info])
            return f"✅ Successfully processed {len(pdf_files)} files: {files_summary}\n📊 {total_pages} pages → {chunks_count} searchable chunks"
            
        except Exception as e:
            return f"❌ Error processing PDFs: {str(e)}"
    
    def retrieve_relevant_context(self, question, num_chunks):
        """Retrieve relevant document chunks for the question"""
        if self.retriever is None:
            return []
        
        try:
            # Retrieve relevant documents
            self.retriever.search_kwargs = {"k": num_chunks}
            relevant_docs = self.retriever.get_relevant_documents(question)
            return relevant_docs
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def answer_question(self, question, model_name, max_tokens, num_chunks, temperature):
        """Generate answer using RAG approach with retrieved context and optional language detection"""
        if self.model is None or self.current_model_name != model_name:
            return "❌ Please load the model first!"
        
        if self.retriever is None:
            return "❌ Please upload and process PDF files first!"
        
        if not question.strip():
            return "❌ Please enter a question!"
        
        try:
            # Detect language of the question (only in multilingual mode)
            if self.multilingual_mode:
                lang_code, lang_name = self.detect_language(question)
                print(f"Detected language: {lang_name} ({lang_code})")
            else:
                lang_code, lang_name = 'en', 'English'
            
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(question, num_chunks)
            
            if not relevant_docs:
                if self.multilingual_mode:
                    if lang_code == 'hinglish':
                        return "❌ Koi relevant information documents mein nahi mila."
                    elif lang_code == 'hi':
                        return "❌ दस्तावेजों में कोई प्रासंगिक जानकारी नहीं मिली।"
                    elif lang_code == 'es':
                        return "❌ No se encontró información relevante en los documentos."
                    else:
                        return "❌ No relevant context found in the documents."
                else:
                    return "❌ No relevant context found in the documents."
            
            # Combine retrieved contexts with source information
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                source_file = doc.metadata.get('source_file', 'Unknown file')
                source_type = doc.metadata.get('source_type', 'Unknown source')
                page_num = doc.metadata.get('page', 'Unknown page')
                context_parts.append(f"**Context {i+1}** [Source: {source_file} ({source_type}), Page: {page_num}]:\n{doc.page_content}\n")
            
            combined_context = "\n".join(context_parts)
            
            # Get language-specific instruction
            language_instruction = self.get_language_instruction(lang_code, lang_name)
            
            # Create the prompt with retrieved context
            if self.multilingual_mode:
                prompt = f"""Based on the following context from multiple documents, please answer the question accurately and comprehensively.

**Language Instruction:** {language_instruction}

**Retrieved Context from Documents:**
{combined_context}

**Question:** {question}

**Instructions:**
- Answer based solely on the provided context from the documents
- IMPORTANT: Respond in the same language as the question was asked
- If the question is in Hinglish, respond in Hinglish (mix Hindi and English using Roman script)
- If the question is in Hindi, respond in Hindi
- If the question is in Spanish, respond in Spanish
- If the context doesn't contain enough information, say so clearly in the same language
- Provide specific details and quotes when relevant
- Mention which document(s) the information comes from when possible
- Structure your response clearly with proper formatting
- Maintain the natural flow and style of the detected language

**Answer:**"""
            else:
                prompt = f"""Based on the following context from multiple documents, please answer the question accurately and comprehensively.

**Retrieved Context from Documents:**
{combined_context}

**Question:** {question}

**Instructions:**
- Answer based solely on the provided context from the documents
- If the context doesn't contain enough information, say so clearly
- Provide specific details and quotes when relevant
- Mention which document(s) the information comes from when possible
- Structure your response clearly with proper formatting

**Answer:**"""

            # Prepare messages for the chat template
            messages = [
                {"role": "user", "content": prompt}
            ]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

            # Extract generated content
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

            # Parse thinking content (if any)
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            # Add context information with cloud source details
            context_info = f"\n\n---\n**📚 Sources:** "
            sources = []
            for doc in relevant_docs:
                source_file = doc.metadata.get('source_file', 'Unknown')
                source_type = doc.metadata.get('source_type', 'Unknown')
                sources.append(f"{source_file} ({source_type})")
            
            unique_sources = list(dict.fromkeys(sources))  # Remove duplicates while preserving order
            context_info += ", ".join(unique_sources)
            
            return content + context_info

        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
        
    def general_answer_question(self, prompt, model_name, max_tokens, temperature):
        """Answer general-purpose questions not based on any documents"""
        if self.model is None or self.current_model_name != model_name:
            return "❌ Please load the model first!"

        if not prompt.strip():
            return "❌ Please enter a valid prompt!"

        try:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            return content
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

# Initialize the configurable QA system
qa_system = ConfigurablePDFQASystem()

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="PDF Q&A Assistant with Cloud Storage", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("# 🔍 PDF Q&A Assistant with Cloud Storage", elem_classes="text-center")
        gr.Markdown("Upload PDFs from local storage, Google Drive, or Dropbox • Ask questions • Get intelligent answers", elem_classes="text-center")
        
        # Main workflow in tabs for better organization
        with gr.Tabs():
            
            # Tab 1: Setup
            with gr.TabItem("🛠️ Setup", id="setup"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🌐 Language Support")
                        language_mode = gr.Radio(
                            choices=[
                                ("English Only (Faster)", False),
                                ("Multilingual Support", True)
                            ],
                            label="Choose Mode",
                            value=False,
                            info="Multilingual supports 15+ languages including Hindi, Spanish, French"
                        )
                        
                        gr.Markdown("### 🤖 AI Model")
                        model_dropdown = gr.Dropdown(
                            choices=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
                            label="Select Model",
                            value="Qwen/Qwen3-1.7B",
                            info="Qwen3-4B is more capable but requires more memory"
                        )
                        
                        load_model_btn = gr.Button("🔄 Load Model", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        model_status = gr.Textbox(
                            label="Status",
                            value="No model loaded",
                            interactive=False,
                            lines=4
                        )
                        
                        # Advanced settings in an accordion (collapsed by default)
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            chunk_size_slider = gr.Slider(
                                minimum=200, maximum=1000, value=500, step=50,
                                label="Text Chunk Size",
                                info="Larger chunks = more context, smaller chunks = more precise"
                            )
                            chunk_overlap_slider = gr.Slider(
                                minimum=0, maximum=200, value=50, step=10,
                                label="Chunk Overlap",
                                info="Overlap helps maintain context between chunks"
                            )

                # Event handlers for setup tab
                language_mode.change(
                    fn=qa_system.set_language_mode,
                    inputs=[language_mode],
                    outputs=[model_status]
                )
                
                load_model_btn.click(
                    fn=qa_system.load_model,
                    inputs=[model_dropdown],
                    outputs=[model_status]
                )

            # Tab 2: Cloud Storage Setup
            with gr.TabItem("☁️ Cloud Storage", id="cloud"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📦 Dropbox Integration")
                        with gr.Accordion("How to get Dropbox Access Token", open=False):
                            gr.Markdown("""
                            1. Go to [Dropbox App Console](https://www.dropbox.com/developers/apps)
                            2. Create a new app with "Full Dropbox" access
                            3. Generate an access token in the app settings
                            4. Copy and paste the token below
                            """)
                        
                        dropbox_token = gr.Textbox(
                            label="Dropbox Access Token",
                            placeholder="Enter your Dropbox access token...",
                            type="password",
                            lines=2
                        )
                        connect_dropbox_btn = gr.Button("🔗 Connect to Dropbox", variant="primary")
                        
                        dropbox_folder = gr.Textbox(
                            label="Dropbox Folder Path (optional)",
                            placeholder="e.g., /Documents/PDFs or leave empty for root",
                            value=""
                        )
                        list_dropbox_btn = gr.Button("📂 List Dropbox PDFs", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("### 🔍 Google Drive Integration")
                        with gr.Accordion("How to setup Google Drive", open=False):
                            gr.Markdown("""
                            **Step 1: Create Google Cloud Project**
                            1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                            2. Create a new project or select existing one
                            3. Enable Google Drive API
                            
                            **Step 2: Create Credentials**
                            4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client IDs"
                            5. Application type: "Desktop application"
                            6. Add `urn:ietf:wg:oauth:2.0:oob` as authorized redirect URI
                            7. Download the credentials JSON file
                            
                            **Alternative: Service Account (Easier)**
                            - Create "Service Account" instead of OAuth2
                            - Download JSON key file
                            - Share your Drive folder with the service account email
                            """)
                        
                        gdrive_credentials = gr.File(
                            label="Google Drive Credentials JSON",
                            file_types=[".json"]
                        )
                        connect_gdrive_btn = gr.Button("🔗 Setup Google Drive", variant="primary")
                        
                        # Add authorization code input (initially hidden)
                        with gr.Column(visible=False) as gdrive_auth_section:
                            gr.Markdown("### 🔐 Complete Authorization")
                            gr.Markdown("After clicking the authorization link above, copy the code and paste it here:")
                            gdrive_auth_code = gr.Textbox(
                                label="Authorization Code",
                                placeholder="Paste the authorization code here...",
                                lines=2
                            )
                            complete_gdrive_btn = gr.Button("✅ Complete Setup", variant="primary")
                        
                        gdrive_folder_id = gr.Textbox(
                            label="Google Drive Folder ID (optional)",
                            placeholder="e.g., 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
                            value=""
                        )
                        list_gdrive_btn = gr.Button("📂 List Drive PDFs", variant="secondary")
                
                with gr.Row():
                    cloud_status = gr.Textbox(
                        label="Cloud Storage Status",
                        value="Not connected to any cloud storage",
                        interactive=False,
                        lines=3
                    )
                
                # Cloud storage event handlers
                connect_dropbox_btn.click(
                    fn=qa_system.setup_dropbox_connection,
                    inputs=[dropbox_token],
                    outputs=[cloud_status]
                )
                
                # Google Drive setup handlers
                def handle_gdrive_setup(credentials_file):
                    result = qa_system.setup_gdrive_connection(credentials_file)
                    # Show auth section if setup requires authorization
                    show_auth = "authorization code" in result.lower() or "click this link" in result.lower()
                    return result, gr.Column(visible=show_auth)
                
                connect_gdrive_btn.click(
                    fn=handle_gdrive_setup,
                    inputs=[gdrive_credentials],
                    outputs=[cloud_status, gdrive_auth_section]
                )
                
                complete_gdrive_btn.click(
                    fn=qa_system.complete_gdrive_setup,
                    inputs=[gdrive_credentials, gdrive_auth_code],
                    outputs=[cloud_status]
                ).then(
                    fn=lambda: gr.Column(visible=False),
                    outputs=[gdrive_auth_section]
                )
                
                def list_dropbox_files_handler(folder_path):
                    status, files = qa_system.get_dropbox_files(folder_path)
                    return status, files if files else []
                
                def list_gdrive_files_handler(folder_id):
                    status, files = qa_system.get_gdrive_files(folder_id)
                    return status, files if files else []
                
                # Store cloud files for processing
                cloud_files_state = gr.State([])
                cloud_source_state = gr.State("")
                
                list_dropbox_btn.click(
                    fn=list_dropbox_files_handler,
                    inputs=[dropbox_folder],
                    outputs=[cloud_status, cloud_files_state]
                ).then(
                    fn=lambda: "Dropbox",
                    outputs=[cloud_source_state]
                )
                
                list_gdrive_btn.click(
                    fn=list_gdrive_files_handler,
                    inputs=[gdrive_folder_id],
                    outputs=[cloud_status, cloud_files_state]
                ).then(
                    fn=lambda: "Google Drive",
                    outputs=[cloud_source_state]
                )
            
            # Tab 3: Documents (Enhanced with Cloud)
            with gr.TabItem("📄 Documents", id="documents"):
                with gr.Tabs():
                    # Local upload subtab
                    with gr.TabItem("💻 Local Upload"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                pdf_upload = gr.File(
                                    label="📁 Upload PDF Documents",
                                    file_types=[".pdf"],
                                    file_count="multiple",
                                    height=200
                                )
                                
                                process_pdf_btn = gr.Button("🔄 Process Local Documents", variant="primary", size="lg")
                            
                            with gr.Column(scale=1):
                                pdf_status = gr.Textbox(
                                    label="Processing Status",
                                    value="No documents uploaded",
                                    interactive=False,
                                    lines=8
                                )

                        process_pdf_btn.click(
                            fn=qa_system.process_multiple_pdfs,
                            inputs=[pdf_upload, chunk_size_slider, chunk_overlap_slider],
                            outputs=[pdf_status]
                        )
                    
                    # Cloud files subtab
                    with gr.TabItem("☁️ Cloud Files"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                cloud_files_display = gr.Markdown("No files loaded. Use Cloud Storage tab to list files.")
                                
                                with gr.Row():
                                    select_all_btn = gr.Button("✅ Select All", variant="secondary")
                                    select_none_btn = gr.Button("❌ Deselect All", variant="secondary")
                                
                                cloud_files_checklist = gr.CheckboxGroup(
                                    label="Select files to process",
                                    choices=[],
                                    value=[]
                                )
                                
                                process_cloud_btn = gr.Button("🔄 Process Selected Cloud Files", variant="primary", size="lg")
                            
                            with gr.Column(scale=1):
                                cloud_process_status = gr.Textbox(
                                    label="Processing Status",
                                    value="No cloud files selected",
                                    interactive=False,
                                    lines=8
                                )
                        
                        # --- FIX STARTS HERE ---

                        # This function now correctly returns a markdown string for display and an
                        # update object for the CheckboxGroup, which sets both the choices and
                        # resets the selected value to empty.
                        def update_cloud_files_list(files, source):
                            if not files:
                                return "No files available.", gr.update(choices=[], value=[])
                            
                            display_text = f"**{len(files)} files found in {source}:**\n"
                            choices = []
                            for file in files:
                                size_mb = (file.get('size') or 0) / (1024 * 1024)
                                modified = file.get('modified', 'N/A')
                                display_text += f"- 📄 **{file['name']}** ({size_mb:.2f} MB, modified: {modified})\n"
                                choices.append(file['name'])
                            
                            return display_text, gr.update(choices=choices, value=[])

                        # The .change event is now simpler, with two outputs matching the two return
                        # values from the function. No more confusing .then() chain.
                        cloud_files_state.change(
                            fn=update_cloud_files_list,
                            inputs=[cloud_files_state, cloud_source_state],
                            outputs=[cloud_files_display, cloud_files_checklist]
                        )

                        # This function correctly gets all possible file names from the state object
                        # and sets the value of the CheckboxGroup to all of them.
                        def select_all_files(all_files_data):
                            if not all_files_data:
                                return gr.update(value=[])
                            all_names = [f['name'] for f in all_files_data]
                            return gr.update(value=all_names)

                        # This function returns an update object to clear the selection.
                        def select_no_files():
                            return gr.update(value=[])
                        
                        # The "Select All" button now correctly uses the cloud_files_state as input
                        # to get the full list of file names to select.
                        select_all_btn.click(
                            fn=select_all_files,
                            inputs=[cloud_files_state],
                            outputs=[cloud_files_checklist]
                        )
                        
                        select_none_btn.click(
                            fn=select_no_files,
                            inputs=None,
                            outputs=[cloud_files_checklist]
                        )
                        
                        # The process button's event handler remains correct.
                        process_cloud_btn.click(
                            fn=qa_system.process_cloud_files,
                            inputs=[cloud_files_checklist, cloud_files_state, cloud_source_state, chunk_size_slider, chunk_overlap_slider],
                            outputs=[cloud_process_status]
                        )
                        
                        # --- FIX ENDS HERE ---

            # Tab 4: General Purpose Chat
            with gr.TabItem("🌐 General Purpose Chat", id="general"):
                with gr.Row():
                    with gr.Column(scale=2):
                        general_prompt = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What is C++?",
                            lines=3
                        )

                        with gr.Row():
                            general_ask_btn = gr.Button("💬 Ask", variant="primary", size="lg")
                            general_clear_btn = gr.Button("🧹 Clear", variant="secondary")

                        with gr.Accordion("🎛️ Response Settings", open=False):
                            general_max_tokens_slider = gr.Slider(
                                minimum=100, maximum=800, value=400, step=50,
                                label="Response Length"
                            )
                            general_temperature_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                                label="Creativity"
                            )

                    with gr.Column(scale=3):
                        general_answer_output = gr.Markdown(
                            value="*Your answer will appear here...*",
                            label="Answer"
                        )

                # Event handlers for general-purpose tab
                general_ask_btn.click(
                    fn=qa_system.general_answer_question,
                    inputs=[general_prompt, model_dropdown, general_max_tokens_slider, general_temperature_slider],
                    outputs=[general_answer_output]
                )

                def clear_general():
                    return "", "*Your answer will appear here...*"

                general_clear_btn.click(
                    fn=clear_general,
                    outputs=[general_prompt, general_answer_output]
                )

            # Tab 5: Q&A (Main interaction)
            with gr.TabItem("💬 Ask Questions", id="qa"):
                with gr.Row():
                    with gr.Column(scale=2):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="What would you like to know about your documents?",
                            lines=3
                        )
                        
                        with gr.Row():
                            ask_btn = gr.Button("🤔 Ask Question", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                        
                        # Generation controls in an accordion
                        with gr.Accordion("🎛️ Response Settings", open=False):
                            with gr.Row():
                                max_tokens_slider = gr.Slider(
                                    minimum=100, maximum=800, value=400, step=50,
                                    label="Response Length", 
                                    info="Maximum length of the response"
                                )
                                temperature_slider = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                                    label="Creativity",
                                    info="Higher = more creative, Lower = more focused"
                                )
                            
                            num_chunks_slider = gr.Slider(
                                minimum=1, maximum=8, value=4, step=1,
                                label="Context Sources",
                                info="How many document sections to consider"
                            )
                    
                    with gr.Column(scale=3):
                        answer_output = gr.Markdown(
                            value="*Your answer will appear here...*",
                            label="Answer"
                        )

                # Event handlers for Q&A tab
                ask_btn.click(
                    fn=qa_system.answer_question,
                    inputs=[question_input, model_dropdown, max_tokens_slider, num_chunks_slider, temperature_slider],
                    outputs=[answer_output]
                )
                
                def clear_qa():
                    return "", "*Your answer will appear here...*"
                
                clear_btn.click(
                    fn=clear_qa,
                    outputs=[question_input, answer_output]
                )
        
        # Footer with quick actions and setup instructions
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💡 Quick Setup Guide")
                gr.Markdown("""
                1. **Setup**: Choose language mode → Load AI model
                2. **Cloud Storage**: Connect Dropbox/Google Drive (optional)
                3. **Documents**: Upload local PDFs or select from cloud
                4. **Ask**: Ask specific questions for better answers
                """)
            
            with gr.Column():
                with gr.Accordion("🔧 Cloud Storage Setup", open=False):
                    gr.Markdown("""
                    **Dropbox:**
                    • Create app at [Dropbox Console](https://www.dropbox.com/developers/apps)
                    • Generate access token
                    
                    **Google Drive:**
                    • Enable Drive API in [Google Cloud Console](https://console.cloud.google.com/)
                    • Create OAuth 2.0 credentials
                    • Download credentials JSON
                    """)
            
            with gr.Column():
                with gr.Accordion("🎯 Example Questions", open=False):
                    gr.Markdown("""
                    • "What are the main findings in these documents?"
                    • "Compare the methodologies used"
                    • "What recommendations are mentioned?"
                    • "Summarize the key conclusions"
                    """)
            
            with gr.Column():
                unload_model_btn = gr.Button("🗑️ Unload Model (Free Memory)", variant="secondary")
                
                def unload_model():
                    qa_system.unload_current_model()
                    qa_system.cloud_manager.cleanup_temp_files()
                    return "Model unloaded - GPU memory freed - Temp files cleaned"
                
                unload_model_btn.click(
                    fn=unload_model,
                    outputs=[model_status]
                )
    
    return interface

# Create the app instance
app = create_interface()

# For Gradio compatibility with WSGI servers like Gunicorn
def create_app():
    return app

# Launch the application
if __name__ == "__main__":
    # Development
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False, # Set to True to share publicly
        debug=True
    )