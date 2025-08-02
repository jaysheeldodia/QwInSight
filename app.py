import gradio as gr
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re
from langdetect import detect

class EnhancedMultilingualPDFQASystem:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.retriever = None
        self.embeddings = None
        self.vector_store = None
        self.documents = None
        self.processed_files = []  # Track processed files
        
        # Language detection and mapping
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
    
    def detect_language(self, text):
        """Detect the language of input text and check for Hinglish patterns"""
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
                    return f"✅ Model {model_name} loaded successfully!\n💾 GPU Memory: {memory_after:.2f} GB (+{memory_used:.2f} GB)\n🌐 Multilingual support enabled!"
                else:
                    return f"✅ Model {model_name} loaded successfully! (CPU mode)\n🌐 Multilingual support enabled!"
                    
            except Exception as e:
                return f"❌ Error loading model: {str(e)}"
        return f"✅ Model {model_name} already loaded!\n🌐 Multilingual support enabled!"
    
    def load_embeddings(self):
        """Load embedding model for vector search"""
        if self.embeddings is None:
            try:
                # Use multilingual embedding model for better cross-language performance
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                return "✅ Multilingual embedding model loaded!"
            except Exception as e:
                # Fallback to English model if multilingual fails
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    return "✅ Embedding model loaded (English-focused)!"
                except Exception as e2:
                    return f"❌ Error loading embeddings: {str(e2)}"
        return "✅ Multilingual embedding model already loaded!"
    
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
            
            # Show sample chunks for preview
            sample_chunks = []
            for i, doc in enumerate(self.documents[:3]):
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                source_info = f"[{doc.metadata.get('source_file', 'Unknown')} - Page {doc.metadata.get('page', 'Unknown')}]"
                sample_chunks.append(f"**Chunk {i+1}** {source_info}: {preview}")
            
            preview_text = "\n\n".join(sample_chunks)
            
            # Create file summary
            files_summary = "\n".join([f"- **{info['name']}**: {info['pages']} pages" for info in file_info])
            
            return f"""✅ Multiple PDFs processed successfully with multilingual support!

**📚 Processed Files ({len(pdf_files)} files):**
{files_summary}

**📊 Document Statistics:**
- Total pages: {total_pages}
- Text chunks created: {chunks_count}
- Chunk size: {chunk_size} characters
- Chunk overlap: {chunk_overlap} characters

**🌐 Multilingual Features:**
- Supports questions in multiple languages
- Auto-detects language and responds accordingly
- Special support for Hinglish (Hindi + English mix)

**🔍 Sample chunks:**
{preview_text}

🔍 Vector search is now ready across all documents in multiple languages!"""
            
        except Exception as e:
            import traceback
            return f"❌ Error processing PDFs: {str(e)}\n\n{traceback.format_exc()}"
    
    def retrieve_relevant_context(self, question, num_chunks):
        """Retrieve relevant document chunks for the question"""
        if self.retriever is None:
            return []
        
        try:
            # Retrieve relevant documents
            relevant_docs = self.retriever.search_kwargs = {"k": num_chunks}
            relevant_docs = self.retriever.get_relevant_documents(question)
            return relevant_docs
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def answer_question(self, question, model_name, max_tokens, num_chunks, temperature):
        """Generate answer using RAG approach with retrieved context and language detection"""
        if self.model is None or self.current_model_name != model_name:
            return "❌ Please load the model first!"
        
        if self.retriever is None:
            return "❌ Please upload and process PDF files first!"
        
        if not question.strip():
            return "❌ Please enter a question!"
        
        try:
            # Detect language of the question
            lang_code, lang_name = self.detect_language(question)
            print(f"Detected language: {lang_name} ({lang_code})")
            
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(question, num_chunks)
            
            if not relevant_docs:
                if lang_code == 'hinglish':
                    return "❌ Koi relevant information documents mein nahi mila."
                elif lang_code == 'hi':
                    return "❌ दस्तावेजों में कोई प्रासंगिक जानकारी नहीं मिली।"
                elif lang_code == 'es':
                    return "❌ No se encontró información relevante en los documentos."
                else:
                    return "❌ No relevant context found in the documents."
            
            # Combine retrieved contexts with source information
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                source_file = doc.metadata.get('source_file', 'Unknown file')
                page_num = doc.metadata.get('page', 'Unknown page')
                context_parts.append(f"**Context {i+1}** [Source: {source_file}, Page: {page_num}]:\n{doc.page_content}\n")
            
            combined_context = "\n".join(context_parts)
            
            # Get language-specific instruction
            language_instruction = self.get_language_instruction(lang_code, lang_name)
            
            # Create the multilingual prompt with retrieved context
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
            
            # Add context information to the response in appropriate language
            if lang_code == 'hinglish':
                context_info = f"\n\n---\n**📚 Retrieved Context Sources (Ye sources se answer mila):**\n"
            elif lang_code == 'hi':
                context_info = f"\n\n---\n**📚 प्राप्त संदर्भ स्रोत:**\n"
            elif lang_code == 'es':
                context_info = f"\n\n---\n**📚 Fuentes de Contexto Recuperadas:**\n"
            else:
                context_info = f"\n\n---\n**📚 Retrieved Context Sources:**\n"
            
            for i, doc in enumerate(relevant_docs):
                source_file = doc.metadata.get('source_file', 'Unknown file')
                page_info = f"Page {doc.metadata.get('page', 'Unknown')}"
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                context_info += f"- **Context {i+1}** ({source_file}, {page_info}): {preview}\n"
            
            # Add language detection info
            lang_info = f"\n**🌐 Detected Language:** {lang_name}"
            
            return content + context_info + lang_info

        except Exception as e:
            import traceback
            return f"❌ Error generating answer: {str(e)}\n\n{traceback.format_exc()}"

# Initialize the enhanced multilingual QA system
qa_system = EnhancedMultilingualPDFQASystem()

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="QwInSight Multi-PDF Multilingual", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🌐 Enhanced Multilingual Multi-PDF Q&A with RAG")
        gr.Markdown("Upload multiple PDF documents and ask questions in any language! Supports English, Hindi, Hinglish, Spanish, French, German, and many more languages.")
        gr.Markdown("**📋 Installation:** `pip install sentence-transformers langchain faiss-cpu transformers torch langdetect`")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                gr.Markdown("### 🤖 Model Configuration")
                model_dropdown = gr.Dropdown(
                    choices=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
                    label="Select Qwen Model",
                    value="Qwen/Qwen3-1.7B",
                    info="Choose the Qwen model for multilingual text generation"
                )
                
                load_model_btn = gr.Button("🔄 Load Model", variant="primary")
                model_status = gr.Textbox(
                    label="Model Status",
                    value="No model loaded",
                    interactive=False
                )
                
                # PDF processing configuration
                gr.Markdown("### 📄 Document Processing")
                pdf_upload = gr.File(
                    label="Upload PDF Documents (Multiple Files Supported)",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                
                with gr.Row():
                    chunk_size_slider = gr.Slider(
                        minimum=200,
                        maximum=1000,
                        value=500,
                        step=50,
                        label="Chunk Size",
                        info="Size of text chunks for processing"
                    )
                    
                    chunk_overlap_slider = gr.Slider(
                        minimum=0,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Chunk Overlap",
                        info="Overlap between consecutive chunks"
                    )
                
                process_pdf_btn = gr.Button("📋 Process All PDFs & Create Vector Store", variant="primary")
                
                pdf_status = gr.Textbox(
                    label="Processing Status",
                    value="No PDF files uploaded",
                    interactive=False,
                    lines=12
                )
            
            with gr.Column(scale=2):
                # Question and answer section
                gr.Markdown("### 💬 Ask Questions in Any Language")
                gr.Markdown("**🌐 Supported Languages:** English, Hindi, Hinglish, Spanish, French, German, Portuguese, Russian, Japanese, Korean, Arabic, Chinese, Italian, Dutch, Turkish")
                
                question_input = gr.Textbox(
                    label="Your Question (किसी भी भाषा में / En cualquier idioma)",
                    placeholder="Ask a question in any language: 'What is...?' / 'Kya hai...?' / '¿Qué es...?' / 'Was ist...?'",
                    lines=3
                )
                
                # Generation parameters
                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=400,
                        step=50,
                        label="Max New Tokens",
                        info="Maximum response length"
                    )
                    
                    num_chunks_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                        label="Retrieved Chunks",
                        info="Number of relevant chunks to retrieve"
                    )
                
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness in generation"
                )
                
                ask_btn = gr.Button("🤔 Ask Question", variant="primary", size="lg")
                
                # Answer output with markdown rendering
                answer_output = gr.Markdown(
                    label="Answer",
                    value="*The answer will appear here in your language...*"
                )
                
                # Control buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                    example_btn = gr.Button("💡 Load Example Questions", variant="secondary")
        
        # Multilingual example questions section
        with gr.Row():
            gr.Markdown("""
            ### 💡 Multilingual Example Questions:
            
            **English:**
            - "What are the main findings across all documents?"
            - "How do the approaches differ between the documents?"
            
            **Hindi:**
            - "सभी दस्तावेजों में मुख्य निष्कर्ष क्या हैं?"
            - "दस्तावेजों के बीच दृष्टिकोण कैसे भिन्न हैं?"
            
            **Hinglish:**
            - "Documents mein kya main findings hai?"
            - "Sabse important recommendations kya hai?"
            - "Kis document mein best analysis hai?"
            
            **Spanish:**
            - "¿Cuáles son los principales hallazgos en todos los documentos?"
            - "¿Cómo difieren los enfoques entre los documentos?"
            
            **French:**
            - "Quelles sont les principales conclusions de tous les documents?"
            - "Comment les approches diffèrent-elles entre les documents?"
            """)
        
        # Event handlers
        load_model_btn.click(
            fn=qa_system.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        process_pdf_btn.click(
            fn=qa_system.process_multiple_pdfs,
            inputs=[pdf_upload, chunk_size_slider, chunk_overlap_slider],
            outputs=[pdf_status]
        )
        
        ask_btn.click(
            fn=qa_system.answer_question,
            inputs=[question_input, model_dropdown, max_tokens_slider, num_chunks_slider, temperature_slider],
            outputs=[answer_output]
        )
        
        def clear_all():
            qa_system.retriever = None
            qa_system.vector_store = None
            qa_system.documents = None
            qa_system.processed_files = []
            return "", "No PDF files uploaded", "*Ask a question in any language...*", ""
        
        def unload_model():
            qa_system.unload_current_model()
            qa_system.current_model_name = None
            return "No model loaded"
        
        clear_btn.click(
            fn=clear_all,
            outputs=[question_input, pdf_status, answer_output, question_input]
        )
        
        # Add model unload button
        with gr.Row():
            unload_model_btn = gr.Button("🗑️ Unload Model (Free GPU Memory)", variant="secondary")
        
        unload_model_btn.click(
            fn=unload_model,
            outputs=[model_status]
        )
        
        def load_examples():
            examples = [
                "What are the main topics covered across all documents?",
                "सभी दस्तावेजों में मुख्य विषय क्या हैं?",
                "Documents mein kya main findings hai?",
                "¿Cuáles son los temas principales cubiertos en todos los documentos?",
                "Quels sont les principaux sujets couverts dans tous les documents?"
            ]
            return examples[0]
        
        example_btn.click(
            fn=load_examples,
            outputs=[question_input]
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
        share=False,
        debug=True
    )

    # Production
    # app.launch(
    #     server_name="0.0.0.0",
    #     server_port=7860,
    #     share=True,
    #     debug=False
    # )