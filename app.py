import gradio as gr
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

class EnhancedPDFQASystem:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_name = None
        self.retriever = None
        self.embeddings = None
        self.vector_store = None
        self.documents = None
        
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
                    return f"✅ Model {model_name} loaded successfully!\n💾 GPU Memory: {memory_after:.2f} GB (+{memory_used:.2f} GB)"
                else:
                    return f"✅ Model {model_name} loaded successfully! (CPU mode)"
                    
            except Exception as e:
                return f"❌ Error loading model: {str(e)}"
        return f"✅ Model {model_name} already loaded!"
    
    def load_embeddings(self):
        """Load embedding model for vector search"""
        if self.embeddings is None:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                return "✅ Embedding model loaded!"
            except Exception as e:
                return f"❌ Error loading embeddings: {str(e)}"
        return "✅ Embedding model already loaded!"
    
    def process_pdf(self, pdf_file, chunk_size, chunk_overlap):
        """Process uploaded PDF with chunking and create vector store"""
        if pdf_file is None:
            return "❌ No PDF file uploaded!"
        
        try:
            # Load embeddings first
            embed_status = self.load_embeddings()
            if "❌" in embed_status:
                return embed_status
            
            # Get the file path from the uploaded file
            if hasattr(pdf_file, 'name'):
                file_path = pdf_file.name
            else:
                file_path = pdf_file
            
            # Load the PDF using PyMuPDFLoader
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            self.documents = text_splitter.split_documents(documents)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            )
            
            pages_count = len(documents)
            chunks_count = len(self.documents)
            
            # Show sample chunks for preview
            sample_chunks = []
            for i, doc in enumerate(self.documents[:3]):
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                sample_chunks.append(f"**Chunk {i+1}:** {preview}")
            
            preview_text = "\n\n".join(sample_chunks)
            
            return f"""✅ PDF processed successfully!

**Document Info:**
- Pages: {pages_count}
- Text chunks created: {chunks_count}
- Chunk size: {chunk_size} characters
- Chunk overlap: {chunk_overlap} characters

**Sample chunks:**
{preview_text}

🔍 Vector search is now ready!"""
            
        except Exception as e:
            import traceback
            return f"❌ Error processing PDF: {str(e)}\n\n{traceback.format_exc()}"
    
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
        """Generate answer using RAG approach with retrieved context"""
        if self.model is None or self.current_model_name != model_name:
            return "❌ Please load the model first!"
        
        if self.retriever is None:
            return "❌ Please upload and process a PDF first!"
        
        if not question.strip():
            return "❌ Please enter a question!"
        
        try:
            # Retrieve relevant context
            relevant_docs = self.retrieve_relevant_context(question, num_chunks)
            
            if not relevant_docs:
                return "❌ No relevant context found in the document."
            
            # Combine retrieved contexts
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                context_parts.append(f"**Context {i+1}:**\n{doc.page_content}\n")
            
            combined_context = "\n".join(context_parts)
            
            # Create the prompt with retrieved context
            prompt = f"""Based on the following context from the document, please answer the question accurately and comprehensively.

**Retrieved Context:**
{combined_context}

**Question:** {question}

**Instructions:**
- Answer based solely on the provided context
- If the context doesn't contain enough information, say so clearly
- Provide specific details and quotes when relevant
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
            
            # Add context information to the response
            context_info = f"\n\n---\n**📚 Retrieved Context Sources:**\n"
            for i, doc in enumerate(relevant_docs):
                page_info = f"Page {doc.metadata.get('page', 'Unknown')}" if 'page' in doc.metadata else "Source document"
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                context_info += f"- **Context {i+1}** ({page_info}): {preview}\n"
            
            return content + context_info

        except Exception as e:
            import traceback
            return f"❌ Error generating answer: {str(e)}\n\n{traceback.format_exc()}"

# Initialize the enhanced QA system
qa_system = EnhancedPDFQASystem()

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Enhanced PDF Q&A with RAG", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔍 Enhanced PDF Q&A with Retrieval-Augmented Generation")
        gr.Markdown("Upload a PDF document and ask questions using advanced vector search and Qwen language models.")
        gr.Markdown("**📋 Installation Requirements:** `pip install sentence-transformers langchain faiss-cpu transformers torch`")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                gr.Markdown("### 🤖 Model Configuration")
                model_dropdown = gr.Dropdown(
                    choices=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
                    label="Select Qwen Model",
                    value="Qwen/Qwen3-1.7B",
                    info="Choose the Qwen model for text generation"
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
                    label="Upload PDF Document",
                    file_types=[".pdf"]
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
                
                process_pdf_btn = gr.Button("📋 Process PDF & Create Vector Store", variant="primary")
                
                pdf_status = gr.Textbox(
                    label="Processing Status",
                    value="No PDF uploaded",
                    interactive=False,
                    lines=8
                )
            
            with gr.Column(scale=2):
                # Question and answer section
                gr.Markdown("### 💬 Ask Questions")
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the uploaded PDF...",
                    lines=3
                )
                
                # Generation parameters
                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=300,
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
                    value="*The answer will appear here...*"
                )
                
                # Control buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                    example_btn = gr.Button("💡 Load Example Questions", variant="secondary")
        
        # Example questions section
        with gr.Row():
            gr.Markdown("""
            ### 💡 Example Questions to Try:
            - **Summary**: "What are the main findings or conclusions of this document?"
            - **Specific Details**: "What methodology was used in this research?"
            - **Analysis**: "What are the key recommendations mentioned?"
            - **Comparison**: "How does this approach differ from previous methods?"
            - **Technical**: "What are the technical specifications mentioned?"
            """)
        
        # Event handlers
        load_model_btn.click(
            fn=qa_system.load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        process_pdf_btn.click(
            fn=qa_system.process_pdf,
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
            return "", "No PDF uploaded", "*Ask a question about the uploaded PDF...*", ""
        
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
                "What is the main topic and purpose of this document?",
                "Can you summarize the key findings or conclusions?",
                "What methodology or approach is described?",
                "Are there any specific recommendations or suggestions mentioned?"
            ]
            return gr.update(choices=examples, value=examples[0])
        
        example_btn.click(
            fn=load_examples,
            outputs=[]
        )
    
    return interface

# Launch the application
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )