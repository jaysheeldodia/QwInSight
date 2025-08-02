from langchain_community.document_loaders import PyMuPDFLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the document
file_path = "./paper.pdf"
loader = PyMuPDFLoader(file_path)
docs = loader.load()

# Extract context from the PDF (first page, for simplicity)
context = docs[0].page_content

# Step 2: Load the model and tokenizer (as per Hugging Face recommended method)
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Step 3: Prepare the question and the prompt
print("✅ Paper Loaded")


while True:
    question = input("Enter your question: ")
    
    if question == 'q':
        break
    
    prompt = f"""Context:
{context}

Question:
"{question}"

Instructions:
- Analyze the provided context carefully and extract all relevant information to accurately answer the question.
- Ensure that your response is based solely on the context, and avoid making assumptions beyond the given data.
- Provide a clear, concise, and informative answer with supporting details directly from the context.
- If the answer cannot be determined from the context, respond with "Unable to answer based on the given information."

Answer:
    """

    # Step 4: Tokenize the input and prepare the model input
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Use the tokenizer's apply_chat_template method for formatting
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Optional: controls thinking mode, default is True
    )

    # Tokenize the input text
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Step 5: Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100  # Adjust based on your needs
    )

    # Step 6: Extract the generated content
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Parse thinking content
    try:
        # Look for the thinking token (e.g., </think>) in the output
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # Step 7: Print the result
    print("Answer:", content)
