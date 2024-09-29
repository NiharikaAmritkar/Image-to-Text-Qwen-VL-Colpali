import streamlit as st
import tempfile
from openai import OpenAI
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)
# Load the models and processor once when the app starts
# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload an image (.png or .jpg or .jpeg)", type=("png", "jpg", "jpeg")
    )

    # Ask the user for a question via `st.text_area`.
    text_query = st.text_area(
        "Now ask a question about the document!",
        placeholder="extract the text?",
        disabled=not uploaded_file,
    )
    
    @st.cache_data
    def load_models():
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8")
        return RAG, model, processor
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())  # Write the uploaded file content
        temp_file_path = temp_file.name  # Get the temporary file path
    
    RAG.index(
        input_path=temp_file_path,
        index_name="image_index",
        store_collection_with_index=False,
        overwrite=True
    )
    results = RAG.search(text_query, k=1)

    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    # "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8", torch_dtype="auto", device_map="auto"
    # )
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8")

     # Step 5: Prepare messages for inference
    if results:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": uploaded_file,  # Use the uploaded image
                    },
                    {"type": "text", "text": text_query},
                ],
            }
        ]

    # Step 6: Prepare input for the model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Assuming process_vision_info is defined
    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenizing and preparing inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Step 7: Inference and generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated output
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    st.write(output_text)

    
    # if uploaded_file and text_query:

    #     # Process the uploaded file and question.
    #     document = uploaded_file.read().decode()
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": f"Here's a document: {document} \n\n---\n\n {text_query}",
    #         }
    #     ]

        # Generate an answer using the OpenAI API.
        # stream = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=messages,
        #     stream=True,
        # )

        # Stream the response to the app using `st.write_stream`.
        # st.write_stream(stream)
