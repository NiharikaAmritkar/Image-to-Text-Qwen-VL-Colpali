from pathlib import Path
from PIL import Image
import streamlit as st
import tempfile
from openai import OpenAI
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload an image below and ask a question about it! "
)

uploaded_file = st.file_uploader(
    "Upload an image (.png or .jpg or .jpeg)", type=("png", "jpg", "jpeg")
)

# Ask the user for a question via `st.text_area`.
text_query = st.text_area(
    "Now ask a question about the document!",
    placeholder="extract the text?",
    disabled=not uploaded_file,
)
    

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.getvalue())  # Write the uploaded file content
        temp_file_path = temp_file.name  # Get the temporary file path
        st.write(temp_file_path)
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")


    RAG.index(
        input_path=temp_file_path,
        index_name="image_index",
        store_collection_with_index=False,
        overwrite=True
    )
    results = RAG.search(text_query, k=1)



    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8")
    
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
   

