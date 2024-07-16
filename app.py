"""
I have tried training llama2 using LoRA but due to space constraint on colab it was unsuccessful, I plan to improve current chatbot for future undertaking regarding this
"""
import streamlit as st
# import random
import time
# import pickle
import google.generativeai as genai
GOOGLE_API_KEY="AIzaSyADBTfBXbStTykdJReJJF0fSpEzkUeEjN4"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro-001')
chat = model.start_chat(history=[])
# from transformers import pispeline

## Future Scope for fine-tuning 
# def fine_tune(model):  
#     operation = genai.create_tuned_model(source_model=model.name,
#                                          train_data=[{}.{}],
#                                          id="new"
#                                         epoch_count = 100,
#                                         batch_size=4,
#                                         learning_rate=0.001,
#     )
#     tuned_model = genai.get_tuned_model(f'new')
#     return tuned_model

# model=fine_tune(model)
hiddenmasala=""" [instruction:]->'STRICTLY ACT AS A CHATBOT , FOR GENERAL QUERIES Customer support queries typically involve a range of tasks, including answering questions about products, services, and policies; troubleshooting and resolving technical issues; managing orders and accounts; addressing billing and payment inquiries; guiding returns and exchanges; handling feedback and complaints; providing technical support; offering general assistance and recommendations; educating customers through knowledge base articles and training sessions; and following up to ensure issues are fully resolved and gather feedback. These tasks aim to assist customers, resolve their issues, and provide valuable information efficiently.' 'keep your answers within 100 words and be PROFESSIONAL'
"""
###################################################################
def ask(query):
    query_str = ' '.join(str(q) for q in query)
    response = chat.send_message(query_str+"\n\n\n\n"+hiddenmasala)
    return response.text


#############################################################
# with open("tokenizer", "rb") as f:
    # tokenizer =pickle.load(f)

# with open(r"llama2_fine_tuned\adapter_model.bin", "rb") as f:
#     model =pickle.load(f) 
# model_path="entbappy/Llama-2-7b-chat-finetune"
# from transformers import AutoModelForCausalLM
# model=AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = "auto")
# def ask(query):
#     prompt = query
#     pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
#     result = pipe(f"<s>[INST] {prompt} [/INST]")
#     return(result[0]['generated_text'])

# from meta_ai_api import MetaAI

# ai = MetaAI()
# def ask(query):
#     response = ai.prompt(message=query)
#     if isinstance(response, dict) and "message" in response:
#         return response["message"]
#     else:
#         return "".join(response)  # If response is a generator
 

st.title("ChatBot")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your Query here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user",avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant",avatar="🤖"):
        stream=ask(m["content"] for m in st.session_state.messages)
        def data_generator(data):
            for i in data:
                yield f"{i}"
                time.sleep(0.01)
        response = st.write_stream(data_generator(stream))
    st.session_state.messages.append({"role": "assistant", "content": response})
