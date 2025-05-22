import streamlit as st
import os
#from pinecone import Pinecone
from openai import AzureOpenAI

deployment_Text_model = "gpt-4o-mini"
deployment_Image_model = "dall-e-3"
deployment_embedding_model = "text-embedding-3-large"

client1 = AzureOpenAI(
    api_key="3yDY1dqMqGmFIGAJmxBbIzCARcOkHTKf9BYZ0wDlZOoSXmmL88RjJQQJ99BEACfhMk5XJ3w3AAAAACOGSOw9",
    azure_endpoint="https://abhy-mav0mtc5-swedencentral.cognitiveservices.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01",
    api_version="2024-04-01-preview"
)
client2 = AzureOpenAI(
  azure_endpoint = "https://abhy-mas6ahj0-eastus2.openai.azure.com/", 
  api_key="G2REPoveOLPGDMkTY1b5ZnhVKNPWS8NTBpLQIH0eE5Iv6xPcAQmEJQQJ99BEACHYHv6XJ3w3AAAAACOGAwDG",  
  api_version="2025-01-01-preview"
)
client3 = AzureOpenAI(
    api_key="G2REPoveOLPGDMkTY1b5ZnhVKNPWS8NTBpLQIH0eE5Iv6xPcAQmEJQQJ99BEACHYHv6XJ3w3AAAAACOGAwDG",
    azure_endpoint="https://abhy-mas6ahj0-eastus2.cognitiveservices.azure.com/",
    api_version="2024-12-01-preview"
)

pn = Pinecone(api_key="pcsk_2XdgXm_PMkAVtoNK8CGW68Z3FyHC3EUYfp7sG42Pa4uzsgJtzGpRhZu6ng8qNumDANyMbT")

index = pn.Index()

# AI Functions Start

def generate_blog(topic, additional_text):
    prompt = [{"role":"user", "content": f"""
    You are a copy writer with years of experience writing impactful blogs that converge and help elevate brands.
    Your task is to write a blog on any topic system provides to you. Make sure to write in a format that works for Medium.
    
    Topic: {topic}
    Additiona pointers: {additional_text}
    """}]
    response = client2.chat.completions.create(
        model= deployment_Text_model,
        messages = prompt,
        temperature=0.8,
        max_tokens = 100
    )

    return response

def generate_image(prompt, no_images):
    response = client1.images.generate(
        model=deployment_Image_model,
        prompt = prompt,
        n=no_images
    )
    return response

def movie_recommendation(movie_description):
    movie_descr_vector = client3.embeddings.create(
        model = deployment_embedding_model,
        input = movie_description
    )
    movie_descr_vector_embeddings = movie_descr_vector.data[0].embedding

    result = index.query(
        movie_descr_vector_embeddings,
        top_k = 10,
        include_metadata = True
    )
    return result


# END AI Function

st.set_page_config(layout="wide")

st.title("OpenAI API Webapp")     # Title on page


st.sidebar.title("AI Apps")       #left side bar

ai_app = st.sidebar.radio("Choose an AI App", ("Blog Generator", "Image Generator", "Movie Recomendation"))

if ai_app == "Blog Generator":
    st.header("Blog Generator")
    st.write("Input a topic to generate a blog about it using OpenAI API")

    topic = st.text_area("Topic", height=68)
    additional_text = st.text_area("Additional Text", height=68)

    if st.button("Generate Blog"):
        with st.spinner("Loading..."):
            res = generate_blog(topic, additional_text)
            st.text_area("Generated Blog: ", value=res.choices[0].message.content, height=68)
            st.write("Blog Generated")

elif ai_app == "Image Generator":
    st.header("Image Generator")
    st.write("Add a prompt to generate an image using OpenAI API DALL-E")
    
    prompt = st.text_area("Prompt", height=68)

    no_images = st.slider("Number of Images", 1, 5, 1)

    if st.button("Generate Image") and prompt != "":
        with st.spinner("Loading..."):
            res = generate_image(prompt, 1)
            for output in res.data:
                st.image(output.url)
            st.write("Image Generated")

elif ai_app == "Movie Recomendation":
    st.header("Movie Recommendation")
    st.write("Input a prompt to recommend a movie ")

    movie_description = st.text_area("Movie Description", height=68)

    if st.button("Get Movie Recommendations"):
        with st.spinner("Loading..."):
            result = movie_recommendation(movie_description)
            for movie in result:
                st.write(movie['metadata']['title'])
            st.write("Movie Recommendations Generated")