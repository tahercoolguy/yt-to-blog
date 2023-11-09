import streamlit as st
from langchain import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from huggingface_hub import InferenceClient
from templates import BLOG_TEMPLATE, IMAGE_TEMPLATE

import os
os.environ["OPENAI_API_BASE"] = "http://chatgpt.multiplewords.com:1337"
os.environ["OPENAI_API_KEY"]= ""

CLIENT = InferenceClient()

# Initialize Streamlit
st.set_page_config(page_title ="📝 Article Generator App")
st.title('📝 Article Generator App')

# Getting the OpenAI API key from the sidebar
openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
chat_model = None
if openai_api_key.startswith('sk-'):
    chat_model = ChatOpenAI(model_name='gpt-4-32k', openai_api_key="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")
else:
    st.warning('Please enter a valid OpenAI API key!', icon='⚠')

@st.cache_data ()
def generate_blog(yt_url):
    """Generate a blog article from a YouTube URL."""
    loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=True)
    transcript = loader.load()

    final_transcript = ""
    main_transcript = transcript[0].page_content
    total_length = len(main_transcript)
    completed_length = 0
    if total_length > 3500:
        while True:
            if total_length > 3500 :
                new_prompt = "summaries given text in 10 points with very less description ' "+main_transcript[completed_length:completed_length+3500] +" '"
                new_data = chat_model.predict(new_prompt)
                final_transcript = final_transcript + new_data
                completed_length = completed_length + 3500
                total_length = total_length - completed_length
            else :
                new_prompt = "summaries given text in 10 points with very less description ' "+main_transcript[completed_length:total_length] +" '"
                new_data = chat_model.predict(new_prompt)
                final_transcript = final_transcript + new_data
                if len(final_transcript) > 3500:
                    main_transcript = final_transcript
                    total_length = len(main_transcript)
                    completed_length = 0
                    final_transcript = ""
                else:
                    break
    else:
        final_transcript = transcript[0].page_content

    # final_transcript = main_transcript
    """Create a response schema for structured output."""
    schema = [
        ResponseSchema(name="title", description="Article title"),
        ResponseSchema(name="meta_description", description="Article Meta Description"),
        ResponseSchema(name="content", description="Article content in markdown"),
    ]
    output_parser = StructuredOutputParser.from_response_schemas(schema)
    format_instructions = output_parser.get_format_instructions()
    
    prompt = PromptTemplate(
        input_variables=['transcript'],
        template=BLOG_TEMPLATE,
        partial_variables={"format_instructions": format_instructions}
    )
    prompt_query = prompt.format(transcript=final_transcript)

    response = chat_model.predict(prompt_query)
    print(response)
    return output_parser.parse(response), transcript[0].metadata["thumbnail_url"]

@st.cache_data ()
def generate_image(title):
    """Generate an image based on the title."""
    prompt = PromptTemplate(
        input_variables=['title'],
        template=IMAGE_TEMPLATE,
    )
    prompt_query = prompt.format(title=title)

    stb_prompt = chat_model.predict(prompt_query)

    tags = [
        stb_prompt,
        'award winning',
        'high resolution',
        'photo realistic',
        'intricate details',
        'beautiful',
        '[trending on artstation]'
    ]
    result = ', '.join(tags)
    response = CLIENT.post(json={
        "inputs": result,
        "parameters": { "negative_prompt": 'blurry, artificial, cropped, low quality, ugly'}
    }, model="stabilityai/stable-diffusion-2-1")

    return response

# Creating a form to get the YouTube URL
with st.form('myform'):
    yt_url = st.text_input('Enter youtube url:', '')
    generate_image_option = st.checkbox('Generate Image Instead of Thumbnail')
    submitted = st.form_submit_button('Submit')
    
    if submitted and chat_model and yt_url:
        with st.spinner("Generating blog... This may take a while⏳"):
            blog, thumbnail = generate_blog(yt_url)
            if generate_image_option:
                with st.spinner("Generating image... This may take a while⏳"):
                    image = generate_image(blog['title'])
                    st.image(image)
            else:
                st.image(thumbnail)
            st.markdown(blog['content'])