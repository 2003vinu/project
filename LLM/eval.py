import moviepy.editor as mp
import whisper
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from os.path import expanduser
from langchain_community.llms import LlamaCpp

def video_to_audio(video_path, audio_path):
    video_clip = mp.VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)

def audio_to_text(audio_path):
    model = whisper.load_model("large-v3-turbo", device="cuda")
    text = model.transcribe(audio_path)
    return text

def get_rating(text):
    text = text['text']
    #model_path = expanduser("llama-2-13b-chat.Q8_0.gguf")

    template_messages = [
        SystemMessage(content="""Imagine You are an AI model who is able to provide ratings and review of tutorials and online course videos after analysing the content that are given as an input,
                                 you must be able to verify whether the given details are true as per your knowledge in that field and provide rating for that particular course out of ten,
                                 add 4 points for accuracy of the content and completeness,
                                 add 2 points for major, factual information given and take away half a star for false factual content,
                                 add 2 points for proper usage of terminology and technical words,
                                 add 2 points for real-time examples and practical applications,
                                 remove points accorgingly for improper usage of terminology"""),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    prompt_template = ChatPromptTemplate.from_messages(template_messages)

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        max_tokens=1024,
        streaming=False,
    )
    model = Llama2Chat(llm=llm)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(llm=model, prompt=prompt_template, memory=memory)

    response = chain.invoke(text)

    return response['text']

def video_to_text(video_path):
    audio_path = "temp_audio.wav"
    video_to_audio(video_path, audio_path)
    text = audio_to_text(audio_path)
    print(f"Transcribed Text: {text['text']}")
    #rating = get_rating(text)
    return text

if __name__ == "__main__":
    video_path = r"K:\project-CMK\demo.mp4"
    text = video_to_text(video_path)
    print("Transcribed Text:")
    print(text)

