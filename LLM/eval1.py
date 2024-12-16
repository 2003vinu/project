import csv
import re
from os.path import expanduser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain_core.messages import SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_community.llms import LlamaCpp

def get_summary(text):
    model_path = expanduser("llama-2-13b-chat.Q8_0.gguf")

    template_messages = [
        SystemMessage(content="""Imagine yourself as reviewer who can analyze a set of reviews of a video, and 
                                 provide a summarized text for those reviews that are given as input.
                                 Your text will be posted in review section below the video, so 
                                 give a precise review and don't add additional dialogue. Just be straight to the review."""),
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

    response = chain.invoke({"text": text})
    print(response['text'])
    quoted_text = re.findall(r'"([^"]*)"', response['text'])
    return quoted_text[0] if quoted_text else ""

def summarize_text(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        comments = [row[0] for row in reader if row]
    
    token_limit = 3777
    all_summaries = []
    chunk = []

    def count_tokens(text):
        return len(text.split())

    current_tokens = 0
    for comment in comments:
        comment_tokens = count_tokens(comment)
        if current_tokens + comment_tokens > token_limit:
            chunk_text = " ".join(chunk)
            summary = get_summary(chunk_text)
            all_summaries.append(summary)
            chunk = [comment]
            current_tokens = comment_tokens
        else:
            chunk.append(comment)
            current_tokens += comment_tokens

    if chunk:
        chunk_text = " ".join(chunk)
        summary = get_summary(chunk_text)
        all_summaries.append(summary)

    while len(all_summaries) > 1:
        new_summaries = []
        current_tokens = 0
        chunk = []
        for summary in all_summaries:
            summary_tokens = count_tokens(summary)
            if current_tokens + summary_tokens > token_limit:
                chunk_text = " ".join(chunk)
                summary = get_summary(chunk_text)
                new_summaries.append(summary)
                chunk = [summary]
                current_tokens = summary_tokens
            else:
                chunk.append(summary)
                current_tokens += summary_tokens
        if chunk:
            chunk_text = " ".join(chunk)
            summary = get_summary(chunk_text)
            new_summaries.append(summary)
        all_summaries = new_summaries

    final_summary = all_summaries[0] if all_summaries else ""

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([final_summary])

    return final_summary

if __name__ == "__main__":
    csv_path = r"K:\project-CMK\cmt.csv"
    text = summarize_text(csv_path)
    print("Summarized Review:")
    print(text)
