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

def get_user_parameters():
    parameters = {}
    parameters['skill_name'] = input("What specific skill do you want to learn? ")
    parameters['current_skill_level'] = input("What is your current proficiency in this skill (beginner, intermediate, advanced)? ")
    parameters['daily_study_time'] = input("How much time can you dedicate to studying this skill each day (in hours)? ")
    parameters['weekly_assessment_schedule'] = input("How frequently would you like to assess your progress (weekly, bi-weekly, monthly)? ")
    parameters['preferred_learning_style'] = input("Do you prefer visual, auditory, reading/writing, or kinesthetic learning methods? ")
    parameters['resources_available'] = input("What resources do you have access to (books, online courses, mentors, practice tools)? ")
    parameters['short_term_goals'] = input("What specific milestones or achievements do you want to reach in the next 1-3 months? ")
    parameters['long_term_goals'] = input("What are your ultimate objectives for learning this skill? ")
    parameters['commitments_and_availability'] = input("Are there any other commitments that might affect your learning schedule (work, family, etc.)? ")
    parameters['motivation_and_interests'] = input("Why do you want to learn this skill? What interests you about it? ")
    parameters['feedback_mechanism'] = input("How would you like to receive feedback (self-assessment, peer review, mentor feedback)? ")
    parameters['challenges_and_barriers'] = input("Are there any known challenges or barriers that might impact your learning process? ")
    parameters['support_system'] = input("Do you have access to a support system (study groups, mentors, online communities)? ")
    parameters['preferred_pace'] = input("Do you prefer a fast-paced or a slow, steady learning approach? ")

    return parameters

def get_road_map(parameters):

    model_path = expanduser("your-model-path-here")

    template_messages = [
        SystemMessage(content="""Imagine you are an analyzer who can be able to provide a precise roadmap for a person who is going to learn a particular skill
                                 Get a series of inputs from the user and give him a perfect roadmap for his learning progress"""),
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

    input_text = f"""
    Skill Name: {parameters['skill_name']}
    Current Skill Level: {parameters['current_skill_level']}
    Daily Study Time: {parameters['daily_study_time']}
    Weekly Assessment Schedule: {parameters['weekly_assessment_schedule']}
    Preferred Learning Style: {parameters['preferred_learning_style']}
    Resources Available: {parameters['resources_available']}
    Short-term Goals: {parameters['short_term_goals']}
    Long-term Goals: {parameters['long_term_goals']}
    Commitments and Availability: {parameters['commitments_and_availability']}
    Motivation and Interests: {parameters['motivation_and_interests']}
    Feedback Mechanism: {parameters['feedback_mechanism']}
    Challenges and Barriers: {parameters['challenges_and_barriers']}
    Support System: {parameters['support_system']}
    Preferred Pace: {parameters['preferred_pace']}
    """

    response = chain.invoke({'text': input_text})

    return response['text']

if __name__ == "__main__":
    user_parameters = get_user_parameters()
    road_map = get_road_map(user_parameters)
    print(road_map)
