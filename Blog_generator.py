import os 
import openai
import env

from openai import OpenAI

os.environ["OPENAI_API_KEY"] = env.OpenAI_API_KEY
client = OpenAI()

#print(client.models.list())

client.completions.create(
    model='gpt-3.5-turbo-instruct',
    prompt="Tell me a story about talking care"
)