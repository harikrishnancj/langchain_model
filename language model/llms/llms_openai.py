from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model=OpenAI(model='gpt-3.5-turbo-instruct',temperature=0.5,max_tokens=10)

result=model.invoke("what is the captial of india")

print(result)