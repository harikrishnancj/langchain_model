from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4')
qus=['']

result=model.invoke(qus)

print(result.content)