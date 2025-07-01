from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
    task="text-generation",                        
    max_new_tokens=512,
    top_k=10,
    temperature=0.7,
    repetition_penalty=1.0
)

model=ChatHuggingFace(llm=llm)

result=model.invoke(
    'what need to be asked'
)

print(result.content)
