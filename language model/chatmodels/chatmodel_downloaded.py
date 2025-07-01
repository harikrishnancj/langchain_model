from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline( model="tiiuae/falcon-7b-instruct",   
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7)


chat_model = ChatHuggingFace(llm=llm)
response = chat_model.invoke("What is the capital of France?")
print(response)
