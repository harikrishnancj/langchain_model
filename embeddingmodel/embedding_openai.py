from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

emb=OpenAIEmbeddings(model='',dimensions=32)

#doc=['sajjddajs',
   #  "ahdaijdad",]
result=emb.embed_query("QUERy")
#result=emb.aembed_documents(doc)

print(str(result))
