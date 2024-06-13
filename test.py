from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence  # , RunnablePassthrough
from langchain.output_parsers.json import SimpleJsonOutputParser

model = Ollama(
    base_url="http://0.0.0.0:7869",
    model="phi3",
)


json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key and `mood` key  that answers the following question: {question} "
)
json_parser = SimpleJsonOutputParser()
json_chain = RunnableSequence(json_prompt, model, json_parser)


result = json_chain.invoke({"question": "Do you want to go to travel with me"})
print(result)
