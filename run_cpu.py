import time, os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader

c_w_d = os.getcwd()
dataset = os.path.join(c_w_d, "dataset/current_cp_dataset.csv")
model = os.path.join(c_w_d, "model/openchat-3.5-0106.Q4_K_M.gguf")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader = CSVLoader(file_path=dataset, encoding="utf-8", csv_args={'delimiter': ','})
pages = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=64,
    separators=['\n\n', '\n', '(?=>\. )', ' ', '']
)
docs = text_splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/average_word_embeddings_glove.6B.300d", model_kwargs={"device": device})


llm = LlamaCpp(
    model_path=model,
    temperature=1,
    verbose=False,
    max_tokens=10,
    stop=["Q:", "\n"],
    echo=True
)

chain = load_qa_chain(llm, chain_type="stuff")

db = FAISS.from_documents(docs, embeddings)


def gpt_three_point_five(query_input):

    docs = db.similarity_search(query_input)
    ans = chain.run(input_documents=docs, question=query_input + ", give me the intent only")
    return ans

if __name__ == "__main__":
    while True:
        query = input("Enter your question: ")
        s = time.time()
        response = gpt_three_point_five(query)
        print(f"response: {response}")
        print("execute time :", time.time() - s)