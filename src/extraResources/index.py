import os
import sys
import numpy as np
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
if len(sys.argv) < 2:
    print("No query provided.")
    sys.exit(1)

query = sys.argv[1]
pdf_directory = sys.argv[2]
def load_all_pdfs(pdf_directory):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    docs = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    return docs
docs = load_all_pdfs(pdf_directory)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
doc_splits = text_splitter.split_documents(docs)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
prompt_template = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use maximum sentences and dont hellucinate <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)
model_path = sys.argv[4]
embeddings = GPT4AllEmbeddings(
    model_name=model_path,
    gpt4all_kwargs={'allow_download': False}
)
class Retriever:
    def __init__(self, docs, embeddings):
        self.embeddings = embeddings
        self.docs = docs
        self.embeds = self._embed_docs(docs)
    def _embed_docs(self, docs):
        doc_contents = [doc.page_content for doc in docs]
        embeds = self.embeddings.embed_documents(doc_contents)
        embeds_np = np.array(embeds)
        return embeds_np / np.linalg.norm(embeds_np, axis=1, keepdims=True)

    def query(self, question, k=3):
        query_embed = np.array(self.embeddings.embed_query(question))
        query_embed = query_embed / np.linalg.norm(query_embed)
        similarities = cosine_similarity([query_embed], self.embeds)[0]
        sorted_ix = np.argsort(-similarities)
        return [self.docs[i] for i in sorted_ix[:k]]
retriever = Retriever(doc_splits, embeddings)
retrieved_docs = retriever.query(query)
formatted_docs = format_docs(retrieved_docs)
def run_local_llama(prompt):
    command = ['ollama', 'run', sys.argv[3]]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=prompt.encode())
    if stderr:
        print("Error:", stderr.decode(), file=sys.stderr)
    return stdout.decode()
prompt = prompt_template.format(question=query, context=formatted_docs)
response = run_local_llama(prompt)
print(response)