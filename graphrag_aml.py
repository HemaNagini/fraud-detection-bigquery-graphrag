import vertexai
from google.cloud import bigquery

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuration
GCP_PROJECT_ID = "fraud-prevention-with-gql"
REGION = "us-central1"
BQ_DATASET_ID = "fingraph_rag"
MODEL_NAME = "gemini-2.5-flash"

# Initialize clients
bq_client = bigquery.Client(project=GCP_PROJECT_ID)
vertexai.init(project=GCP_PROJECT_ID, location=REGION)

print("Clients initialized for", GCP_PROJECT_ID)

class FinGraphRetriever(BaseRetriever):
    project: str
    dataset: str

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1. Vector Search
        vector_query = f"""
            SELECT id, audit_details
            FROM `{self.dataset}.AccountAudits`
            ORDER BY COSINE_DISTANCE(
            embedding,
            (
                SELECT embedding
                FROM AI.GENERATE_EMBEDDING(
                MODEL `{self.dataset}.embedding_model`,
                (SELECT @query AS content)
                )
            )
            )
            LIMIT 1
        """
        res = bq_client.query(vector_query, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("query", "STRING", query)]
        )).result()

        start_id = None
        audit_text = ""
        for row in res:
            start_id = row.id
            audio_text = row.audit_details
        
        if not start_id: return[]

        #2. Native Graph Traversal
        graph_query = f"""
            GRAPH `{self.dataset}.FinGraph`
            MATCH
            (sender_person:Person)-[:Owns]->(sender_acc:Account)
            -[tx:Transfers]->
            (a:Account)
            -[repays:Repays]->(l:Loan),
            (owner:Person)-[:Owns]->(a)
            WHERE a.id = @id
            RETURN
            owner.name as owner_name,
            a.type as account_type,
            sender_person.name as sender_name,
            tx.amount as transfer_amount,
            repays.amount as repayment_amount,
            l.id as loan_id
        """
        graph_res = bq_client.query(graph_query, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("id", "INT64", start_id)]
        )).result()

        context_docs = [Document(page_content=f"Primary Audit Log (Target Account): {audit_text}")]
        sender_names=[]
        for row in graph_res:
            sender_names.append(row['sender_name'])
            doc_str = (f"Account Owner: {row['owner_name']} (Account Type: {row['account_type']}). "
                       f"Received transfer of ${row['transfer_amount']} from {row['sender_name']}. "
                       f"Made loan repayment of ${row['repayment_amount']} to Loan {row['loan_id']}.")
            context_docs.append(Document(page_content=doc_str))
        if sender_names:
            names_list = "','".join(sender_names)
            sender_audit_query = f"""
                SELECT p.name, au.audit_details
                FROM `{self.dataset}.AccountAudits` au
                JOIN `{self.dataset}.Account` a ON au.id = a.id
                JOIN `{self.dataset}.PersonOwnAccount` poa ON a.id = poa.account_id
                JOIN `{self.dataset}.Person` p ON poa.id = p.id
                WHERE p.name IN ('{names_list}')
            """
            sender_audits = bq_client.query(sender_audit_query).result()
            for row in sender_audits:
                context_docs.append(Document(page_content=f"Audit Log for Sender {row['name']}: {row['audit_details']}"))

        return context_docs

# Initialize LLM & Retriever
llm = ChatVertexAI(model_name=MODEL_NAME)
retriever = FinGraphRetriever(project=GCP_PROJECT_ID, dataset=BQ_DATASET_ID)
prompt = ChatPromptTemplate.from_template("""
You are a Lead Fraud Analyst. use the floowing audit logs and graph transaction history to answer the question.
Your goal is to connect the dots between the entities and explain the flow of funds.
If you see transfers from flagged users or shell companies, highlight the money laundering risk.

Context: {context}

Question: {question}

Detailed Fraud Report:
""")

chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    |llm
    | StrOutputParser()
)
question = "Why is Menville's loan repayment at risk? Flag any suspicious activityif you notice."
print(chain.invoke(question))