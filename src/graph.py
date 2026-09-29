import json
import re
from typing import List, TypedDict

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, START, StateGraph

from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    retrieval_scores: List[float]
    answer: str
    grounded: bool
    score: float


def build_rag_graph():

    # -----------------------------
    # Embedding model
    # -----------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL
    )

    # -----------------------------
    # Pinecone vector store
    # -----------------------------
    vectorstore = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # -----------------------------
    # Groq LLM
    # -----------------------------
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=0
    )

    # -----------------------------
    # RETRIEVE NODE
    # -----------------------------
    def retrieve_node(state: AgentState):

        results = vectorstore.similarity_search_with_score(
            state["question"],
            k=config.TOP_K
        )

        context = []
        scores = []

        for doc, score in results:
            context.append(doc.page_content)
            scores.append(float(score))

        return {
            "context": context,
            "retrieval_scores": scores
        }

    # -----------------------------
    # ROUTING
    # -----------------------------
    def route_after_retrieve(state: AgentState):

        scores = state["retrieval_scores"]

        if not scores:
            return "refuse"

        best_score = max(scores)

        # Only refuse when retrieval is genuinely weak.
        if best_score < config.MIN_RETRIEVAL_SCORE:
            return "refuse"

        return "generate"

    # -----------------------------
    # REFUSAL NODE
    # -----------------------------
    def refuse_node(state: AgentState):

        return {
            "answer": config.REFUSAL_MESSAGE,
            "grounded": False,
            "score": 0.0
        }

    # -----------------------------
    # GENERATE NODE
    # -----------------------------
    def generate_node(state: AgentState):

        context_str = "\n\n---\n\n".join(
            state["context"]
        )

        prompt = f"""
You are answering questions about the Agentic AI eBook.

You MUST answer from the retrieved CONTEXT.

IMPORTANT:
- The CONTEXT below was retrieved specifically for the user's question.
- If the CONTEXT contains ANY relevant information, answer the question.
- Combine information from multiple context chunks when necessary.
- For broad questions, summarize the relevant information found in the CONTEXT.
- Do NOT refuse merely because the exact wording of the question is not present.
- Do NOT use your own outside knowledge.
- Do NOT invent information.
- Only refuse if the CONTEXT contains no information relevant to the question.

If there is genuinely no relevant information, reply exactly:

{config.REFUSAL_MESSAGE}

CONTEXT:
{context_str}

QUESTION:
{state["question"]}

ANSWER:
"""

        response = llm.invoke(prompt)

        answer = response.content.strip()

        return {
            "answer": answer
        }

    # -----------------------------
    # GROUNDEDNESS GRADING
    # -----------------------------
    def grade_node(state: AgentState):

        answer = state["answer"]
        scores = state["retrieval_scores"]

        # Average retrieval score
        avg_retrieval = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        # If the answer is a refusal,
        # confidence must be zero.
        if config.REFUSAL_MESSAGE.lower() in answer.lower():

            return {
                "grounded": False,
                "score": 0.0
            }

        context_str = "\n\n---\n\n".join(
            state["context"]
        )

        grader_prompt = f"""
You are a strict fact-checking grader.

Determine whether the ANSWER is supported by the CONTEXT.

Return ONLY valid JSON in this exact format:

{{
    "groundedness": 0.0,
    "reason": "short reason"
}}

The groundedness value must be between 0.0 and 1.0.

CONTEXT:
{context_str}

ANSWER:
{answer}
"""

        try:

            raw = llm.invoke(
                grader_prompt
            ).content.strip()

            match = re.search(
                r"\{.*\}",
                raw,
                re.DOTALL
            )

            if match:

                data = json.loads(
                    match.group(0)
                )

                groundedness = float(
                    data.get(
                        "groundedness",
                        0.0
                    )
                )

            else:
                groundedness = 0.5

        except Exception:

            groundedness = 0.5

        # Keep value in valid range
        groundedness = max(
            0.0,
            min(1.0, groundedness)
        )

        grounded = groundedness >= 0.7

        # Assignment confidence formula
        final_score = round(
            (
                config.W_RETRIEVAL
                * avg_retrieval
            )
            +
            (
                config.W_GROUNDED
                * groundedness
            ),
            2
        )

        return {
            "grounded": grounded,
            "score": final_score
        }

    # -----------------------------
    # BUILD LANGGRAPH
    # -----------------------------
    workflow = StateGraph(
        AgentState
    )

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_node(
        "grade",
        grade_node
    )

    workflow.add_node(
        "refuse",
        refuse_node
    )

    workflow.add_edge(
        START,
        "retrieve"
    )

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "generate": "generate",
            "refuse": "refuse"
        }
    )

    workflow.add_edge(
        "generate",
        "grade"
    )

    workflow.add_edge(
        "grade",
        END
    )

    workflow.add_edge(
        "refuse",
        END
    )

    return workflow.compile()