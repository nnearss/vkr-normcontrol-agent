"""
ИИ-агент нормоконтроля ВКР.

Отвечает на вопросы студента об оформлении выпускной работы строго по нормативным
документам из папки norms/ и проверяет фрагменты текста. Под капотом:
гибридный поиск (FAISS + BM25), переформулировка вопроса с учётом диалога и память.

Запуск:
    python agent.py               # интерактивный чат
    python agent.py --demo        # прогон тестовых вопросов
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

NORMS_DIR = Path(__file__).parent / "norms"
INDEX_DIR = Path(__file__).parent / ".faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SYSTEM_PROMPT = (
    "Ты ИИ-агент нормоконтроля ВКР.\n"
    "Правила:\n"
    "1. Отвечай ТОЛЬКО по контексту из нормативных документов.\n"
    "2. Если информации нет, скажи: «Данная информация отсутствует в регламентах».\n"
    "3. Всегда указывай источник в квадратных скобках, например [ГОСТ 7.0.11-2011].\n"
    "4. При проверке фрагмента дай структурированную рецензию: что соответствует, что нет, как исправить.\n\n"
    "Контекст:\n{context}"
)

REWRITE_PROMPT = (
    "Переформулируй вопрос студента в самодостаточный поисковый запрос по нормативной базе, "
    "учитывая историю диалога. Верни только запрос."
)


# ---------- документы ----------

def load_norms(folder: Path = NORMS_DIR) -> list[Document]:
    """Загружает нормативные документы: .txt и .md как текст, .pdf постранично."""
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip()
            docs.append(Document(page_content=text, metadata={"source": title}))
        elif path.suffix.lower() == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            for page in PyPDFLoader(str(path)).load():
                page.metadata["source"] = path.stem
                docs.append(page)
    if not docs:
        raise FileNotFoundError(f"В папке {folder} нет документов .txt, .md или .pdf")
    return docs


def split(docs: list[Document]) -> list[Document]:
    return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80).split_documents(docs)


# ---------- поиск ----------

def build_retriever(chunks: list[Document], embeddings, k: int = 4, index_dir: Path | None = INDEX_DIR):
    """Гибридный поиск: смысловой (FAISS) + по точным словам (BM25), объединение через RRF."""
    if index_dir and index_dir.exists():
        faiss_store = FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)
    else:
        faiss_store = FAISS.from_documents(chunks, embeddings)
        if index_dir:
            faiss_store.save_local(str(index_dir))
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k
    return EnsembleRetriever(
        retrievers=[faiss_store.as_retriever(search_kwargs={"k": k}), bm25],
        weights=[0.5, 0.5],
    )


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


# ---------- агент ----------

def build_agent(model, retriever, verbose: bool = True):
    rewriter = ChatPromptTemplate.from_messages([
        ("system", REWRITE_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "Вопрос: {question}\nПоисковый запрос:"),
    ]) | model | StrOutputParser()

    def log_query(query: str) -> str:
        if verbose:
            print(f"  [поиск] {query}")
        return query

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": rewriter | RunnableLambda(log_query) | retriever | format_docs,
            "question": lambda x: x["question"],
            "history": lambda x: x["history"],
        }
        | answer_prompt
        | model
        | StrOutputParser()
    )

    sessions: dict[str, ChatMessageHistory] = {}

    def get_history(session_id: str) -> ChatMessageHistory:
        return sessions.setdefault(session_id, ChatMessageHistory())

    return RunnableWithMessageHistory(
        chain, get_history, input_messages_key="question", history_messages_key="history"
    )


def ask(agent, question: str, session: str = "student") -> str:
    return agent.invoke({"question": question}, config={"configurable": {"session_id": session}})


# ---------- запуск ----------

DEMO = [
    ("t1", "Какой шрифт и интервал в основной части диплома?"),
    ("t2", "Как оформлять список литературы по ГОСТу?"),
    ("t2", "А если это электронный ресурс?"),
    ("t3", "Проверь фрагмент введения: «Целью данной работы является исследование MCP и разработка агента». Напиши замечания."),
    ("t4", "Каковы правила оформления дипломов в Гарвардском университете?"),
]


def make_model():
    from langchain_openai import ChatOpenAI

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Не найден OPENAI_API_KEY. Скопируйте .env.example в .env и вставьте ключ.")
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
    )


def main():
    parser = argparse.ArgumentParser(description="ИИ-агент нормоконтроля ВКР")
    parser.add_argument("--demo", action="store_true", help="прогнать тестовые вопросы")
    args = parser.parse_args()

    model = make_model()  # сразу проверяем ключ, до долгой загрузки эмбеддингов

    from langchain_huggingface import HuggingFaceEmbeddings

    print("Загружаю нормативную базу и модель эмбеддингов...")
    chunks = split(load_norms())
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})
    agent = build_agent(model, build_retriever(chunks, embeddings))
    print(f"Готово: {len(chunks)} фрагментов из папки norms/\n")

    if args.demo:
        for session, question in DEMO:
            print(f"Вы: {question}")
            print(f"Агент: {ask(agent, question, session)}\n")
        return

    print("Задайте вопрос об оформлении ВКР. Для выхода введите /exit\n")
    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() == "/exit":
            break
        print(f"\nАгент: {ask(agent, question)}\n")
    print("Удачи с защитой! 🎓")


if __name__ == "__main__":
    main()
