"""Проверка сборки агента без API-ключа: фейковая LLM и детерминированные эмбеддинги."""

import sys
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import agent  # noqa: E402


def make_retriever():
    chunks = agent.split(agent.load_norms())
    return chunks, agent.build_retriever(chunks, DeterministicFakeEmbedding(size=64), index_dir=None)


def test_norms_are_loaded_with_sources():
    docs = agent.load_norms()
    sources = {d.metadata["source"] for d in docs}
    assert {"ГОСТ 7.0.11-2011", "ГОСТ Р 7.0.5-2008", "Методические указания", "Приказ ректора № 42"} <= sources


def test_hybrid_search_finds_exact_terms():
    _, retriever = make_retriever()
    found = retriever.invoke("Times New Roman межстрочный интервал")
    assert any("Times New Roman" in d.page_content for d in found)
    assert "[ГОСТ 7.0.11-2011]" in agent.format_docs(found)


def test_agent_answers_and_keeps_history():
    _, retriever = make_retriever()
    # первая реплика модели — переписанный запрос, вторая — ответ
    model = FakeListChatModel(responses=[
        "шрифт и интервал основной части", "Times New Roman 14 пт, интервал 1,5 [ГОСТ 7.0.11-2011]",
        "оформление электронного ресурса", "Указывается URL и дата обращения [ГОСТ Р 7.0.5-2008]",
    ])
    bot = agent.build_agent(model, retriever, verbose=False)

    first = agent.ask(bot, "Какой шрифт в дипломе?", session="s")
    second = agent.ask(bot, "А для электронного ресурса?", session="s")

    assert "Times New Roman" in first
    assert "URL" in second
    history = bot.get_session_history("s").messages
    assert len(history) == 4  # два вопроса и два ответа сохранены
