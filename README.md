### 🎓 ии-агент нормоконтроля вкр · thesis formatting assistant

агент на langchain, который отвечает на вопросы об оформлении диплома строго по гостам и методичке, помнит контекст диалога и проверяет фрагменты текста. если ответа в документах нет, он так и говорит, а не выдумывает
*a langchain agent that answers thesis formatting questions strictly from the official standards, keeps dialogue context and reviews text fragments. if the answer isn't in the documents, it says so instead of making things up*

`python` `langchain` `rag` `faiss` `bm25` `llm` `openai api` `pytest`

---

**как работает** · *how it works*

```
вопрос → переформулировка с учётом диалога → гибридный поиск (faiss + bm25) → фрагменты норм → llm → ответ со ссылкой на источник
question → rewrite with dialogue context → hybrid search → norm excerpts → llm → answer with source
```

- **гибридный поиск** · *hybrid search*: faiss ищет по смыслу, bm25 по точным словам вроде «гост 7.0.5». результаты объединяются методом reciprocal rank fusion
- **переформулировка запроса** · *query rewriting*: «а если это электронный ресурс?» превращается в самостоятельный поисковый запрос с учётом прошлого вопроса
- **память** · *memory*: у каждой сессии своя история диалога
- **защита от галлюцинаций** · *grounding*: агент отвечает только по найденным фрагментам и всегда называет источник
- **мультиязычные эмбеддинги** · *multilingual embeddings*: модель `paraphrase-multilingual-MiniLM-L12-v2` нормально понимает русский текст

---

**пример диалога** · *example*

так выглядит работа агента, точные формулировки ответов зависят от модели
*an illustration; exact wording depends on the model*

```
Вы: Как оформлять список литературы по ГОСТу?
  [поиск] оформление списка литературы ГОСТ
Агент: Список составляется в алфавитном порядке. Книга: Автор. Название. —
       Город: Издательство, Год. — Количество страниц с. [ГОСТ Р 7.0.5-2008]

Вы: А если это электронный ресурс?
  [поиск] оформление электронного ресурса в списке литературы ГОСТ
Агент: Автор. Название [Электронный ресурс]. — URL: адрес (дата обращения:
       ДД.ММ.ГГГГ). [ГОСТ Р 7.0.5-2008]

Вы: Каковы правила оформления дипломов в Гарварде?
Агент: Данная информация отсутствует в регламентах.
```

---

**как запустить** · *how to run*

```bash
pip install -r requirements.txt
cp .env.example .env          # вставьте ключ openai или совместимого сервиса
python agent.py               # чат в терминале, /exit для выхода
python agent.py --demo        # прогон тестовых вопросов
```

нормативные документы лежат в папке [`norms/`](norms/). сейчас там учебные выдержки из гостов и методички, но можно положить свои `.pdf`, `.md` или `.txt`, и агент подхватит их при следующем запуске (удалите папку `.faiss_index`, чтобы пересобрать индекс)
*the knowledge base lives in [`norms/`](norms/): training excerpts for now, but you can drop in your own `.pdf`, `.md` or `.txt` files (delete `.faiss_index` to rebuild the index)*

**тесты** · *tests*: `pytest` собирает агента на фейковой модели и проверяет загрузку норм, поиск, ответ и память, ключ api не нужен
*`pytest` builds the agent with a fake llm and checks loading, search, answering and memory, no api key needed*

---

итоговый проект курса «ии-агенты на langchain», мтуси, 2026. в курсе агент был ячейкой ноутбука, я переделала его в скрипт с отдельной базой документов, настройками через `.env` и тестами
*final project of a langchain agents course, mtuci, 2026. i turned the notebook cell into a script with a separate document base, `.env` config and tests*
