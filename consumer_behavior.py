# Шаг 1. Импорт библиотек
import os
import logging
import json
import datetime
import pickle
import telebot
from telebot import types
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

# Получение значений переменных окружения
openai_api_key = os.getenv('OPENAI_API_KEY')
telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

# Шаг 2. Настройка векторной базы данных
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base='https://api.proxyapi.ru/openai/v1')
DATABASE = Chroma(persist_directory='DataBase', embedding_function=embedding_model)
documents1 = []
for i in DATABASE.get()['documents']:
    doc = Document(page_content=i)
    documents1.append(doc)
bm25_retriever = BM25Retriever.from_documents(documents1)
bm25_retriever.k = 10

# Шаг 3. Настройки логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Шаг 4. Словарь для хранения истории чата в сессии, список для хранения полной истории
uniq = datetime.datetime.now()
if not os.listdir('UserStates'):
    store = {}
else:
    files = [os.path.join('UserStates', f) for f in os.listdir('UserStates') if
             os.path.isfile(os.path.join('UserStates', f))]
    latest_file = max(files, key=os.path.getctime)
    with open(latest_file, 'rb') as file:
        store = pickle.load(file)
print(store)

passwords = pd.read_excel('Adjusted_Generated_Passwords_ConsBeh.xlsx', dtype='str')
passwords = passwords.fillna('')

# Инициализация бота
bot = telebot.TeleBot(telegram_bot_token)

# Шаг 5. Функция для определения истории чата
def get_session_history(session_id: str, k=3) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    if len(store[session_id].messages) / 2 > k:
        del store[session_id].messages[0:2]
    return store[session_id]

whole_chat_hist = []

# Шаг 6. Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id,
                     '''Привет! Я - эксперт МШУ "Сколково" по поведению потребителей! Вы можете задавать мне любые вопросы по курсу!\n\nПожалуйста, оставьте свою обратную связь по следующей ссылке: https://forms.yandex.ru/u/6729d6a4eb614697263f2079/''')
    user_id = str(message.chat.id)
    if user_id not in passwords['USER_ID'].values:
        bot.send_message(message.chat.id, '''Для авторизации отправьте мне пароль в следующем сообщении''')

# Шаг 9. Обработчик команды /clear_history
@bot.message_handler(commands=['clear_history'])
def clear_history(message):
    store[message.chat.id] = ChatMessageHistory()
    bot.send_message(message.chat.id, "Память очищена!")

# Шаг 7. Обработка авторизации
def handle_verification(message):
    global passwords
    user_id = str(message.chat.id)
    response = message.text
    if (response in passwords['Password'].values) and (passwords[passwords['Password'] == response]['USER_ID'].values[0] == ''):
        ind = passwords[passwords['Password'] == response].index[0]
        passwords.loc[ind, 'USER_ID'] = user_id
        print(passwords.head())
        bot.send_message(message.chat.id, 'Авторизация пройдена!\nМожете задавать вопросы.')
    else:
        if message.chat.id not in store:
            bot.send_message(message.chat.id, 'Неправильный пароль! Попробуйте еще раз.')
        else:
            bot.send_message(message.chat.id,
                             'Авторизация не пройдена! Для диалога с ботом, пожалуйста, введите пароль.')

# Шаг 8. Обработка запроса пользователя
@bot.message_handler(func=lambda message: True, content_types=['text'])
def query(message):
    global whole_chat_hist
    global store
    global passwords
    logging.info(f"Processing query from user {message.chat.id}")
    logging.info(f"Processing query from message {message}")

    user_id = str(message.chat.id)
    print(passwords['USER_ID'].values)
    print(user_id)
    if user_id not in passwords['USER_ID'].values:
        handle_verification(message)
    else:
        thinking_message = bot.reply_to(message, f"⏳ Бот обрабатывает ваш запрос...")

        processing = request_processing(message)
        result = processing['answer']
        print(processing['context'])

        qa = {'id': message.chat.id, 'question': message.text, 'answer': result}
        whole_chat_hist.append(qa)

        with open('ChatHistory/' + uniq.strftime("%d-%m-%Y-%H-%M") + ' chat_history.json', 'w') as file:
            file.write(json.dumps(whole_chat_hist))
        with open('UserStates/' + uniq.strftime("%d-%m-%Y-%H-%M") + ' user_state.pkl', 'wb') as file:
            file.write(pickle.dumps(store))
        passwords.loc[:, 'USER_ID'] = passwords['USER_ID'].astype('str')
        passwords.to_excel('Adjusted_Generated_Passwords.xlsx', index=False)

        bot.delete_message(message.chat.id, thinking_message.message_id)
        bot.send_message(message.chat.id, result)

# Шаг 10. Основная функция для обработки запроса
def request_processing(message):
    global DATABASE

    chatbot_purpose = f"# Роль\nВы являетесь доктором наук в области когнитивной и поведенческой психологии с обширным опытом в исследованиях, ориентированных на бизнес. Вы пишете энциклопедические статьи по когнитивной, поведенческой и нейропсихологии, маркетингу, исследованию пользователей и потребителей, наддированию (nudging) и геймификации по запросу пользователя."
    skills = """## Навыки
    ### Навык 1: Написание статей
    - Пользователь делает запрос на русском или английском языке, а вы пишете длинную, подробную, хорошо структурированную статью на русском языке, опираясь исключительно на свои фрагменты памяти и предоставленный вам контекст из базы знаний.
    - Сохраняйте текст фрагментов памяти и контекст из базы знаний как можно точнее, интегрируя фрагменты друг с другом и объединяя их для поддержания логического повествования и сохранения максимального количества оригинального текста.
    - Не выдумывайте информацию; интегрируйте только то, что доступно в вашем знании.
    - Дублируйте научные и бизнес-термины (отдельные слова или устойчивые фразы) на английском языке в скобках.

    ### Навык 2: Структурирование информации
    - Сначала суммируйте все фрагменты памяти в виде структурированного плана (аналогично плану википедийной статьи), обязательно включив разделы "Исследовательские направления" и "Ключевые слова".
    - Затем напишите статью, объединяя фрагменты памяти так, чтобы ни один фрагмент не был утерян.
    - При необходимости добавьте ссылки в стиле Чикаго.

    ### Навык 3: Исследовательские перспективы
    - В предпоследнем абзаце статьи приведите список потенциальных исследовательских направлений, вопросов и тем для будущих исследований, вытекающих из содержания статьи.
    - Игнорируйте этические аспекты, сосредоточившись на экспериментальных (лабораторных и полевых) исследованиях.

    ### Навык 4: Ключевые слова
    - В конце статьи укажите список ключевых слов и фраз, разделенных запятыми (не в виде списка).
"""

    constraints1 = '''## Ограничения:
- Используйте только информацию, доступную в вашем контексте; не добавляйте вымышленную информацию.
- В конце процесса всегда убедитесь, что вы написали полную статью, а не только ее план. Также проверьте, что статья написана на русском языке. Убедитесь, что статья начинается с плана, а основной текст следует за ним.
- Все статьи всегда должны быть написаны на русском языке, независимо от языка запроса.'''

    system_prompt = chatbot_purpose + skills +constraints1
    ensemble_retriever = DATABASE.as_retriever(search_kwargs={'k': 15})

    model = ChatOpenAI(model='gpt-4o-mini', max_tokens=1024, api_key=openai_api_key,
                       base_url='https://api.proxyapi.ru/openai/v1')
    contextualize_q_system_prompt = (
        "Для ответа на вопросы пользователя учитывай историю чата и последний вопрос пользователя"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        model, ensemble_retriever, contextualize_q_prompt
    )

    system_prompt_new = (
        "Ты персона, которая описывается ниже:\n" + system_prompt + "Пользователь хочет знать:\n{input}\n.Используй следующие фрагменты извлеченного контекста из твоей базы знаний, чтобы максимально точно ответить на вопрос \n{context}\n"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_new),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    inv = conversational_rag_chain.invoke(
        {"input": message.text},
        config={
            "configurable": {"session_id": message.chat.id}
        },
    )
    return inv

# Шаг 11. Запуск бота
if __name__ == '__main__':
    bot.polling(none_stop=True)