# Bot GUI
import threading
import pandas as pd
import json
import pickle
import nltk
from nltk.corpus import stopwords
import sklearn
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import *
from random import randrange
from pathlib import Path
import gensim
import psycopg2

# define directories
path = './Data/'
entities_model_dir = './Data/entities_model.pickle'
intents_model_dir = './Data/intents_model.pickle'
intents_dir = './Data/intents_mod.json'
data_tokens = pd.DataFrame()
chat_context = {
    "intent": "",
    "libName": "",
    "lang": "",
    "w2vInput":""
}
waitResponses = ["Give me a second...", "Please wait while I find a good answer...", "Just a bit longer...", "Hang in there with me...", "I'm trying my best...", "Looking for an answer...", "Almost there..."]
botReply = ""

# load trained data
def load_trained_data():
    print('start loading trained data')
    try:
        desc_json_path = path + 'Desc_Word2Vec.json'

        with open(desc_json_path) as file:
            reader = json.load(file)

            langs = []
            descriptions = []
            # desc_tokens = []
            ids = []
            names = []
            repourls = []
            desc_lengths = []
            # desc_vectors = []
            average_pooling = []
            for row in reader:
                langs.append(row['Lang'])
                descriptions.append(row['Description'])
                # desc_tokens.append(row['Desc_Tokens'].split())
                ids.append(row['ID'])
                names.append(row['Name'])
                repourls.append(row['RepoUrl'])
                desc_lengths.append(row['Desc_Length'])
                # desc_vectors.append(row['Desc_Vectors'])
                average_pooling.append(row['Average_Pooling'])
            global data_tokens
            data_tokens = pd.DataFrame({'Lang': langs,
                                        'Description': descriptions,
                                        # 'Desc_Tokens': desc_tokens,
                                        'ID': ids,
                                        'Name': names,
                                        'RepoUrl': repourls,
                                        'Desc_Length': desc_lengths,
                                        # 'Desc_Vectors': desc_vectors,
                                        'Average_Pooling': average_pooling})
    except:
        pass
    print('data loaded')


def clean_up_sentence(sentence):
    stop_words = stopwords.words("english")

    # Remove non english words
    sentence = [re.sub('[^a-z(JavaScript)(Python)(Java)(PHP)(C++)]', ' ', x.lower()) for x in sentence]
    # Tokenization
    sentence_tokens = [nltk.word_tokenize(t) for t in sentence]
    # Removing Stop Words
    sentence_tokens = [[t for t in tokens if (t not in stop_words) and (3 < len(t.strip()) < 15)]
                       for tokens in sentence_tokens]

    sentence_tokens = pd.Series(sentence_tokens)
    return sentence_tokens


def match_repo_to_input(data_by_language, model, msg_tokens):
    cosines = []
    try:
        # Get vectors and average pooling
        question_vectors = []
        for token in msg_tokens:
            try:
                vector = model[token]
                question_vectors.append(vector)
            except:
                continue
        question_ap = list(pd.DataFrame(question_vectors[0]).mean())

        # Calculate cosine similarity
        for t in data_by_language['Average_Pooling']:
            if t is not None and len(t) == len(question_ap):
                val = cosine_similarity([question_ap], [t])
                cosines.append(val[0][0])
            else:
                cosines.append(0)
    except:
        pass

    # If not in the topic trained
    if len(cosines) == 0:
        not_understood = "I'm sorry I could not find anything that you might like."
        return not_understood, 999

    else:
        # Sort similarity
        index_s = []
        score_s = []
        for i in range(len(cosines)):
            x = cosines[i]
            if x >= 0.9:
                index_s.append(i)
                score_s.append(cosines[i])

        reply_indexes = pd.DataFrame({'index': index_s, 'score': score_s})
        reply_indexes = reply_indexes.sort_values(by="score", ascending=False)

        if len(reply_indexes) == 0:
            not_understood = "I'm sorry I could not find anything that you might like."
            return not_understood, 999
        else:
            # Find Top Questions and Score
            r_index = int(reply_indexes['index'].iloc[0])
            r_score = float(reply_indexes['score'].iloc[0])

            # reply = str(data_by_language.iloc[:,0][r_index])
            reply = str("My suggestion: " + data_by_language['Description'].iloc[r_index] + ". Repo name: " +
                        data_by_language['Name'].iloc[r_index] + ". Repo URL: " + data_by_language['RepoUrl'].iloc[
                            r_index])
            global chat_context
            chat_context["libName"] = data_by_language['Name'].iloc[r_index]
            return reply, r_score

def query_db(user_intent, query):
    global chat_context
    resp = ""
    try:
        connection = psycopg2.connect(user = "postgres",
                                    password = "1qaz!QAZ",
                                    host = "114.203.211.52",
                                    port = "49153",
                                    database = "anhdb")

        cursor = connection.cursor()
        cursor.execute(query)
        record = cursor.fetchone()
        #process each intent here
        if record:
            if user_intent == 'popularity':
                baseAnswer = 'The library {} has {} stars, {} forks, {} active issues, {} people watching and {} contributors'
                resp = baseAnswer.format(record[0], record[1], record[2], record[3], record[4], record[5])
            elif user_intent == 'lib_author':
                baseAnswer = 'The author of library {} is {}'
                resp = baseAnswer.format(record[0].split('/')[1], record[0].split('/')[0])
            chat_context["libName"] = record[0]
        else:
            resp = "Sorry I could not find what you are looking for. Please make sure your input is correct."

    except (Exception, psycopg2.Error) as error :
        resp = "Sorry I am having problems connecting to server, please come back later."
    finally:
        #closing database connection.
            if(connection):
                cursor.close()
                connection.close()
    return resp

def process_generic_response(user_intent):
    json_file = open(intents_dir)
    intents = json.load(json_file)

    for intent in intents['intents']:
        if intent['tag'] == user_intent:
            return intent['responses'][generate_response(len(intent['responses']))]

    return 'Sorry I could not catch what you were saying. Can you rephrase?'

def process_query_response(user_intent, lib_name):
    resp = 'Sorry I could not catch what you were saying. Can you rephrase?'
    if user_intent == 'popularity':
        base_query = "SELECT repository_name, repository_stars, repository_forks, repository_open_issues, repository_watches, repository_contributors\
            FROM projects_with_repository_fields WHERE repository_name like '%{}%'\
            AND repository_name IS NOT NULL\
            AND repository_stars IS NOT NULL\
            AND repository_forks IS NOT NULL\
            AND repository_open_issues IS NOT NULL\
            AND repository_watches IS NOT NULL\
            AND repository_contributors IS NOT NULL\
            ORDER BY repository_stars DESC"
        query = base_query.format(lib_name)
        resp = query_db(user_intent, query)
    elif user_intent == 'lib_author':
        base_query = "SELECT repository_name FROM projects_with_repository_fields WHERE repository_name like '%{}%' AND repository_name IS NOT NULL ORDER BY repository_stars DESC"
        query = base_query.format(lib_name)
        resp = query_db(user_intent, query)

    return resp

def process_libsearch_response(language, message):
    global botReply
    data_by_language = data_tokens[data_tokens['Lang'] == language]
    data_by_language = pd.DataFrame({'Description': list(data_by_language['Description']),
                                     # 'Desc_Tokens': list(data_by_language['Desc_Tokens']),
                                     'ID': list(data_by_language['ID']),
                                     'Name': list(data_by_language['Name']),
                                     'RepoUrl': list(data_by_language['RepoUrl']),
                                     'Lang': list(data_by_language['Lang']),
                                     # 'Desc_Vectors': list(data_by_language['Desc_Vectors']),
                                     'Average_Pooling': list(data_by_language['Average_Pooling'])
                                     })
    word2vec_pickle_path = path + 'desc_word2vec_' + language + '.bin'
    model = gensim.models.KeyedVectors.load(word2vec_pickle_path)
    msg_tokens = clean_up_sentence(pd.Series(message))
    reply, score = match_repo_to_input(data_by_language, model, msg_tokens)
    botReply = reply
    return reply

def generate_response(num):
    return randrange(num)

def chatbot_response(msg):
    global chat_context
    global botReply
    botReply = ""
    # check intent
    f = open(intents_model_dir, 'rb')
    intents_model = pickle.load(f)

    user_intent = intents_model.predict([msg])
    if user_intent == 'greeting' or user_intent == 'goodbye' or user_intent == 'thanks' or user_intent == 'no_answer' \
        or user_intent == 'options':
        botReply = process_generic_response(user_intent)

    elif user_intent == 'popularity' or user_intent == 'lib_author':
        stop_words = stopwords.words("english")
        #identify ner
        ner = spacy.load(Path(path))
        doc = ner(msg)
        libName = ""
        for ent in doc.ents:
            if ent.label_ == "libName" and ent.text not in stop_words:
                libName = ent.text
                # chat_context["libName"] = ent.text
                botReply = process_query_response(user_intent, ent.text)
                break

        if libName == "" and chat_context["libName"] != "":
            botReply = process_query_response(user_intent, chat_context["libName"])

    elif user_intent == 'lib_search':
        f = open(entities_model_dir, 'rb')
        entities_model = pickle.load(f)
        language = entities_model.predict([msg])
        if language:
            if language[0] == 'Python' or language[0] == 'JavaScript' or language[0] == 'Java' or language[0] == 'PHP' or language[0] == 'C++': 
                botReply =  process_libsearch_response(language[0], msg)
            else:
                botReply = 'Sorry, I can only help you with Python, JavaScript, Java, PHP or C++'
        else:
            botReply = 'Sorry I could not catch what you were saying. Can you rephrase?'

    else:
        botReply = 'Sorry I could not catch what you were saying. Can you rephrase?'

class MyThread(threading.Thread):
    def __init__(self, msg, chat_log, send_button):
        threading.Thread.__init__(self)
        self.msg = msg
        self.status = 0
        self.chat_log = chat_log
        self.send_button = send_button
    
    def run(self):
        self.status = 1
        print("Starting")
        
        chatbot_response(self.msg)
        self.status = 2
    
    def wait_response(self):
        global botReply
        
        print('here', self.status, botReply)

        if self.status == 2:
            self.chat_log.config(state=NORMAL)
            self.chat_log.insert(END, "Bot: " + botReply + '\n\n')
            self.chat_log.config(state=DISABLED)
            self.chat_log.yview(END)
            self.chat_log.update_idletasks()
            
            send_button["state"] = "normal"
        else:
            self.chat_log.config(state=NORMAL)
            self.chat_log.insert(END, "Bot: " + waitResponses[generate_response(len(waitResponses))] + '\n\n')
            self.chat_log.config(state=DISABLED)
            self.chat_log.yview(END)
            self.chat_log.after(5000, self.wait_response)

# Creating GUI with tkinter
def send(master):
    global botReply
    global send_button
    
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if msg != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))
        chat_log.config(state=DISABLED)
        chat_log.yview(END)
        
        send_button["state"] = "disabled"
        
        t = MyThread(msg, chat_log, send_button)
        t.start()

        chat_log.after(2000, t.wait_response)

load_trained_data()
print('loading ui')
# UI Main
base = Tk()
base.title("Software Lookup Bot")
# base.destroy()
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
chat_log = Text(base, bd=0, bg="white", height="8", width="50", foreground="#442265", font=("Verdana", 12) )
chat_log.insert(END, "Bot: Hello how may I help you today?\n\n")
chat_log.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=chat_log.yview, cursor="heart")
chat_log['yscrollcommand'] = scrollbar.set

# Create Button to send message
send_button = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                     bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                     command= lambda: send(base))

# Create the box to enter message
entry_box = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
#entry_box.bind("<Return>", send)

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
chat_log.place(x=6, y=6, height=386, width=370)
entry_box.place(x=128, y=401, height=90, width=265)
send_button.place(x=6, y=401, height=90)
base.mainloop()
