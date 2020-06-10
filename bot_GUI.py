# Bot GUI
import threading
import pandas as pd
import json
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import *
from random import randrange

# define directories
path = './Data/'
entities_model_dir = './Data/entities_model.pickle'
intents_model_dir = './Data/intents_model.pickle'
intents_dir = './Data/intents_mod.json'


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
    main_ui()


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
        not_understood = "Apology, I do not understand. Can you rephrase?"
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
            not_understood = "Apology, I do not understand. Can you rephrase?"
            return not_understood, 999
        else:
            # Find Top Questions and Score
            r_index = int(reply_indexes['index'].iloc[0])
            r_score = float(reply_indexes['score'].iloc[0])

            # reply = str(data_by_language.iloc[:,0][r_index])
            reply = str("My suggestion: " + data_by_language['Description'].iloc[r_index] + ". Repo name: " +
                        data_by_language['Name'].iloc[r_index] + ". Repo URL: " + data_by_language['RepoUrl'].iloc[
                            r_index])

            return reply, r_score


def process_response(user_intent):
    json_file = open(intents_dir)
    intents = json.load(json_file)

    for intent in intents['intents']:
        if intent['tag'] == user_intent:
            return intent['responses'][generate_response(len(intent['responses']))]

    return 'intent not found'


def generate_response(num):
    return randrange(num)


def chatbot_response(msg):
    # TODO add code to determine lang here, for now hardcode as python
    # check intent
    f = open(intents_model_dir, 'rb')
    intents_model = pickle.load(f)

    user_intent = intents_model.predict([msg])

    if user_intent == 'greeting' or user_intent == 'goodbye' or user_intent == 'thanks' or user_intent == 'no_answer' \
        or user_intent == 'options':
        return process_response(user_intent)

    elif user_intent == 'lib_search':
        # TODO pass search query to ner model
        print('searching lib')

    # based on response from intent model, either response with a text or an action
    f = open(entities_model_dir, 'rb')
    entities_model = pickle.load(f)

    language = entities_model.predict([msg])

    # data_by_language = data_tokens[data_tokens['Lang'] == language]
    # data_by_language = pd.DataFrame({'Description': list(data_by_language['Description']),
    #                                  # 'Desc_Tokens': list(data_by_language['Desc_Tokens']),
    #                                  'ID': list(data_by_language['ID']),
    #                                  'Name': list(data_by_language['Name']),
    #                                  'RepoUrl': list(data_by_language['RepoUrl']),
    #                                  'Lang': list(data_by_language['Lang']),
    #                                  # 'Desc_Vectors': list(data_by_language['Desc_Vectors']),
    #                                  'Average_Pooling': list(data_by_language['Average_Pooling'])
    #                                  })
    # word2vec_pickle_path = path + 'desc_word2vec_' + language + '.bin'
    # model = gensim.models.KeyedVectors.load(word2vec_pickle_path)
    # msg_tokens = clean_up_sentence(pd.Series(msg))
    # reply, score = matchRepoToInput(data_by_language, model, msg_tokens)
    return 'reply'


# Creating GUI with tkinter
def send(entry_box, chat_log):
    msg = entry_box.get("1.0", 'end-1c').strip()
    entry_box.delete("0.0", END)

    if msg != '':
        chat_log.config(state=NORMAL)
        chat_log.insert(END, "You: " + msg + '\n\n')
        chat_log.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        chat_log.insert(END, "Bot: " + res + '\n\n')

        chat_log.config(state=DISABLED)
        chat_log.yview(END)


def check_if_running(thread, window):
    if thread.is_alive():
        window.after(1000, check_if_running, thread, window)
    else:
        window.destroy()


def loading_ui():
    # Loading UI
    loading = Tk()
    loading.title("Software Lookup Bot")
    loading.geometry("400x500")
    loading.resizable(width=FALSE, height=FALSE)

    # loading label
    label = Label(loading, text='Loading...')
    label.config(font=('Courier', 30))
    label.pack(side=TOP, ipady=100)

    # run window
    loading.mainloop()


def main_ui():
    print('loading ui')
    # UI Main
    base = Tk()
    base.title("Software Lookup Bot")
    # base.destroy()
    base.geometry("400x500")
    base.resizable(width=FALSE, height=FALSE)

    # Create Chat window
    chat_log = Text(base, bd=0, bg="white", height="8", width="50", font="Arial", )

    chat_log.config(state=DISABLED)

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=chat_log.yview, cursor="heart")
    chat_log['yscrollcommand'] = scrollbar.set

    # Create Button to send message
    send_button = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                         bd=0, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                         command=send)

    # Create the box to enter message
    entry_box = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
    # EntryBox.bind("<Return>", send)

    # Place all components on the screen
    scrollbar.place(x=376, y=6, height=386)
    chat_log.place(x=6, y=6, height=386, width=370)
    entry_box.place(x=128, y=401, height=90, width=265)
    send_button.place(x=6, y=401, height=90)

    base.mainloop()


load_trained_data()
main_ui()