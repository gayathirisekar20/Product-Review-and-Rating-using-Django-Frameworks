import pickle
# import train_model
import os
import string
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
clf = pickle.load(open('saved_model.pickle', 'rb'))




punctuation = string.punctuation
stop_words = list(STOP_WORDS)
nlp = spacy.load("en_core_web_sm")
numbers = string.digits
    
    
def data_preparation(comment):
    # cleaned_comment_temp = train_model.cleaning_function(comment)
    text = nlp(comment)
    tokens = []
    for token in text:
        temp = token.lemma_.lower()
        tokens.append(temp)
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and token not in punctuation and token not in numbers:
            cleaned_tokens.append(token)
    cleaned_joint_comment_temp = ' '.join(cleaned_tokens)
    comment_arr=[]
    comment_arr.append(cleaned_joint_comment_temp)
				
comment='product is notbad'
data_preparation(comment)
print(comment_arr)
# pred_arr = []
# pos = 0
# neg = 0
for comment in comment_arr:
        comment_pred = clf.predict([comment])
        if comment_pred[0]==1:
            print('pos')
        else:
            print('neg')
        # pred_arr.append(comment_pred)


# sample_texts = ["congrats! 1 year special cinema pass for 2 is yours. call 09061209465 now! c suprman v, matrix3, starwars3, etc all 4 free! bx420-ip4-5we. 150pm. dont miss out! "]
# sample_texts = [data_preparation(sentence) for sentence in sample_texts]

# txts = tokenizer.texts_to_sequences(sample_texts)
# txts = pad_sequences(txts, maxlen=max_len)
# preds = model.predict(txts, verbose=0)
# print(np.around(preds))