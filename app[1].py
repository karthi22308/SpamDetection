import streamlit as st
import pickle
import string
import csv
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def lt(s):
 strl=""
 for ele in s:
  strl+=ele
 return strl

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
opt=0
options=[]
st.title("Malicious E-mail detector")
mde = st.radio(
     "Spam customizer ",
     ('disable', 'enable'))
if mde =="enable":
 options = st.multiselect(
     'select categories',
     ['sports', 'offers', 'e-shopping', 'job alert'],
     ['sports'])
 opt=1
sports= ['basketball' , 'football' , 'basketball' , 'tennis' , 'basketball' , 'tennis' , 'cricket' ]
offers= ['discount' , 'sale' , 'hurry' ,  'limited' ]
shopping=['flipkart' , 'amazon' , 'ebay' ,  'meesho' ]
jobalert= ['vacancy' , 'job' , 'tcs' , 'wipro' , 'government' , 'railway' , 'public' ]
cn= st.radio("Select Mode",('INBOX','TEST'))
if cn =="TEST":
 mode = st.radio(
      "Select Mode",
      ('Single Message', 'Multiple Messages'))
 if mode =="Single Message":
  input_sms = st.text_area("Enter the mail recieved")

  if st.button('Predict'):
      transformed_sms = transform_text(input_sms)
      vector_input = tfidf.transform([transformed_sms])
      result = model.predict(vector_input)[0]
      if result == 1:
         new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;">Warning!</p>'
         st.markdown(new_title, unsafe_allow_html=True)
         st.header("Malicious email")
         test=0
         if 'sports' in options:
          for sport in sports:
           if sport in input_sms:
            test+=1
         if 'offers' in options:
           for offer in offers:
            if offer in input_sms:
             test+=1
         if 'e-shopping' in options:
          for eshop in shopping:
           if eshop in input_sms:
            test+=1 
         if 'job alert' in options:
          for job in jobalert:
           if job in input_sms:
            test+=1
         if test>0:
          new_le = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Related to your Interests</p>'
          st.markdown(new_le, unsafe_allow_html=True)
      else:
         new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Safe E-mail</p>'
         st.markdown(new_title, unsafe_allow_html=True)
 elif opt==0:
  sentences = []
  t = 0
  paragraph = st.text_area("Enter the mails received separated by a double space")
  a = []
  if st.button('Predict'):
   while paragraph.find('  ') != -1:
      index = paragraph.find('  ')
      sentences.append(paragraph[:index + 1])
      t += 1
      paragraph = paragraph[index + 1:]
   p=0
   while (p<t):
    transformed_sms = transform_text(sentences[p])
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
     a.append("Malicious Email")
    else:
     a.append("Safe Email")
    p+=1
  with open('out.csv', 'w', newline='') as csvfile:
   fieldnames = ['message','result']
   thewriter= csv.DictWriter(csvfile, fieldnames=fieldnames)
   thewriter.writeheader()
   b=0
   for sentence in sentences:
    thewriter.writerow({'message':sentence, 'result':a[b]})
    b+=1
  def data():
   df=pd.read_csv("out.csv")
   return df
  df=data()
  st.dataframe(data=df)
 else:
  c=[]
  sentences = []
  t = 0
  paragraph = st.text_area("Enter the mails received separated by a double space")
  a = []
  if st.button('Predict'):
   while paragraph.find('  ') != -1:
      index = paragraph.find('  ')
      sentences.append(paragraph[:index + 1])
      t += 1
      paragraph = paragraph[index + 1:]
   p=0
   while (p<t):
    transformed_sms = transform_text(sentences[p])
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
     a.append("Malicious Email")
     test=0
     if 'sports' in options:
      for sport in sports:
       if sport in sentences[p]:
        test+=1
     if 'offers' in options:
      for offer in offers:
       if offer in sentences[p]:
        test+=1
     if 'e-shopping' in options:
      for eshop in shopping:
       if eshop in sentences[p]:
        test+=1 
     if 'job alert' in options:
      for job in jobalert:
       if job in sentences[p]:
        test+=1
     if test>0:
      c.append("Related to your Interests")
     else:
      c.append("Not Related to you")
    else:
     a.append("Safe Email")
     c.append("'N/A'")
    p+=1
  with open('out.csv', 'w', newline='') as csvfile:
   fieldnames = ['message','result','Interested']
   thewriter= csv.DictWriter(csvfile, fieldnames=fieldnames)
   thewriter.writeheader()
   b=0
   for sentence in sentences:
    thewriter.writerow({'message':sentence, 'result':a[b],'Interested':c[b]})
    b+=1
  def data():
   df=pd.read_csv("out.csv")
   return df
  df=data()
  st.dataframe(data=df)
elif cn=="INBOX":
 st.title("INBOX")
 t=0
 file = open("in.csv")
 csvreader = csv.reader(file)
 header = next(csvreader)
 print(header)
 rows = []
 for row in csvreader:
     rows.append(row)
     t+=1
 file.close()
 def inbox():
   df=pd.read_csv("in.csv")
   return df
 df=inbox()
 st.dataframe(data=df)
 if st.button('DETECT'):
  if opt==0:
   a=[]
   p=0
   while (p<t):
    r=lt(rows[p])
    transformed_sms = transform_text(r)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
     a.append("Malicious Email")
    else:
     a.append("Safe Email")
    p+=1
   with open('out.csv', 'w', newline='') as csvfile:
    fieldnames = ['message','result']
    thewriter= csv.DictWriter(csvfile, fieldnames=fieldnames)
    thewriter.writeheader()
    b=0
    for row in rows:
     thewriter.writerow({'message':lt(row), 'result':a[b]})
     b+=1
   def data():
    df=pd.read_csv("out.csv")
    return df
   df=data()
   st.dataframe(data=df)
  else:
    p=0
    c=[]
    a=[]
    while (p<t):
     r=lt(rows[p])
     transformed_sms = transform_text(r)
     vector_input = tfidf.transform([transformed_sms])
     result = model.predict(vector_input)[0]
     if result == 1:
      a.append("Malicious Email")
      test=0
      if 'sports' in options:
       for sport in sports:
        if sport in r:
         test+=1
      if 'offers' in options:
       for offer in offers:
        if offer in r:
         test+=1
      if 'e-shopping' in options:
       for eshop in shopping:
        if eshop in r:
         test+=1 
      if 'job alert' in options:
       for job in jobalert:
        if job in r:
         test+=1
      if test>0:
       c.append("Related to your Interests")
      else:
       c.append("Not Related to you")
     else:
      a.append("Safe Email")
      c.append("'N/A'")
     p+=1
    with open('out.csv', 'w', newline='') as csvfile:
     fieldnames = ['message','result','Interested']
     thewriter= csv.DictWriter(csvfile, fieldnames=fieldnames)
     thewriter.writeheader()
     b=0
     for row in rows:
      thewriter.writerow({'message':lt(row), 'result':a[b],'Interested':c[b]})
      b+=1
    def data():
     df=pd.read_csv("out.csv")
     return df
    df=data()
    st.dataframe(data=df)
  
 
  
 
 
