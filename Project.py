from tkinter import *
import re
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("indian_spam.csv", encoding= 'latin')

data2 = pd.read_csv("spam.csv", encoding='latin')
data2.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1, inplace=True)
data = pd.concat([data,data2], ignore_index= True)

data.columns=['result', 'text']
data['result'] = data['result'].map({'ham': 0, 'spam': 1})

vec = TfidfVectorizer()
x= vec.fit_transform(data['text'])
y= data['result']


model = SVC(kernel = 'linear', C=1)
model.fit(x,y)
samplex = ''
sampley = 0
window = Tk()

window.title('Spam Detection Project')
window.geometry("600x600")
window.config(background='#6200EE', pady=10)

# Creating a heading
head = Label(window, text='Spam Detection Machine', font = ('Times New Roman', 30, 'bold'))
head.pack()

spacer = Label(window, pady=10, background='#6200EE')
spacer.pack()

# Creating a description
des = Label(window, text = 'Welcome!!\nTo use this Machine, enter content of your message in textbox below and press Submit Button', font = ('Arial', 10))
des.pack()

spacer = Label(window, pady=5, background='#6200EE')
spacer.pack()


txt_var = StringVar()
# Creating a textbox
txt = Entry(window, width=80, bd=5, textvariable=txt_var, font = ('Arial', 10))
txt.pack()

spacer = Label(window, pady=1, background='#6200EE')
spacer.pack()

def dlt_btn():
     txt.delete(0,END)
#creating a delete button
delete = Button(window, text = 'Delete', font = ('Arial', 10, 'bold'), command=dlt_btn)
delete.pack()
# creating a spacer between elements
spacer = Label(window, pady=1, background='#6200EE')
spacer.pack()
def result():
        samplex = txt_var.get().strip()
        vec_samplex = vec.transform([samplex])
        sampley = model.predict(vec_samplex)
        if(samplex==''):
            messagebox.showerror('Result','Please Enter Text!!')
        else:
            # if():
            #      messagebox.showinfo('Result','Please Enter a Valid Text')
            if(sampley==0):
                messagebox.showinfo('Result','Not a SPAM!!')
            else:
                messagebox.showinfo('Result','SPAM!!')


#creating a submit button
submit = Button(window, text= 'Submit', command= result, font = ('Arial', 10, 'bold'))
submit.pack()

spacer = Label(window, pady=10, background='#6200EE')
spacer.pack()

rep = PhotoImage(file='Report.png')
report = Label(window, text= 'Performance Report of Machine\n Model = SVM', font = ('Arial', 10,'bold'), image=rep, compound='bottom', anchor='center')
report.pack()
window.mainloop()