import numpy as np
import tkinter as tk
import pickle

root = tk.Tk()
canvas1 = tk.Canvas(root, width=800, height=330)
canvas1.pack()

label1 = tk.Label(root, text='Prediction of E. Coli in the Englishman River')
label1.config(font=('Arial', 20))
canvas1.create_window(400, 50, window=label1)

label2 = tk.Label(root, text='Hardness [MG/L]:')
label2.config(font=('helvetica', 10))
canvas1.create_window(250, 100, window=label2)
entry1 = tk.Entry(root)
canvas1.create_window(400, 100, window=entry1)

label3 = tk.Label(root, text='Conductivity [USIE/CM]:')
label3.config(font=('helvetica', 10))
canvas1.create_window(250, 120, window=label3)
entry2 = tk.Entry(root)
canvas1.create_window(400, 120, window=entry2)

label4 = tk.Label(root, text='Water temp [DEG C]:')
label4.config(font=('helvetica', 10))
canvas1.create_window(250, 140, window=label4)
entry3 = tk.Entry(root)
canvas1.create_window(400, 140, window=entry3)

label5 = tk.Label(root, text='Turbidity [NTU]:')
label5.config(font=('helvetica', 10))
canvas1.create_window(250, 160, window=label5)
entry4 = tk.Entry(root)
canvas1.create_window(400, 160, window=entry4)

label6 = tk.Label(root, text='Air temp [DEG C]:')
label6.config(font=('helvetica', 10))
canvas1.create_window(250, 180, window=label6)
entry5 = tk.Entry(root)
canvas1.create_window(400, 180, window=entry5)

label7 = tk.Label(root, text='Precip today [mm]:')
label7.config(font=('helvetica', 10))
canvas1.create_window(250, 200, window=label7)
entry6 = tk.Entry(root)
canvas1.create_window(400, 200, window=entry6)

label8 = tk.Label(root, text='Precip last 3 days [mm]:')
label8.config(font=('helvetica', 10))
canvas1.create_window(250, 220, window=label8)
entry7 = tk.Entry(root)
canvas1.create_window(400, 220, window=entry7)

def prediction():
    global x1
    global x2
    global x3
    global x4
    global x5
    global x6
    global x7
    global bar1
    global pie2
    x1 = float(entry1.get())
    x2 = float(entry2.get())
    x3 = float(entry3.get())
    x4 = float(entry4.get())
    x5 = float(entry5.get())
    x6 = float(entry6.get())
    x7 = float(entry7.get())

    x_input = np.array([[x1, x2, x3, x4, x5, x6, x7]])
    with open(r'finalized_model.sav', 'rb') as f:
        loaded_model = pickle.load(f)
    clf = loaded_model['Model']
    stdscX = loaded_model['Scaler']
    x_input = stdscX.transform(x_input)
    result = clf.predict(x_input)

    if result == 1:
        label9 = tk.Label(root, text= 'There is more than 20 CFU/100ML of E. Coli with a probability of 0.69')
    else:
        label9 = tk.Label(root, text= 'There is less than 20 CFU/100ML of E. Coli with a probability of 0.69')

    label9.config(font=('helvetica', 10))
    canvas1.create_window(400, 300, window=label9)

button1 = tk.Button (root, text=' Predict',command=prediction, bg='palegreen2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 260, window=button1)

root.mainloop()