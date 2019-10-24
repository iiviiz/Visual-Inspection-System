##----------------------------------------------------
## Created by Ivan Luiz de Oliveira
## Unesp Sorocaba Aug. 2019
##----------------------------------------------------

import tkinter as tk
import webbrowser
import os
top = tk.Tk()

def callback_operacao():
    top.quit()
    exec(open("Operation.py").read())

def callback_aquisicao():
    top.destroy()
    exec(open("Data_Aquisition.py").read())


def callback_treinamento():
    exec(open("Training_NN.py").read())

def callback_configuracao():
    try:
        os.system('TASKKILL /F /IM notepad.exe')
    except:
        pass
    try:
        webbrowser.open('config.cfg')
    except:
        print("Não foi possível abrir o arquivo de configurações!")
top.geometry("345x380+400+200")
top.title("Menu")
top.configure(background="#d9d9d9")
 
Frame1 = tk.Frame(top)
Frame1.place(relx=0.029, rely=0.026, relheight=0.645, relwidth=0.942)
Frame1.configure(relief='groove')
Frame1.configure(borderwidth="2")
Frame1.configure(relief="groove")
Frame1.configure(background="#b5dcdd")
Frame1.configure(width=295)

titulo = tk.Label(Frame1)
titulo.place(relx=0.015, rely=0.041, height=33, width=309)
titulo.configure(background="#b5dcdd")
titulo.configure(font="-family {Segoe UI} -size 14 -weight bold -slant italic")
titulo.configure(foreground="#000000")
titulo.configure(text='''Sistema de Inspeção Visual''')


ButtonMO = tk.Button(Frame1, command = callback_operacao)
ButtonMO.place(relx=0.185, rely=0.245, height=44, width=197)
ButtonMO.configure(background="#a5c8c9")
ButtonMO.configure(font="-family {Segoe UI} -size 14 -slant italic")
ButtonMO.configure(foreground="#000000")
ButtonMO.configure(text='''Modo Operação''')


ButtonMA = tk.Button(Frame1, command = callback_treinamento)
ButtonMA.place(relx=0.185, rely=0.735, height=44, width=197)
ButtonMA.configure(background="#a5c8c9")
ButtonMA.configure(font="-family {Segoe UI} -size 14 -slant italic")
ButtonMA.configure(foreground="#000000")
ButtonMA.configure(text='''Modo Treinamento''')


ButtonMT = tk.Button(Frame1, command = callback_aquisicao)
ButtonMT.place(relx=0.185, rely=0.49, height=44, width=197)
ButtonMT.configure(background="#a5c8c9")
ButtonMT.configure(font="-family {Segoe UI} -size 14 -slant italic")
ButtonMT.configure(foreground="#000000")
ButtonMT.configure(text='''Modo Aquisição''')

Frame2 = tk.Frame(top)
Frame2.place(relx=0.029, rely=0.658, relheight=0.224, relwidth=0.942)
Frame2.configure(relief='groove')
Frame2.configure(borderwidth="2")
Frame2.configure(background="#b5dcdd")
Frame2.configure(width=295)

ButtonCFG = tk.Button(Frame2, command = callback_aquisicao)
ButtonCFG.place(relx=0.185, rely=0.235, height=44, width=197)
ButtonCFG.configure(activebackground="#ececec")
ButtonCFG.configure(activeforeground="#000000")
ButtonCFG.configure(background="#a5c8c9")
ButtonCFG.configure(disabledforeground="#a3a3a3")
ButtonCFG.configure(font="-family {Segoe UI} -size 14 -slant italic")
ButtonCFG.configure(foreground="#000000")
ButtonCFG.configure(highlightbackground="#d9d9d9")
ButtonCFG.configure(highlightcolor="black")
ButtonCFG.configure(pady="0")
ButtonCFG.configure(text='''Configurações''')
ButtonCFG.configure(command = callback_configuracao)

Frame3 = tk.Frame(top)
Frame3.place(relx=0.029, rely=0.868, relheight=0.118, relwidth=0.942)
Frame3.configure(relief='groove')
Frame3.configure(borderwidth="2")
Frame3.configure(relief="groove")
Frame3.configure(background="#b5dcdd")
Frame3.configure(highlightbackground="#d9d9d9")
Frame3.configure(highlightcolor="black")
Frame3.configure(width=295)

autor = tk.Label(Frame3)
autor.place(relx=0.031, rely=0.222, height=23, width=296)
autor.configure(background="#b5dcdd")
autor.configure(font="-family {Segoe UI} -size 9")
autor.configure(foreground="#000000")
autor.configure(text='''Ivan Luiz de Oliveira - UNESP Sorocaba - 2019''')

top.mainloop()