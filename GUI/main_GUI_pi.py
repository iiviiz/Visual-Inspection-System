#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# GUI module generated by PAGE version 4.22
#  in conjunction with Tcl version 8.6
#    May 06, 2019 05:07:58 PM -0300  platform: Windows NT

import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import main_GUI_pi_support

def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = tk.Tk()
    top = Main (root)
    main_GUI_pi_support.init(root, top)
    root.mainloop()

w = None
def create_Main(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = tk.Toplevel (root)
    top = Main (w)
    main_GUI_pi_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_Main():
    global w
    w.destroy()
    w = None

class Main:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85'
        _ana2color = '#ececec' # Closest X11 color: 'gray92'

        top.geometry("345x380+400+200")
        top.title("Sistema de Inspe��o Visual")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.Frame1 = tk.Frame(top)
        self.Frame1.place(relx=0.029, rely=0.026, relheight=0.645
                , relwidth=0.942)
        self.Frame1.configure(relief='groove')
        self.Frame1.configure(borderwidth="2")
        self.Frame1.configure(relief="groove")
        self.Frame1.configure(background="#b5dcdd")
        self.Frame1.configure(highlightbackground="#d9d9d9")
        self.Frame1.configure(highlightcolor="black")
        self.Frame1.configure(width=325)

        self.ButtonMO = tk.Button(self.Frame1)
        self.ButtonMO.place(relx=0.185, rely=0.245, height=44, width=197)
        self.ButtonMO.configure(activebackground="#f9f9f9")
        self.ButtonMO.configure(activeforeground="black")
        self.ButtonMO.configure(background="#a5c8c9")
        self.ButtonMO.configure(command=lambda:main_GUI_pi_support.exec(open("BAQ.py").read()))
        self.ButtonMO.configure(disabledforeground="#a3a3a3")
        self.ButtonMO.configure(font="-family {Segoe UI} -size 15 -slant italic")
        self.ButtonMO.configure(foreground="#000000")
        self.ButtonMO.configure(highlightbackground="#d9d9d9")
        self.ButtonMO.configure(highlightcolor="black")
        self.ButtonMO.configure(pady="0")
        self.ButtonMO.configure(text='''Modo Opera��o''')

        self.ButtonMA = tk.Button(self.Frame1)
        self.ButtonMA.place(relx=0.185, rely=0.735, height=44, width=197)
        self.ButtonMA.configure(activebackground="#f9f9f9")
        self.ButtonMA.configure(activeforeground="black")
        self.ButtonMA.configure(background="#a5c8c9")
        self.ButtonMA.configure(disabledforeground="#a3a3a3")
        self.ButtonMA.configure(font="-family {Segoe UI} -size 15 -slant italic")
        self.ButtonMA.configure(foreground="#000000")
        self.ButtonMA.configure(highlightbackground="#d9d9d9")
        self.ButtonMA.configure(highlightcolor="black")
        self.ButtonMA.configure(pady="0")
        self.ButtonMA.configure(text='''Modo Treinamento''')

        self.ButtonMT = tk.Button(self.Frame1)
        self.ButtonMT.place(relx=0.185, rely=0.49, height=44, width=197)
        self.ButtonMT.configure(activebackground="#f9f9f9")
        self.ButtonMT.configure(activeforeground="black")
        self.ButtonMT.configure(background="#a5c8c9")
        self.ButtonMT.configure(disabledforeground="#a3a3a3")
        self.ButtonMT.configure(font="-family {Segoe UI} -size 15 -slant italic")
        self.ButtonMT.configure(foreground="#000000")
        self.ButtonMT.configure(highlightbackground="#d9d9d9")
        self.ButtonMT.configure(highlightcolor="black")
        self.ButtonMT.configure(pady="0")
        self.ButtonMT.configure(text='''Modo Aquisi��o''')

        self.Label2 = tk.Label(self.Frame1)
        self.Label2.place(relx=0.015, rely=0.041, height=33, width=309)
        self.Label2.configure(activebackground="#f9f9f9")
        self.Label2.configure(activeforeground="black")
        self.Label2.configure(background="#b5dcdd")
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(font="-family {Segoe UI} -size 14 -weight bold -slant italic")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(highlightbackground="#d9d9d9")
        self.Label2.configure(highlightcolor="black")
        self.Label2.configure(text='''Sistema de Inspe��o Visual''')
        self.Label2.configure(width=309)

        self.Frame4 = tk.Frame(top)
        self.Frame4.place(relx=0.029, rely=0.868, relheight=0.118
                , relwidth=0.942)
        self.Frame4.configure(relief='groove')
        self.Frame4.configure(borderwidth="2")
        self.Frame4.configure(relief="groove")
        self.Frame4.configure(background="#b5dcdd")
        self.Frame4.configure(highlightbackground="#d9d9d9")
        self.Frame4.configure(highlightcolor="black")
        self.Frame4.configure(width=325)

        self.fra44_lab46 = tk.Label(self.Frame4)
        self.fra44_lab46.place(relx=0.031, rely=0.222, height=23, width=296)
        self.fra44_lab46.configure(activebackground="#f9f9f9")
        self.fra44_lab46.configure(activeforeground="black")
        self.fra44_lab46.configure(background="#b5dcdd")
        self.fra44_lab46.configure(disabledforeground="#a3a3a3")
        self.fra44_lab46.configure(font="-family {Segoe UI} -size 9")
        self.fra44_lab46.configure(foreground="#000000")
        self.fra44_lab46.configure(highlightbackground="#d9d9d9")
        self.fra44_lab46.configure(highlightcolor="black")
        self.fra44_lab46.configure(text='''Ivan Luiz de Oliveira - UNESP Sorocaba - 2019''')
        self.fra44_lab46.configure(width=296)

        self.Frame3 = tk.Frame(top)
        self.Frame3.place(relx=0.029, rely=0.658, relheight=0.224
                , relwidth=0.942)
        self.Frame3.configure(relief='groove')
        self.Frame3.configure(borderwidth="2")
        self.Frame3.configure(relief="groove")
        self.Frame3.configure(background="#b5dcdd")
        self.Frame3.configure(highlightbackground="#d9d9d9")
        self.Frame3.configure(highlightcolor="black")
        self.Frame3.configure(width=325)

        self.ButtonCFG = tk.Button(self.Frame3)
        self.ButtonCFG.place(relx=0.185, rely=0.235, height=44, width=197)
        self.ButtonCFG.configure(activebackground="#f9f9f9")
        self.ButtonCFG.configure(activeforeground="black")
        self.ButtonCFG.configure(background="#a5c8c9")
        self.ButtonCFG.configure(disabledforeground="#a3a3a3")
        self.ButtonCFG.configure(font="-family {Segoe UI} -size 15 -slant italic")
        self.ButtonCFG.configure(foreground="#000000")
        self.ButtonCFG.configure(highlightbackground="#d9d9d9")
        self.ButtonCFG.configure(highlightcolor="black")
        self.ButtonCFG.configure(pady="0")
        self.ButtonCFG.configure(text='''Configura��es''')

if __name__ == '__main__':
    vp_start_gui()





