#Import tkinter library.
from tkinter import *
from PIL import Image
from PIL import ImageTk
from Image_Processing import Application
def click_Button():
    window.destroy()
    #Go-To-Application.
    ui = Application()
    ui.mainloop()
####################################################
window = Tk()
window.title('DIGITAL IMAGE PROCESSING APPLICATION')
window.eval('tk::PlaceWindow . center')
width = 1250  # Width.
height = 650  # Height.
screen_width = window.winfo_screenwidth()  # Width of the screen.
screen_height = window.winfo_screenheight()  # Height of the screen.
# Calculate Starting X and Y coordinates for Window.
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)
window.geometry('%dx%d+%d+%d' % (width, height, x, y))
# the Maximum size-Don't-Change.
# tkinter fixed window size.
window.maxsize(width=width,height=height)
window.config(bg='#013A63',highlightthickness=0)
print(width)
print(height)
#Create Image Background.
#image = Image.open('BG-scaled.jpg')
image = Image.open('SanFranciscoAhmed.jpg')
image = image.resize((1350, 800))
image = ImageTk.PhotoImage(image)
canvas = Canvas(width=width,height=height,highlightthickness=0)
#images = PhotoImage(file = image)
canvas.create_image(width/2,height/2,image=image)
canvas.pack()
#Label.
l1 = Label(canvas, text='DIGITAL IMAGE PROCESSING APPLICATION',
           font=('Arial',44, 'bold'),
           pady=25, justify=CENTER,bg='#ff4d4d',fg='#3333ff',highlightthickness=0)
l1.place(relx=0.5, rely=0.35, anchor='center')
#Button.
b1 = Button(canvas, text='CONTINUE', bg='#ff4d4d', fg='#3333ff',
              font=('Arial',30,'bold'),activebackground='#012A4A',
            highlightthickness=0,command=click_Button)
b1.place(relx=0.5, rely=0.75, anchor='center')
window.mainloop()
##############################################
##############################################

