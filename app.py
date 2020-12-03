import tkinter as tk
import pickle
import numpy as np
import os
import io
from PIL import Image
from image_preprocessing import img_to_mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
WIDTH = 400
HEIGHT = 350
with open('ANN/model.h5', 'rb') as file:
    MODEL = pickle.load(file)  # deserialization of the Neural Network model


class App:
    ''' The main application class, with GUI built with Tkinter module. '''

    def __init__(self):
        self.main()

    def main(self):
        root = tk.Tk()
        root.title('Handwritten digits recognition')
        root.geometry(f'{WIDTH}x{HEIGHT}')
        root.resizable(False, False)

        global canvas
        canvas = tk.Canvas(root, width=WIDTH, height=0.7 * HEIGHT)
        canvas.grid(row=0, column=0, columnspan=2)
        canvas.old_coords = None

        clear_button = tk.Button(root, text='CLEAR', font=('Arial', 14), command=lambda x=canvas: x.delete('all'))
        clear_button.grid(row=1, column=0, pady=30)

        submit_button = tk.Button(root, text='CHECK', font=('Arial', 14), command=self.run)
        submit_button.grid(row=1, column=1, pady=30)

        root.bind('<B1-Motion>', self.draw)
        root.bind('<ButtonRelease-1>', self.reset_coords)

        root.mainloop()

    def draw(self, event):
        """
        Draws on canvas.
        """
        x, y = event.x, event.y
        if canvas.old_coords:
            x1, y1 = canvas.old_coords
            canvas.create_oval(x, y, x1, y1, width=12)
        canvas.old_coords = x, y

    def reset_coords(self, event):
        """
        Resets values of coordinates after every move on canvas in order for line to be fully continuous.
        """
        canvas.old_coords = None

    def save_drawing(self):
        """
        Takes screenshot of the canvas and saves the image.
        """
        ps = canvas.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))  # canvas screenshot

        img.save('IMG.jpg', 'jpeg')

    def predict_number(self):
        """
        Uses model to predict the number drawn by user.
        Then, prints out the probabilities and predicted number in console.
        """
        image = np.array(img_to_mnist('IMG.jpg')).reshape((1, 28, 28, 1))
        predicted_values = MODEL.predict(image)  # probability distribution for digits from 0-9

        for number in range(10):
            probability = predicted_values[0][number]
            print('Probability for {}:\t{:.5f}'.format(number, probability))
        print(f'Predicted: {np.argmax(predicted_values)}')
        os.remove('IMG.jpg')  # deletes the drawing after prediction

    def run(self):
        """
        Function executed with clicking the 'CHECK' button. Program saves drawing and uses it to predict and print the output.
        """
        self.save_drawing()
        self.predict_number()


App()
