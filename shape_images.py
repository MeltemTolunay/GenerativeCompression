import numpy as np
from PIL import Image
import os

new_width = 128
new_height = 128

directory = os.fsencode('/Users/Meltem/Desktop/celeb10')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith('.jpg'):
        im = Image.open(filename)
        width, height = im.size

        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        new = im.crop((left, top, right, bottom))
        new.save('./output/' + filename)
        print(new.size)

        
