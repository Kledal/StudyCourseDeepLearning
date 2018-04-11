from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import numpy as np


font = ImageFont.truetype("monofont.ttf", 64)

def generateImage(msg):
  W, H = (64,64)
  img = Image.new("L", (W, H))
  draw = ImageDraw.Draw(img)
  w, h = draw.textsize(msg, font=font)
  draw.text(((W-w)/2,(H-h)/2-4), msg, fill=255, font=font)
  return img
