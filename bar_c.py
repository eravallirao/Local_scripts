from PIL import Image
from pyzbar.pyzbar import decode
data = decode(Image.open('b1.png'))
print(data)