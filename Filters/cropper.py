from PIL import Image

#Load the tif image and convert it to grayscale
img = Image.open('tif_test.tif').convert('L')


left, top, right, bottom = 38, 19, 548, 529  #510x510 for the tif files, missing pixels >:(
cropped = img.crop( ( left, top, right, bottom ) )

cropped.save('cropped_greyscale.png')