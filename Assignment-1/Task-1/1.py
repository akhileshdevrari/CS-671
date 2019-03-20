######################################################################################################################
# Purpose of this file:
# Make a dataset of (28 × 28 × 3) images of straight lines on a black background with the
# following variations:
# 1. Length - 7 (short) and 15 (long) pixels.
# 2. Width - 1 (thin) and 3 (thick) pixels.
# 3. Angle with X-axis - Angle θ ∈ [0 o , 180 o ) at intervals of 15 o .
# 4. Color - Red and Blue.

# Values like length of line and size of images has been hard coded due to simplicity of code
# Author: Akhilesh Devrari <devrari.akhilesh@gmail.com>
########################################################################################################################



from PIL import Image, ImageDraw
import random
import numpy as np
import os
from os.path import isfile, join
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc


########################################################
# Creating images
########################################################

folderPNG = "./imagesPNG"
folderJPG = "./imagesJPG"
ID = 0
# Create folders if they don't already exist
if not os.path.exists(folderPNG):
    os.makedirs(folderPNG)
if not os.path.exists(folderJPG):
    os.makedirs(folderJPG)

# Iterating over all attributes of a class
for length in range(0,2):
	for width in range(0,2):
		for color in range(0,2):
			for imgID in range(0,1000):
				# np-array representing data for the image
				data = np.zeros((28, 28, 3), dtype=np.uint8)
				# l = length, w = width of the image to be generated
				l = 7 + 8*length
				w = 1 + 2*width
				# (row,col) if the upper-left corner of the image to be generated
				row = random.randrange(int(5+l/2), int(28-5-l/2-1), 1)
				col = random.randrange(1+5, int(28-l-5), 1)

				# Fill color RGB values in image-np-array
				for i in range(row-1, row+w-1):
					for j in range(col-1, col+l-1):
						data[i][j][2*color] = 255

				# Create image from array
				img = Image.fromarray(data, 'RGB')
				# Upto now we have only created an horizontal line with three class attributes. Now rotate the image to get fourth attribute of class
				for angle in range(0,12):
					tempImg = img.rotate(15*angle)
					# folders where this paricular image belongs
					classFolderPNG = folderPNG+"/"+str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)
					classFolderJPG = folderJPG+"/"+str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)
					# Create folders of corresponding class if they don't already exist
					if not os.path.exists(classFolderPNG):
						os.makedirs(classFolderPNG)
					if not os.path.exists(classFolderJPG):
						os.makedirs(classFolderJPG)
					# Finally save the images in corresponding folders
					tempImg.save(classFolderPNG+"/"+str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)+"_"+str(ID)+".png")
					tempImg.save(classFolderJPG+"/"+str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)+"_"+str(ID)+".jpg")
					# Image ID acts like a primary key: unique identifier of an image
					ID += 1



###########################################
#Creating video
###########################################
# params: folder = name of folder where images of all classes reside
# 			outputVideoName = name of video to be produced
def makeVideo(folder, outputVideoName):
	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	outputVideo = cv2.VideoWriter(outputVideoName, fourcc, 2, (84, 84))

	# Iterating over all classes in given folder
	for length in range(0,2):
		for width in range(0,2):
			for angle in range(0,12):
				for color in range(0,2):
					# Directory containing images of this particular class
					dir_path = folder+"/"+str(length)+"_"+str(width)+"_"+str(angle)+"_"+str(color)
					# Get 90 images for dir_path
					images = []
					num = 1
					for img in os.listdir(dir_path):
						if num > 90:
							break
						num += 1
						images.append(img)
					# Creating frames containg 9 images each
					for i in range(0,10):
						# Pick 9 images in oreder and create a frame
						frame = Image.new('RGB', (84,84))
						frame.paste(Image.open(dir_path+"/"+images[i*9]), (0,0,28,28))
						frame.paste(Image.open(dir_path+"/"+images[i*9+1]), (28,0,56,28))
						frame.paste(Image.open(dir_path+"/"+images[i*9+2]), (56,0,84,28))
						frame.paste(Image.open(dir_path+"/"+images[i*9+3]), (0,28,28,56))
						frame.paste(Image.open(dir_path+"/"+images[i*9+4]), (28,28,56,56))
						frame.paste(Image.open(dir_path+"/"+images[i*9+5]), (56,28,84,56))
						frame.paste(Image.open(dir_path+"/"+images[i*9+6]), (0,56,28,84))
						frame.paste(Image.open(dir_path+"/"+images[i*9+7]), (28,56,56,84))
						frame.paste(Image.open(dir_path+"/"+images[i*9+8]), (56,56,84,84))

						# CV2 assumes BGR by default, but our images are in RGB
						frame = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_BGR2RGB)
						# Write the frame in output video
						outputVideo.write(frame)
	# Finally write the images
	outputVideo.release()


# Calling makeVideo() to make videos from PNG and JPG images seperately
makeVideo(folderPNG, "PNGImagesVideo.mp4")
makeVideo(folderJPG, "JPGImagesVideo.mp4")