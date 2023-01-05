import cv2, os, glob
from tkinter import filedialog, Tk
import numpy as np
import PlantSzClr

root = Tk()
root.withdraw()
RawImagePath = filedialog.askdirectory(title='Select The folder with JPG files that need background removal')
os.chdir(RawImagePath)
ImList = glob.glob('*.JPG')

cnt = 0
for imageAn in ImList:  
    cnt = cnt+1
    print("Processing File: {}/{}".format(cnt,len(ImList)))
    PlantSzClr.bgremove(imageAn,150)
#This function removes a white background, leaving only objects
#It may be possible to alter it by changing the lower/upper thresholds
#to remove coloured backgrounds but this hasn't been trialled.
#Lowthresh is used to determine what is background, change this value to change
#how much background is removed. Higher = stricter (more bg removed)