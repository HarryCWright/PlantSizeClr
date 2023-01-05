import cv2, os, glob
from tkinter import filedialog, Tk
from tkinter.filedialog import askopenfilename
import PlantSzClr

  
root = Tk()
root.withdraw()
BackCorr = askopenfilename(title='Please select the JPG file for background correction')
BackIm = cv2.imread(BackCorr)

RawImagePath = filedialog.askdirectory(title='Select The folder with JPG files that need background correcting')
os.chdir(RawImagePath)
ImList = glob.glob('*.JPG')

cnt = 0
for i in ImList:
    cnt = cnt+1
    print("Processing File: {}/{}".format(cnt,len(ImList)))
    PlantSzClr.bgcorr(i,BackIm)
