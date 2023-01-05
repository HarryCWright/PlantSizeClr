import os, glob
from tkinter import filedialog, Tk
import PlantSzClr

  
   
root = Tk()
root.withdraw()
RawImagePath = filedialog.askdirectory(title='Select The folder with JPG files that need cropping')
os.chdir(RawImagePath)
ImList = glob.glob('*.JPG')

cnt = 0
for imageAn in ImList:  
    cnt = cnt+1
    print("Processing File: {}/{}".format(cnt,len(ImList)))
    PlantSzClr.ImCrop(imageAn,25)

    

