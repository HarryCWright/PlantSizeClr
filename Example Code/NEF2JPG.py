import os, glob
from tkinter import filedialog, Tk
import PlantSzClr

root = Tk()
root.withdraw()
RawImagePath = filedialog.askdirectory(title='Select The folder with raw files')
os.chdir(RawImagePath)

#########Change this '*.Nef' to alternative if your camera captures in a
#########different formal

ImList = glob.glob('*.Nef')

cnt = 0
for i in ImList:
    cnt = cnt+1
    print("Processing File: {}/{}".format(cnt,len(ImList)))
    PlantSzClr.NEF2JPG(i)
