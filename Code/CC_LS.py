import os, glob,csv
from tkinter import filedialog, Tk
from tkinter.filedialog import askopenfilename
import PlantSzClr


root = Tk()
root.withdraw()


CCRGB = []
SwatchFile = askopenfilename(title='Please select the CSV file with ACTUAL swatch information')
with open(SwatchFile) as CCfile:
    csv_read = csv.reader(CCfile,quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_read:
        CCRGB.append(row)

RawImagePath = filedialog.askdirectory(title='Select The folder with .jpg files for correction')
os.chdir(RawImagePath)
ImList = glob.glob('*.JPG')

cnt = 0
for i in ImList:
    cnt = cnt+1
    print("Processing File: {}/{}".format(cnt,len(ImList)))
    PlantSzClr.ColourCorrect(i,CCRGB)
