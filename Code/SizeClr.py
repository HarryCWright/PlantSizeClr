import os, glob,csv, cv2
from tkinter import filedialog, Tk
from tkinter.filedialog import askopenfilename
import PlantSzClr

root = Tk()
root.withdraw()


RawImagePath = filedialog.askdirectory(title='Select The folder with .jpg files for size and colour ID')
os.chdir(RawImagePath)
ImList = glob.glob('*.JPG')

csvname = input("Enter name of csv to save data:  ") ##### INSERT NAME OF CHOSEN CSV FILE
isExist = os.path.isfile(csvname+".csv")
with open(csvname+".csv",'a', newline='') as f:
    head = ['Name','Dist 1','Dist 2','Area','R','G','B','L','a','b','Y','U','V']
    csvwrite = csv.DictWriter(f,fieldnames = head)
    
    if not isExist:
        csvwrite.writeheader()

    cnt = 0
    for i in ImList:
        cnt = cnt+1
        print("Processing File: {}/{}".format(cnt,len(ImList)))
        Im = cv2.imread(i)
        RGB = PlantSzClr.GetRGB(Im,150)
        YUV = PlantSzClr.rgb2yuv(RGB)
        Lab = PlantSzClr.rgb2lab(RGB)
        
        if cnt ==1:
            ObN = askopenfilename(title='Please select the JPG file for object of known size')
            
            Ob =  cv2.imread(ObN)
            
        size = PlantSzClr.sizefinder (Im,20.3,Ob)  ##### REPLACE 20.3mm WITH WIDTH OF REFERENCE OBJECT ON FAR LEFT (ex 1P COIN)
        csvwrite.writerow({'Name': i, 'Dist 1': size[1], 'Dist 2': size[2],'Area': size[0],
                           'R': RGB[0], 'G': RGB[1], 'B': RGB[2],'L': Lab[0],
                           'a': Lab[1], 'b': Lab[2], 'Y': YUV[0], 'U': YUV[1], 'V': YUV[2]})

