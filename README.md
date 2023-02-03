# PlantSizeClr

## Introduction 
Figure 1 A-F shows the general protocol and pipeline using the presented scripts for image correction, object detection and separation and colour and size determination and in depth details are provided.

1.	A background image should be taken with the digital camera to perform a background correction that corrects for any light gradient due to the camera sensor or inhomogeneous lighting conditions. An example of a background image is provided in supplementary file S1.
2.	Photograph plant tissue using a digital camera, ensuring that plant tissue, colour checker board as well as an object of known size are all visible in the photo. Background removal and object separation relies on using a white background so users should capture images with a white background. The colour checker should be in a landscape orientation and the object of interest should be the left most object (other than the colour checker board) an example of a good image is given in Figure 1 A and in more detail in supplementary file S2.

*Note 1: Software scripts are available for photos taken in both landscape and portrait, if using the SpyderChecker24 colour checker, the light blue swab should be in the top right.*
 
*Note 2: Photographs can be taken in any lighting condition; however, the use of natural light or a light box reduces the initial error of the images. It is important to ensure that there are no shadows cast onto the image or projected from the plant tissues, as this affects colour correction and object separation.*

3.	If images are captured in RAW format, they need to be converted to *.jpg format, if a Nikon camera is used that saves RAW files as *.NEF file, then the included script NEF2JPG.py can be used to convert the file to jpg. This script can batch process all the files in a folder.
4.	Background correction is done on the *.jpg files. The script BG_Corr.py is able to batch process all files in a particular folder. The script will prompt you to select the background image file captured for background correction. It will then ask you for the folder with all the images that required background correction before corrected the files. Files will be saved using their initial filename plus ”_BGcorrected.jpg”.
5.	Colour correction is done on background corrected jpg files. It is important use the script ClrCorr.py (portrait) or ClrCorr_LS.py (landscape) depending on the orientation of the images. The script will prompt you for the *.csv file that contains the RGB data of your colour correction swatches (if using SpyderCheckr24 this is supplied in supplementary file S3). The order of the colour swatches entered into the .csv file is important and supplementary file S4 demonstrates the order the values need to be entered into the file. It will then ask you for the folder that contains the images that you wish to correct and proceed to colour correct the batch of images. It will print out the r2 values of the colour correction matrix and these should all be above 0.95 if images have been successfully corrected. Corrected images will be saved using their initial filename plus the suffix “_fin.jpg”. Furthermore a .csv file will be generated that saves the average swatch error for each file before and after colour correction. 
6.	For object identification and separation it is necessary to crop the colour corrector out of the colour corrected images as shown in Figure 1 D. The provided script Crop.py is able to do this cropping. It will prompt you to select the folder with the corrected images and then display the images one at a time. The user must click and drag a rectangle on the image with their mouse to select the region of interest (ROI). The region of interest should include all objects of interest as well as the object of known size. The object of known size should now be the left most object in the image. Once the user is happy with the ROI they press enter and this will crop and save the image and open the following image. Images will be saved as their initial filename plus the suffix “_crop.jpg”.
7.	Background removal is done using the script BGRem.py. The user will be prompted to select the folder that has the images which require background removal. This script also has a threshold value that can be changed if the background removal is either cropping the objects of interest by being too aggressive in thresholding or not removing the background entirely. This is changed on line 16 of the script BGRem.py (a good starting value is 150). Images with their background removed are saved as their initial name plus the suffix “_BGRem.jpg”.
8.	Object separation is done using the script ObID.py which will prompt the user to select the folder which has the image files for object separation. Once the script is run, an image will be displayed with an identified object highlighted and the user will be asked to give this object a name in the IDE software used to run the script. The object will then be saved as a jpg file with this name. This will then repeat for the next object in the image and once all objects have been named the script will move to the next image in the folder. 
9.	Finally colour and size information can be extracted from the individual saved objects. This is done using the script SizeClr.py. This will prompt the user for the folder with the separated object images and for the file that contains the object of known size. The user will be prompted for the height and width of the object in the unit of interest and should enter this in the IDE. The script will then work through each of the objects and extract their size data (width, height and area) as well as the colour in three colour spaces; RGB, YUV and CIELAB. For CIELAB it is assumed that the observer is 2° and the illuminant is a CIE standard illuminant D65. If this is not the case the user should change these values on line 583-585 of script PlantSzClr.py. Colour and size information is saved in a new .csv file. The user will be prompted to give the name of this csv file and information is saved in columns as shown in Figure 1 F.

<p align="center">
  <img src="https://user-images.githubusercontent.com/104008615/216567524-8fb0970a-f0c6-41cb-9e39-6877a8ba120d.png" />
</p>
Figure 1: The framework for determining the size and colour of objects for plant phenotyping. The steps consist of (A) converting the raw file into a jpg file (optional) using the script NEF2JPG.py, (B) doing a background correction to account for the vignette effect by the camera lens using the BG_Corr.py file, (C) colour correction using a colour checker card using ClrCorr.py, (D) cropping of image for object identification using Crop.py, (E) removal of the white background using BGRem.py (F) separation of objects using ObID.py and (G) determination of the size and colour of objects using SizeClr.py

Above text and figure from: Wright, H.C. *et al.*, L. Free and Open-Source Software for object detection, size, and colour determination for use in plant phenotyping. *Plant Methods.* (Submitted). (I.F. 5.82)

## Required Packages
cv2 version 4.5.1<br>
imutils version 0.5.4<br>
csv version 1.0<br>
imagio version 2.9.0<br>
rawpy version 0.17.1<br>
numpy version 1.18.5<br>
sklearn version 0.23.1<br>
PIL version 7.2.0<br>
colour_checker_detection version 0.1.2<br>
as well as the OS, glob and tkinter which are part of the standard python library.<br>

## Test Images

Test images are provided for using the software.

