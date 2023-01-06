import cv2, os, imutils, csv, imageio, rawpy
import numpy as np
from imutils import perspective
from imutils import contours
from sklearn.linear_model import LinearRegression
from PIL import Image, ImageDraw,ImageOps
from colour_checker_detection import (colour_checkers_coordinates_segmentation)

###This converts a nikon raw file *.NEF to a *.jpg file for further 
##processing
def NEF2JPG(imFile):
    FileName = imFile.split('.')[0]
    with rawpy.imread(imFile) as raw:
       rgb = raw.postprocess(rawpy.Params(use_camera_wb=True))
    imageio.imsave(FileName + '.jpg', rgb)

##This function divides an image by the division matrix normalising the actual
##image and saves it as the file name __BGcorrected.jpg
def bgcorr(img, BackIm):
    
    BackImBlur = cv2.blur(BackIm,(5,5))
    BackImMul = np.zeros(np.shape(BackIm))
    BackImMul[:,:,0] = BackImBlur[:,:,0]/(np.max(BackImBlur[:,:,0]))
    BackImMul[:,:,1] = BackImBlur[:,:,1]/(np.max(BackImBlur[:,:,1]))
    BackImMul[:,:,2] = BackImBlur[:,:,2]/(np.max(BackImBlur[:,:,2]))
    
    FileName = img.split('.')[0]
    NewIm = cv2.imread(img)
    FinIm = np.zeros(np.shape(BackImMul))
    FinIm = NewIm/BackImMul
    FinIm = FinIm.clip(0,255)
    FinIm = FinIm.astype(np.uint8)
    cv2.imwrite(FileName+"_BGCorrected"+".jpg",FinIm)


###This is a function for determining the midpoint of a two coordinates
def mdpt(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

def fillEdge (NoPoints,Coord1,Coord2):
    appendVal = 0
    edgeArr = np.array([])

    if Coord2 > Coord1:
        delCoord = (Coord2 - Coord1)/NoPoints
        smalCoord = Coord1
    else:
        delCoord = (Coord1 - Coord2)/NoPoints
        smalCoord = Coord2
        
    for k in range(NoPoints):
        if k == 0:
            appendVal = smalCoord + 0.5 * delCoord + appendVal
        else:
            appendVal = appendVal + delCoord
        edgeArr = np.append(edgeArr,appendVal)
        
    return edgeArr

##############################################################################
#Determines the average difference between reference colour  and actul colour
def ErrCalc(ref,act):
    Err = np.average(((act[:,1]- ref[:,1])**2 + \
                         (act[:,2]- ref[:,2])**2 + \
                         (act[:,3]- ref[:,3])**2)**0.5)
    return Err

##########################GETS THE AVERAGE RGB OF EACH SWATCH#################
def avgRGB(IM, x, y):
    CropToSwatch = (x-10,y-10,x+10,y+10)
    CroppedIM = IM.crop(CropToSwatch)
    RGBImage = np.array(CroppedIM)
    Ravg = np.average(RGBImage[:,:,0])
    Gavg = np.average(RGBImage[:,:,1])
    Bavg = np.average(RGBImage[:,:,2])
    return Ravg, Gavg, Bavg

#########################FINDS THE RGB CORRECTION EQUATION####################
#####Ract = a0Robs + a1Gobs + a2Bobs + a3Robs^2 + a4Gobs%2 + a5Bobs^2#########
def corrEq(RGBAct,RGBObs):
    yr = np.zeros((24))
    yg = np.zeros((24))
    yb = np.zeros((24))
    x = np.zeros((24,9))
    
    for k in range(24):
        yr[k] = RGBAct[k,1]
        yg[k] = RGBAct[k,2]
        yb[k] = RGBAct[k,3]
        
        x[k,:] = [RGBObs[k,1], RGBObs[k,2], RGBObs[k,3], RGBObs[k,1]**2, 
                  RGBObs[k,2]**2, RGBObs[k,3]**2,RGBObs[k,1]**3, 
                  RGBObs[k,2]**3, RGBObs[k,3]**3]

    modelR = LinearRegression(fit_intercept=False).fit(x, yr)
    modelG = LinearRegression(fit_intercept=False).fit(x, yg)
    modelB = LinearRegression(fit_intercept=False).fit(x, yb)
    
    print("RGB Model Scores = ",
          np.round(modelR.score(x,yr),3),np.round(modelG.score(x,yg),3),
          np.round(modelB.score(x,yb),3))
    
    return modelR,modelG,modelB   

##############################################################################
#Given a set of values, returns the value, squared and to the power of three 
#value all as floats
def genpower(Value):
    Value = Value.astype(np.float)
    Value2 = np.square(Value)
    Value3   = np.power(Value, 3)
    return (Value,Value2,Value3)    

##############################################################################
#Corrects colour bands given model coefficents and the colour values, values 
#squared and values to the power of 3
def corrVal(modelCoeff,BandPowers):
    CorrS = modelCoeff[0]* BandPowers[0] + modelCoeff[1]* BandPowers[1]\
        + modelCoeff[2]* BandPowers[2] + modelCoeff[3]* BandPowers[3]\
        + modelCoeff[4]* BandPowers[4] +modelCoeff[5]* BandPowers[5]\
        + modelCoeff[6]* BandPowers[6] +modelCoeff[7]* BandPowers[7]\
        + modelCoeff[8]* BandPowers[8] 
    return CorrS

def GenFinalPosL(ThreshCentres):
        c0 = ThreshCentres[0:4,:]
        c0 = c0[c0[:,0].argsort()]
        c0x = np.array([3,2,1,0]).reshape(4,1)
        c0 = np.hstack((c0x,c0))
        c1 = ThreshCentres[4:8,:]
        c1 = c1[c1[:,0].argsort()]
        c1x = np.array([7,6,5,4]).reshape(4,1)
        c1 = np.hstack((c1x,c1))
        c2 = ThreshCentres[8:12,:]
        c2 = c2[c2[:,0].argsort()]
        c2x = np.array([11,10,9,8]).reshape(4,1)
        c2 = np.hstack((c2x,c2))
        c3 = ThreshCentres[12:16,:]
        c3 = c3[c3[:,0].argsort()]
        c3x = np.array([15,14,13,12]).reshape(4,1)
        c3 = np.hstack((c3x,c3))
        c4 = ThreshCentres[16:20,:]
        c4 = c4[c4[:,0].argsort()]
        c4x = np.array([19,18,17,16]).reshape(4,1)
        c4 = np.hstack((c4x,c4))
        c5 = ThreshCentres[20:24,:]
        c5 = c5[c5[:,0].argsort()]
        c5x = np.array([23,22,21,20]).reshape(4,1)
        c5 = np.hstack((c5x,c5))
    
        
        FinalPos = np.vstack((c0,c1,c2,c3,c4,c5))
        FinalPos = FinalPos[FinalPos[:,0].argsort()]
        return FinalPos
    
def GenFinalPosP(ThreshCentres):
        R0 = ThreshCentres[0:6,:]
        R0 = R0[R0[:,0].argsort()]
        R0x = np.array([0,4,8,12,16,20]).reshape(6,1)
        R0 = np.hstack((R0x,R0))
        R1 = ThreshCentres[6:12,:]
        R1 = R1[R1[:,0].argsort()]
        R1x = np.array([1,5,9,13,17,21]).reshape(6,1)
        R1 = np.hstack((R1x,R1))
        R2 = ThreshCentres[12:18,:]
        R2 = R2[R2[:,0].argsort()]
        R2x = np.array([2,6,10,14,18,22]).reshape(6,1)
        R2 = np.hstack((R2x,R2))
        R3 = ThreshCentres[18:24,:]
        R3 = R3[R3[:,0].argsort()]
        R3x = np.array([3,7,11,15,19,23]).reshape(6,1)
        R3 = np.hstack((R3x,R3))
        
        FinalPos = np.vstack((R0,R1,R2,R3))
        FinalPos = FinalPos[FinalPos[:,0].argsort()]
        return FinalPos
    
def GetCenL(SortCorn2):
        xcorn, ycorn = np.array([0, 0, 0, 0]),np.array([0, 0, 0, 0])
        xcorn[0], ycorn [0] = SortCorn2[3][0], SortCorn2[3][1]
        xcorn[1], ycorn [1] = SortCorn2[1][0], SortCorn2[1][1]
        xcorn[2], ycorn [2] = SortCorn2[0][0], SortCorn2[0][1]
        xcorn[3], ycorn [3] = SortCorn2[2][0], SortCorn2[2][1]

        edgex1 = fillEdge(4,xcorn[1],xcorn[0])
        edgex2= fillEdge(4,xcorn[2],xcorn[3])
        edgey1 = fillEdge(6,ycorn[2],ycorn[1])
        edgey2 = fillEdge(6,ycorn[3],ycorn[0])
         
        edgey1x, edgey2x = np.array([]),np.array([])
        for j in range(6):
            edgey1x = np.append(edgey1x,xcorn[1]+(xcorn[2]-xcorn[1])/(ycorn[1]-ycorn[2])*(edgey1[j]-ycorn[2]))
            edgey2x = np.append(edgey2x,xcorn[0]+(xcorn[3]-xcorn[0])/(ycorn[0]-ycorn[3])*(edgey2[j]-ycorn[3]))
        
        edgex1y, edgex2y = np.array([]),np.array([])
        for j in range(4):
            edgex1y = np.append(edgex1y,ycorn[1]+(ycorn[0]-ycorn[1])/(xcorn[0]-xcorn[1])*(edgex1[j]-xcorn[1]))
            edgex2y = np.append(edgex2y,ycorn[2]+(ycorn[3]-ycorn[2])/(xcorn[3]-xcorn[2])*(edgex2[j]-xcorn[2]))
        
        
        xcent, ycent = np.array([]),np.array([])
        for k in range(4):
            for j in range(6):        
                xcent = np.append(xcent,(0.5+k)*(edgey2x[j]-edgey1x[j])/4 + edgey1x[j])
                ycent = np.append(ycent,(0.5+j)*(edgex2y[k]-edgex1y[k])/6 + edgex1y[k])
                
        return(xcent,ycent)
                
def GetCenP(SortCorn2):
        xcorn, ycorn = np.array([0, 0, 0, 0]),np.array([0, 0, 0, 0])
        xcorn[1], ycorn [1] = SortCorn2[1][0], SortCorn2[1][1]
        xcorn[2], ycorn [2] = SortCorn2[0][0], SortCorn2[0][1]
        xcorn[3], ycorn [3] = SortCorn2[2][0], SortCorn2[2][1]
        xcorn[0], ycorn [0] = SortCorn2[3][0], SortCorn2[3][1]
        
        edgex1 = fillEdge(6,xcorn[1],xcorn[0])
        edgex2= fillEdge(6,xcorn[2],xcorn[3])
        edgey1 = fillEdge(4,ycorn[2],ycorn[1])
        edgey2 = fillEdge(4,ycorn[3],ycorn[0])
        
        
        edgey1x, edgey2x = np.array([]),np.array([])
        for j in range(4):
            edgey1x = np.append(edgey1x,xcorn[1]+(xcorn[2]-xcorn[1])/(ycorn[1]-ycorn[2])*(edgey1[j]-ycorn[2]))
            edgey2x = np.append(edgey2x,xcorn[0]+(xcorn[3]-xcorn[0])/(ycorn[0]-ycorn[3])*(edgey2[j]-ycorn[3]))
        
        edgex1y, edgex2y = np.array([]),np.array([])
        for j in range(6):
            edgex1y = np.append(edgex1y,ycorn[1]+(ycorn[0]-ycorn[1])/(xcorn[0]-xcorn[1])*(edgex1[j]-xcorn[1]))
            edgex2y = np.append(edgex2y,ycorn[2]+(ycorn[3]-ycorn[2])/(xcorn[3]-xcorn[2])*(edgex2[j]-xcorn[2]))
        
        
        xcent, ycent = np.array([]),np.array([])
        for k in range(6):
            for j in range(4):        
                xcent = np.append(xcent,(0.5+k)*(edgey2x[j]-edgey1x[j])/6 + edgey1x[j])
                ycent = np.append(ycent,(0.5+j)*(edgex2y[k]-edgex1y[k])/4 + edgex1y[k])
                
        return(xcent,ycent)

def ColourCorrect(CCimg,CCRGB):

    ###############Start (needs more functions)###########

    FileName = CCimg.split('.')[0] 
    j , k , c= 0, 0, 0
    x = Image.open(CCimg)
    x = ImageOps.exif_transpose(x)
    rgbshape = np.shape(x)
    
    
    ##Add 180 degrees if your photos are upside down
    if rgbshape[0] < rgbshape[1]:
        x = x.transpose(Image.ROTATE_90)
    else:
        x = x#.transpose(Image.ROTATE_0)
    
    rgbarr = np.array(x)
    
    xcorn, ycorn = np.array([0, 0, 0, 0]),np.array([0, 0, 0, 0])
    SortCorn2 = np.zeros((4,2))
    
    coords2, coords, COORDS_Each, segmented_image = (
            colour_checkers_coordinates_segmentation(rgbarr, additional_data=True))
    if coords == []:
        print("COLOURCHECKER NOT FOUND, ABORTED")
    else:
        coordsarr = np.array(coords[0])
        SortCorn = sorted(coordsarr , key=lambda k: [k[0], k[1]])
        SortCorn2[0] = sorted([SortCorn[0],SortCorn[1]] , key=lambda k: k[1])[0]
        SortCorn2[1] = sorted([SortCorn[0],SortCorn[1]] , key=lambda k: k[1])[1]
        SortCorn2[2] = sorted([SortCorn[2],SortCorn[3]] , key=lambda k: k[1])[0]
        SortCorn2[3] = sorted([SortCorn[2],SortCorn[3]] , key=lambda k: k[1])[1]
        
        if rgbshape[0] < rgbshape[1]:
            xcent,ycent = GetCenL(SortCorn2)
        else:
            xcent,ycent = GetCenP(SortCorn2)

        
        
        #########################MASK IMAGE###########################################
        ImNoDraw = x.transpose(Image.ROTATE_270)
        MaskImage = Image.new('RGB',x.size ,color = 'black')
        
        MaskImage = MaskImage.transpose(Image.ROTATE_270 )
        ar2 = np.shape(MaskImage)[0]/np.shape(MaskImage)[1]
        resy2 = int((round(1440*ar2)))
        draw2 = ImageDraw.Draw(MaskImage)
        
        
        #######################CHANGE X AND Y TO NON RESIZED###########################
        
        xcorn = xcorn * np.shape(MaskImage)[1]/1440
        xcorn = xcorn.astype(np.uint16)
        ycorn = ycorn * np.shape(MaskImage)[0]/resy2
        ycorn = ycorn.astype(np.uint16)
        
        xcent = xcent * np.shape(MaskImage)[1]/1440
        xcent = xcent.astype(int)
        ycent = ycent * np.shape(MaskImage)[0]/resy2
        ycent = ycent.astype(int)

        ######################Draws BLUE on MASK IMAGE#################################
        
        draw2.polygon((xcorn[0],ycorn[0],
                       xcorn[1],ycorn[1],
                       xcorn[2],ycorn[2],
                       xcorn[3],ycorn[3]), fill = 'blue')
        
        ######################Draws White swatches on MASK IMAGE#######################
        
        
        for k in range(len(COORDS_Each)):
            Swx = np.array([COORDS_Each[k][0][0],COORDS_Each[k][1][0],COORDS_Each[k][2][0],COORDS_Each[k][3][0]])
            Swy = np.array([COORDS_Each[k][0][1],COORDS_Each[k][1][1],COORDS_Each[k][2][1],COORDS_Each[k][3][1]])
            Swx = Swx * np.shape(MaskImage)[1]/1440
            Swx = Swx.astype(np.uint16)
            Swy = Swy * np.shape(MaskImage)[0]/resy2
            Swy = Swy.astype(np.uint16)
   
            addsubx = int(max(Swx) - min(Swx))/2
            addsuby = int(max(Swy) - min(Swy))/2
            
            if MaskImage.getpixel((int(Swx[0]),int(Swy[0]))) == (0,0,255):
                draw2.polygon((min(Swx)+addsubx,min(Swy)+addsuby,
                              max(Swx)-addsubx,min(Swy)+addsuby,
                              max(Swx)-addsubx,max(Swy)-addsuby,
                              min(Swx)+addsubx,max(Swy)-addsuby), fill = 'white', outline  = 'white')
            
        ######################Makes unkown / missed swatches white#######################
        
        SwatchFound = 0      
        for k in range(24):
                if MaskImage.getpixel((int(xcent[k]),int(ycent[k]))) == (255,255,255):
                    SwatchFound = SwatchFound+1
                else:
                    draw2.polygon((xcent[k]-25,ycent[k]-25,
                                   xcent[k]+25,ycent[k]-25,
                                   xcent[k]+25,ycent[k]+25,
                                   xcent[k]-25,ycent[k]+25), fill = 'white', outline  = 'white')
                    
        ######################END MASK IMAGE###########################################
        open_cv_image = np.array(MaskImage)
    
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
        
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        ThreshCentres = np.array([])
        # loop over the contours
        for c in cnts:
        	# compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            ThreshCentres = np.append(ThreshCentres,(cX,cY))
                  
        MaskImage = MaskImage.transpose(Image.ROTATE_90)
    
        ThreshCentres = ThreshCentres.reshape(24,2)
        
        if rgbshape[0] < rgbshape[1]:
            FinalPos = GenFinalPosL(ThreshCentres)
        else:
            FinalPos = GenFinalPosP(ThreshCentres)
        
         
        RGBObservedMat = np.zeros((24,4))
        
        for j in range(24):
            avgR, avgG, avgB = avgRGB(ImNoDraw,int(FinalPos[j,1]),int(FinalPos[j,2]))
            RGBObservedMat[j,:] = [j,avgR,avgG,avgB]
        
        RGBobs255 = RGBObservedMat.astype(np.uint8)
        CCRGB255 = np.asarray(CCRGB)
        Rmodel, Gmodel, Bmodel = corrEq(CCRGB255,RGBobs255)
           
        ############Generates the final numpy array for the corrected image###
        finIm = np.zeros(np.shape(rgbarr))
        
        rfin, rfin2, rfin3 = genpower(rgbarr[:,:,0])
        gfin, gfin2, gfin3 = genpower(rgbarr[:,:,1])
        bfin, bfin2, bfin3 = genpower(rgbarr[:,:,2])
        
        allpowers = [rfin, gfin, bfin, rfin2, gfin2, bfin2, rfin3, gfin3, bfin3]
        
        rCorr = corrVal(Rmodel.coef_,allpowers)
        gCorr = corrVal(Gmodel.coef_,allpowers)
        bCorr = corrVal(Bmodel.coef_,allpowers)
    
        finIm[:,:,0] = rCorr
        finIm[:,:,1] = gCorr
        finIm[:,:,2] = bCorr
        finIm[finIm > 254] = 254
        finIm[finIm < 0] = 0
        finIm = finIm.astype(np.uint8)
        finIm = Image.fromarray(finIm)
        finIm = finIm.transpose(Image.ROTATE_270)
        finIm = finIm.save(FileName + "_Fin.jpg") 
        ######################Determines the corrected swatch values##########
        
        AdjSwatch = np.zeros(np.shape(RGBobs255))
        
        rsw, rsw2, rsw3 = genpower(RGBobs255[:,1])
        gsw, gsw2, gsw3 = genpower(RGBobs255[:,2])
        bsw, bsw2, bsw3 = genpower(RGBobs255[:,3])
        
        BandPowers = [rsw,gsw,bsw,rsw2,gsw2,bsw2,rsw3,gsw3,bsw3]
        rswCorr = corrVal(Rmodel.coef_,BandPowers)  
        gswCorr = corrVal(Gmodel.coef_,BandPowers) 
        bswCorr = corrVal(Bmodel.coef_,BandPowers) 
        
        AdjSwatch[:,1] = rswCorr
        AdjSwatch[:,2] = gswCorr
        AdjSwatch[:,3] = bswCorr
        AdjSwatch = AdjSwatch.astype(np.uint8)
        
        #####################CALCULATE DIFFERENCES BETWEEN SWATCHES BEFIRE AFTER######
        
        ErrBeforeCorr = ErrCalc(CCRGB255,RGBobs255)
        ErrAfterCorr = ErrCalc(CCRGB255,AdjSwatch)

        #########################SAVE IMAGES AND ERROR###############################################
        
        isExist = os.path.isfile("ErrorRed.csv")
        with open('ErrorRed.csv','a+') as statWrite:
            head = ['FileName', 'ErrorNoCor', 'ErrorCorr']
            csvwrite = csv.DictWriter(statWrite, delimiter=',', lineterminator='\n', fieldnames = head)
        
            if not isExist:
                csvwrite.writeheader()
            csvwrite.writerow({'FileName': FileName,
                   'ErrorNoCor':    ErrBeforeCorr,
                   'ErrorCorr':    ErrAfterCorr})

######Function for cropping images, scale_percent is a scaling factor to 
#######make the image fit in the screen. change this value as needed.

def ImCrop(CropImg,scale_percent):
    FileName = CropImg.split('.')[0]
    img = cv2.imread(CropImg)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #select ROI function
    roi = cv2.selectROI(resized,cv2.WINDOW_NORMAL)
    #Crop selected roi from raw image
    roi_cropped = img[int(roi[1]*(100/scale_percent)):int((roi[1]+roi[3])*(100/scale_percent)),
                         int(roi[0]*(100/scale_percent)):int((roi[0]+roi[2])*(100/scale_percent))]
    
    cv2.imwrite(FileName+"_crop.jpg",roi_cropped)
    #hold window
    cv2.destroyAllWindows() 
    cv2.waitKey(0)

#This function removes a white background, leaving only objects
#It may be possible to alter it by changing the lower/upper thresholds
#to remove coloured backgrounds but this hasn't been trialled.
#Lowthresh is used to determine what is background, change this value to change
#how much background is removed. Higher = stricter (more bg removed)
def bgremove(imageAn,LowThresh):
    img = cv2.imread(imageAn)
    FileName = os.path.splitext(imageAn)[0]
    #Set the low threshold, this will depend on the brightness of your background
    lower = np.array([LowThresh, LowThresh, LowThresh])
    upper = np.array([255, 255, 255])
    thresh = cv2.inRange(img, lower, upper)

    #Morphology excludes small pixels and removes all except large objects
    #You can change the values below (10,10) depending on the size of your
    #objects
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    nobg = np.bitwise_or(img, morph[:,:,np.newaxis])
    cv2.imwrite(FileName+"_BGRem.jpg", nobg)

#########This function seperates an image into each object within the image

def ObSep(imageAn,scale_percent):
    
    nobg = cv2.imread(imageAn)

    # Canny Edge Detection on no background image
    img_gray = cv2.cvtColor(nobg, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=200) 
    
    # find contours in the edge map 
    cntours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntours = imutils.grab_contours(cntours)
    
    # sort contours left-to-right   
    (cntours, _) = contours.sort_contours(cntours)
    
    # loop over the contours individually
    for c in cntours:
        if cv2.contourArea(c) < 100: #ignore/fly through contours that are not big enough
            continue
    
    # compute the rotated bounding box of the contour
    
        orig = nobg.copy() ####img
        x,y,w,h = cv2.boundingRect(c)
        bbox = cv2.minAreaRect(c)
        bbox = cv2.boxPoints(bbox)
        bbox = np.array(bbox, dtype="int")
    
    # order the contours and draw bounding box
        
        bbox = perspective.order_points(bbox)
        cv2.drawContours(orig, [bbox.astype("int")], -1, (0, 255, 0), 2)
        bbox = bbox.astype(int)
    
    #Cropped image of just 1 object
        
        crop = nobg[y:y+h,x:x+w]
        
    # loop over the original points in bbox and draw them; 5px red dots
        
        for (x, y) in bbox:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255),-1)
            
            # unpack the ordered bounding bbox; find midpoints
            
            (tl, tr, br, bl) = bbox
            (tltrX, tltrY) = mdpt(tl, tr)
            (blbrX, blbrY) = mdpt(bl, br)
            (tlblX, tlblY) = mdpt(tl, bl)
            (trbrX, trbrY) = mdpt(tr, br)
    
            # draw the mdpts on the image (blue);lines between the mdpts (yellow)
            
            cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0,0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0,0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0,0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0,0), -1)
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX),int(blbrY)),(0, 255, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX),int(trbrY)),(0, 255, 255), 2)
            
    
    # show the object to be named in a seperate window
        width = int(nobg.shape[1] * scale_percent / 100)
        height = int(nobg.shape[0] * scale_percent / 100)
        dim = (width, height)
            #    resize image
        resized = cv2.resize(orig, dim, interpolation = cv2.INTER_AREA)
    
        cv2.imshow(imageAn,resized)
        cv2.waitKey(1)
    
    #Ask for name and saves cropped image of chosen object

        ObName = input('Name object: ')
        cv2.destroyAllWindows()
        cv2.imwrite("{0}_crop.jpg".format(ObName), crop)
        
#Converts RGB to L*a*b* colour space
def rgb2lab ( inputColor ) :

    num = 0
    RGB = [0, 0, 0]

    for value in inputColor :
        value = float(value) / 255
 
        if value > 0.04045 :
           value = ((value + 0.055) / 1.055 ) ** 2.4
        else :
            value = value / 12.92
 
        RGB[num] = value * 100
        num = num + 1
 
    XYZ = [(RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805)/ 95.047 , # ref_X =  95.047   Observer= 2°, Illuminant= D65
           (RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722)/ 100.0,# ref_Y = 100.000
          (RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505)/ 108.883] # ref_Z = 108.883
 
    num = 0
    for value in XYZ :
 
        if value > 0.008856 :
            value = value ** ( 0.3333333333333333 )
        else :
            value = ( 7.787 * value ) + ( 16 / 116 )
 
        XYZ[num] = value
        num = num + 1
 
    Lab = np.round([( 116 * XYZ[ 1 ] ) - 16,
           500 * ( XYZ[ 0 ] - XYZ[ 1 ]), 
           200 * ( XYZ[ 1 ] - XYZ[ 2 ] )],3)

    return Lab 


#Convert RGB to YUV colour space
def rgb2yuv(RGB):
    YUV =  (0.256788235294118 * RGB[0] + 0.504129411764706 * RGB[1] + 0.0979058823529412 * RGB[2]+16,
            -0.148222900898508 * RGB[0] -0.290992785376001 * RGB[1] + 0.43921568627451  * RGB[2]+128,
            0.43921568627451 * RGB[0] -0.367788313613605 * RGB[1] -0.0714273726609046 * RGB[2]+128)
    YUV = np.round(YUV,0)
    return YUV

def GetRGB (imageAn,LowThresh):
    lower = np.array([LowThresh, LowThresh, LowThresh])
    upper = np.array([255, 255, 255])
    
    thresh = cv2.inRange(imageAn, lower, upper)

    #Morphology excludes small pixels and removes all except large objects
    #You can change the values below (10,10) depending on the size of your
    #objects
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    nobg = np.bitwise_or(imageAn, morph[:,:,np.newaxis])
    nobg2 = nobg[np.any(nobg != [255, 255, 255], axis=-1)]
    RGB = np.flip(np.average(nobg2, axis = (0)))
    
    return RGB

def sizefinder(imageAn,obSize,obIm):

    dA = imageAn.shape[0]
    dB = imageAn.shape[1]

    # use pixel_to_size ratio to compute object size

    pixel_to_size1 = obIm.shape[0] / obSize  
    pixel_to_size2 = obIm.shape[1] / obSize  
    avgPix = np.average([pixel_to_size1,pixel_to_size2])
    
    distA = dA / avgPix
    distB = dB / avgPix
    
    pixels = np.count_nonzero(np.all(imageAn != [255,255,255], axis = 2))
    # object area in cm^2
    objectArea = pixels/(avgPix**2)
    
    return (objectArea, distA, distB)

