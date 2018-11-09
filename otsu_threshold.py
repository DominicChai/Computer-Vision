#-*- conding:UTF-8 -*-
import numpy as np
import cv2
import matplotlib as plt
import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   ifPrintThreshold = False
   try:
      opts, args = getopt.getopt(argv,'h',["input=","output=","threshold"])
   except getopt.GetoptError:
      print "python otsu_threshold.py --input input.png --output binary.png --threshold"
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print "python otsu_threshold.py --input input.png --output binary.png --threshold"
         sys.exit()
      elif opt == "--input":
         inputfile = arg
      elif opt =="--output":
         outputfile = arg
      elif opt =="--threshold":
          ifPrintThreshold=True
   otsu(inputfile,outputfile,ifPrintThreshold)

def otsu(inputfile,outputfile,ifPrintThreshold):
    img=cv2.imread(inputfile)
    #print(img.shape) #(width 510, length 636, layers 3)
    width=img.shape[0] #510
    length=img.shape[1] #636
    layers=img.shape[2]  #3

    listOfInter_class_variance=[]
    for threshold in range(256):#the threshold will be 0-255(closed)
        listBig=[]
        listSmall=[]

        #since the layers are in the same value. we can just iterate first layer
        for column in range(length):
            for row in range(width):
                if img[row,column,0]>=threshold:
                    listBig.append(img[row,column,0])
                else:
                    listSmall.append(img[row,column,0])
        if len(listBig)==0 or len(listSmall) ==0:
            inter_class_variance = 0
        else:
            inter_class_variance = len(listBig)*len(listSmall)*(np.mean(listBig)-np.mean(listSmall))**2
        print "inter_class_variance for threshold "+str(threshold)+" is "+str(inter_class_variance)
        listOfInter_class_variance.append(inter_class_variance)
        

    bestThreshold = listOfInter_class_variance.index(max(listOfInter_class_variance))

    for column in range(int(length)):
            for row in range(int(width)):
                if img[row,column,0]>=bestThreshold or img[row,column,1]>=bestThreshold or img[row,column,2]>=bestThreshold:
                    img[row,column,:]=255
                else:
                    img[row,column,:]=0


    cv2.imwrite(outputfile,img)
    if(ifPrintThreshold):
        print "the bestThreshold by otsu method is "+str(bestThreshold)


if __name__ == "__main__":
   main(sys.argv[1:])
