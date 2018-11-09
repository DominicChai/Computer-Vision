#-*- coding:UTF-8 -*-
import numpy as np
import cv2
import matplotlib as plt
import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   numberOfGrid=0
   try:
      opts, args = getopt.getopt(argv,'h',["input=","output=","number="])
   except getopt.GetoptError:
      print "python otsu_threshold.py --input input.png --output binary.png --number n"
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print "python otsu_threshold.py --input input.png --output binary.png --number n"
         sys.exit()
      elif opt == "--input":
         inputfile = arg
      elif opt =="--output":
         outputfile = arg
      elif opt =="--number":
          numberOfGrid=arg
   grid_otsu(inputfile,outputfile,numberOfGrid)





def grid_otsu(inputfile,outputfile,n):

    
    img=cv2.imread(inputfile)

    width=img.shape[0] #510
    length=img.shape[1] #636
    layers=img.shape[2]  #3

    listOfInter_class_variance=[]
    #n=10
    n=int(n)
    remaider = img.shape[1] % n
    bandWidth= int((img.shape[1]- remaider)/n)


    for i in range(1+n):#iterate through each grid
        if i<n:
            process(i*bandWidth,i*bandWidth+bandWidth,width,i,img)
        if i==n: #the last grid, might overflow
            process(i*bandWidth,i*bandWidth+remaider,width,i,img)
            

    #cv2.imshow("img",img)
    cv2.imwrite(outputfile,img)

def process(startOfLength,endOfLength,width,i,img):
        listOfInter_class_variance=[]
        for threshold in range(256):#the threshold will be 0-255(closed)
            listBig=[]
            listSmall=[]
            
            #since the layers are in the same value. we can interate first layer
            for column in range(startOfLength,endOfLength):
                for row in range(width):
                    #print(img[column,row,0])
                    if img[row,column,0]>=threshold:
                        listBig.append(img[row,column,0])
                    else:
                        listSmall.append(img[row,column,0])

            if len(listBig)==0 or len(listSmall) ==0:
                inter_class_variance = 0
            else:
                inter_class_variance = len(listBig)*len(listSmall)*(np.mean(listBig)-np.mean(listSmall))**2
            #print(inter_class_variance)
            listOfInter_class_variance.append(inter_class_variance)
        #print("now is doing grid"+str(i+1))
        #print(listOfInter_class_variance.index(max(listOfInter_class_variance)))


        bestThreshold = listOfInter_class_variance.index(max(listOfInter_class_variance))
        #print(bestThreshold)

        for column in range(startOfLength,endOfLength):
                for row in range(int(width)):
                    if img[row,column,0]>=bestThreshold or img[row,column,1]>=bestThreshold or img[row,column,2]>=bestThreshold:
                        img[row,column,:]=255
                    else:
                        img[row,column,:]=0


                        

if __name__ == "__main__":
   main(sys.argv[1:])
