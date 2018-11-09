#  -*- coding:UTF-8 -*-
import numpy as np
import cv2
import matplotlib as plt
import sys, getopt

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,'h',["input=","output="])
   except getopt.GetoptError:
      print "python filter.py –-input input.png –-output output.png"
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print "python filter.py –-input input.png –-output output.png"
         sys.exit()
      elif opt == "--input":
         inputfile = arg
      elif opt =="--output":
         outputfile = arg
         


   img=cv2.imread(inputfile)
   #threshold is 144
   threshold = 144


   img = cv2.blur(img,(5,5))


   for column in range(int(636)):
           for row in range(int(510)):
               if img[row,column,0]>=threshold or img[row,column,1]>=threshold or img[row,column,2]>=threshold:
                   img[row,column,:]=255
               else:
                   img[row,column,:]=0
                   
   cv2.imshow('img',img)

   cv2.imwrite(outputfile,img)



if __name__ == "__main__":
   main(sys.argv[1:])
