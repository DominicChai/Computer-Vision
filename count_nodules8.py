#-*- coding:UTF-8 -*-
import numpy as np
import cv2
import matplotlib as plt
import random
import time
import sys, getopt


def main(argv):
   inputfile = ''
   outputfile = ''
   size=0
   try:
      opts, args = getopt.getopt(argv,'h',["input=","optional_output=","size="])
   except getopt.GetoptError:
      print "python count_nodules4.py --input binary_image.png --size n --optional_output nodules.png"
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print "python count_nodules4.py --input binary_image.png --size n --optional_output nodules.png"
         sys.exit()
      elif opt == "--input":
         inputfile = arg
      elif opt =="--optional_output":
         outputfile = arg
      elif opt =="--size":
          size=arg
   count8(inputfile,outputfile,size)



def count8(inputfile,outputfile,n):
    n=int(n)
    newImg=cv2.imread(inputfile)

    newImg[:,:,:]=255 #newImg is a white board

    img=cv2.imread(inputfile)

    numOfRows=img.shape[0] #510
    numOfColumns=img.shape[1] #636
    layers=img.shape[2]  #3

    label=1 #count the number

    levelMat=np.zeros((numOfRows,numOfColumns))
    for column in range(numOfRows): 
        for row in range(numOfColumns):
            levelMat[column,row]=int(0)


    dictionary={1:set()}


    for column in range(numOfRows): #column and numOfRows can not be the same name
        for row in range(numOfColumns):
            if img[column,row,0] == 255:
                continue
            else:
                try:
                    img[column-1,row,0]=img[column-1,row,0]
                    img[column,row-1,0]=img[column,row-1,0]
                    img[column-1,row-1,0]=img[column-1,row-1,0]
                except:
                    continue 
                
                if (levelMat[column-1,row]==0 and levelMat[column,row-1]==0 and levelMat[column-1,row-1]==0) :
                        #print("no neighbors with labels")
                        levelMat[column,row]=label
                        dictionary[label]=set()
                        label=label+1
                else:
                    a=[levelMat[column-1,row],levelMat[column,row-1],levelMat[column-1,row-1]]
                    #check three elements at a time
                    #rather than two
                    try:
                        a.remove(0)
                        a.remove(0)
                    except:
                        pass
                    if len(a)==1:
                        levelMat[column,row]=a[0]
                    if len(a)==2:
                        levelMat[column,row]=min(a)#assign smallest value
                        t=max(a)
                        try:
                            dictionary[levelMat[column,row]].add(t)
                            dictionary[t].add(levelMat[column,row])
                        except:
                            pass
                    if len(a)==3:#there are three available candidates
                        levelMat[column,row]=min(a)#assign smallest value
                        t=max(a)
                        try:
                            dictionary[levelMat[column,row]].add(t)
                            dictionary[t].add(levelMat[column,row])
                        except:
                            pass
                #print(dictionary)

    summ=set()
    for column in range(numOfRows): 
        for row in range(numOfColumns):
            summ.add(levelMat[column,row])

    for each in range(label):
        try:
            for eachElement in dictionary[each]:
                dictionary[eachElement]= dictionary[eachElement] | dictionary[each]
        except:
            pass

                    
    for column in range(numOfRows): #column and numOfRows can not be the same name
        for row in range(numOfColumns):
            if img[column,row,0] == 255:
                pass 
            else:
                try:
                    levelMat[column,row]=min(dictionary[levelMat[column,row]])
                except:
                    pass



    count=set()
    for column in range(numOfRows):
        for row in range(numOfColumns):
                if levelMat[column,row]!=0 and levelMat[column,row]!=0 :
                    count.add(levelMat[column,row])

    countlist=[]
    for column in range(numOfRows):
        for row in range(numOfColumns):
                if levelMat[column,row]!=0 and levelMat[column,row]!=255 :
                    countlist.append(levelMat[column,row])


    for i in range(label):
        try:
            if len(dictionary[i])==0:
                count.remove(i)
        except:
            pass
        
    t=0
    for LabelForNodules in count:
            
        if countlist.count(LabelForNodules)>n:
            t=t+1
        else:
            continue

        R=random.randint(0,255)
        G=random.randint(0,255)
        B=random.randint(0,255)
        
        for row in range(numOfColumns):
            for column in range(numOfRows):
                    if levelMat[column,row] == LabelForNodules:
                            newImg[column,row,0] = R
                            #print(newImg[column,row,0])
                            newImg[column,row,1] = G
                            #print(newImg[column,row,1])
                            newImg[column,row,2] = B
                            #print(newImg[column,row,2])
            
            
        
    print "the number of nodules is"+ str(t)
    cv2.imwrite(outputfile,newImg)


if __name__ == "__main__":
   main(sys.argv[1:])
