# -*- coding: UTF-8 -*-
from __future__ import print_function  
import cv2
import argparse
import os
import numpy as np
from time import time
import random

def calculateHomography(correspondences):
    aList = []
    for eachcorr in correspondences:
        p1 = np.matrix([eachcorr.item(0), eachcorr.item(1), 1])
        p2 = np.matrix([eachcorr.item(2), eachcorr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)
    matrixA = np.matrix(aList)
    u, s, v = np.linalg.svd(matrixA)
    h = np.reshape(v[8], (3, 3))
    h = (1/h.item(8)) * h
    return h


def geometricDistance(correspondence, h):
    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2
    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    distance = np.linalg.norm(error)
    return distance

def ransac(correspondence, thresh):
    maxInliers = []
    finalH = None
    for i in range(100):
        corr1 = correspondence[random.randrange(0, len(correspondence))]
        corr2 = correspondence[random.randrange(0, len(correspondence))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = correspondence[random.randrange(0, len(correspondence))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = correspondence[random.randrange(0, len(correspondence))]
        randomFour = np.vstack((randomFour, corr4))

        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(correspondence)):
            d = geometricDistance(correspondence[i], h)
            if d < 5:
                inliers.append(correspondence[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        if len(maxInliers) > (len(correspondence)*thresh):
            break
    return finalH


def stitchTwoImg(img0,img6,IfisResult,IFAfterCentroid):

    
    if IfisResult==0:
        img0=cv2.resize(img0,(360,540))
        img6=cv2.resize(img6,(360,540))
    if IfisResult==1 and IFAfterCentroid==0 :
        img6=cv2.resize(img6,(img0.shape[1],img0.shape[0]))
    if IfisResult==1 and IFAfterCentroid==1:
        img0=cv2.resize(img0,(img6.shape[1],img6.shape[0]))
        
    i=2000
    surf = cv2.xfeatures2d.SURF_create(i)
    kp0, des0=surf.detectAndCompute(img0,None)
    surf = cv2.xfeatures2d.SURF_create(i)
    kp6, des6=surf.detectAndCompute(img6,None)
    outimg6 = cv2.drawKeypoints(img6,kp6,None,(255,0,0),4)


    matches=[]

    lenKP_0=len(kp0)


    lenKP_6=len(kp6)

    #两张img提取出来的特征点数目不同

    for i in range(lenKP_0):
        queryVector=des0[i]
        holder=[]
        for j in range(lenKP_6):
            holder.append(np.linalg.norm(des0[i] - des6[j]))
        candidateOfT=holder.index(min(holder))
        holder=[]
        for z in range(lenKP_0):
            holder.append(np.linalg.norm(des0[z] - des6[candidateOfT]))
        candidateOfQ=holder.index(min(holder))
        if i==candidateOfQ:
            newDMatch=cv2.DMatch(i,candidateOfT,0,min(holder))
            matches.append(newDMatch)
        if i!=candidateOfQ:
            pass

    matchedImg= cv2.drawMatches(img0,kp0,img6,kp6,matches1to2=matches,outImg=None,flags=2)


    kp_list=[kp0,kp6]

    correspondenceList=[]


    for match in matches:
        (x1, y1) = kp_list[0][match.queryIdx].pt
        (x2, y2) = kp_list[1][match.trainIdx].pt
        correspondenceList.append([x1, y1, x2, y2])
        
    corrs = np.matrix(correspondenceList)
    finalH = ransac(corrs,0.8)
    print("current homography matrix is: ")
    print(finalH)
    h1,w1 = img0.shape[:2]
    h2,w2 = img6.shape[:2]
    coordinateImg0 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]],dtype=np.float32)
    coordinateImg6 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]],dtype=np.float32)
    pts1 = coordinateImg0.reshape(-1,1,2)
    pts2 = coordinateImg6.reshape(-1,1,2)
    

    pts2_ = cv2.perspectiveTransform(pts2, finalH)
    #print(pts2_)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) 
    

    result = cv2.warpPerspective(img0, Ht.dot(finalH), (xmax-xmin, ymax-ymin))
    #print((xmax-xmin, ymax-ymin))

    for i in range(0,h1):
        for j in range(0,w1):
            if img6[i,j,:].all()==0:
                pass
            else:
                result[t[1]+i,t[0]+j,:] = img6[i,j,:]

    return result





def up_to_step_1(imgs):
    """Complete pipeline up to step 3: Detecting features and descriptors"""
    # ... your code here ...
    output=[]
    for img0 in imgs:
        img0=cv2.resize(img0,(360,540))
        i=2000
        surf = cv2.xfeatures2d.SURF_create(i)
        kp0, des0=surf.detectAndCompute(img0,None)
        outputImg0 = cv2.drawKeypoints(img0,kp0,None,-1,4)
        output.append(outputImg0)
    return output


def save_step_1(imgs, output_path='./output/step1'):
    """Save the intermediate result from Step 1"""
    i=0
    for eachImg in imgs:
        i=i+1
        cv2.imwrite(output_path+"/"+str(i)+".jpg",eachImg)

def up_to_step_2(imgs,fileNameList):
    """Complete pipeline up to step 2: Calculate matching feature points"""
    # ... your code here ...
    output = []
    i=0
    j=0
    MatchList=[]
    for x in range(len(imgs)-1):
        for y in range(x+1,len(imgs)):
            img0 = imgs[x]
            img6 = imgs[y]
            img0 = cv2.resize(img0,(360,540))
            img6 = cv2.resize(img6,(360,540))
            i=2000
            surf = cv2.xfeatures2d.SURF_create(i)
            kp0, des0=surf.detectAndCompute(img0,None)

            surf = cv2.xfeatures2d.SURF_create(i)
            kp6, des6=surf.detectAndCompute(img6,None)

            matches=[]
            lenKP_0=len(kp0)
            lenKP_6=len(kp6)
            for i in range(lenKP_0):
                queryVector=des0[i]
                holder=[]
                for j in range(lenKP_6):
                    holder.append(np.linalg.norm(des0[i] - des6[j]))
                candidateOfT=holder.index(min(holder))
                holder=[]
                for z in range(lenKP_0):
                    holder.append(np.linalg.norm(des0[z] - des6[candidateOfT]))
                candidateOfQ=holder.index(min(holder))
                if i==candidateOfQ:
                    newDMatch=cv2.DMatch(i,candidateOfT,0,min(holder))
                    matches.append(newDMatch)
                if i!=candidateOfQ:
                    pass
            matchedImg= cv2.drawMatches(img0,kp0,img6,kp6,matches1to2=matches,outImg=None,flags=2)
            output.append(matchedImg)
            
            MatchList.append([fileNameList[x],len(kp0),fileNameList[y],len(kp6),len(matches)])
    return output,MatchList

def save_step_2(imgs, match_list, output_path="./output/step2"):
    """Save the intermediate result from Step 2"""
    i=0
    for eachImg in imgs:
        #$imagenameA_#featurePointA_$imagenameB_#featurePointB_#matchingPair.jpg
        cv2.imwrite(output_path+"/"+match_list[i][0]+"_"+str(match_list[i][1])+"_"+match_list[i][2]+"_"+str(match_list[i][3])+"_"+str(match_list[i][4])+".jpg",eachImg)
        i=i+1


def up_to_step_3(imgs,fileNameList):
    """Complete pipeline up to step 3: estimating homographies and warpings"""
    i=0
    j=0
    pairs=[]
    match_list=[]
    for x in range(len(imgs)-1):
        for y in range(x+1,len(imgs)):
            img0 = imgs[x]
            img6 = imgs[y]
            img0 = cv2.resize(img0,(360,540))
            img6 = cv2.resize(img6,(360,540))
            i=2000
            surf = cv2.xfeatures2d.SURF_create(i)
            kp0, des0=surf.detectAndCompute(img0,None)

            surf = cv2.xfeatures2d.SURF_create(i)
            kp6, des6=surf.detectAndCompute(img6,None)

            matches=[]
            lenKP_0=len(kp0)
            lenKP_6=len(kp6)
            for i in range(lenKP_0):
                queryVector=des0[i]
                holder=[]
                for j in range(lenKP_6):
                    holder.append(np.linalg.norm(des0[i] - des6[j]))
                candidateOfT=holder.index(min(holder))
                holder=[]
                for z in range(lenKP_0):
                    holder.append(np.linalg.norm(des0[z] - des6[candidateOfT]))
                candidateOfQ=holder.index(min(holder))
                if i==candidateOfQ:
                    newDMatch=cv2.DMatch(i,candidateOfT,0,min(holder))
                    matches.append(newDMatch)
                if i!=candidateOfQ:
                    pass
            kp_list=[kp0,kp6]
            correspondenceList=[]

            for match in matches:
                (x1, y1) = kp_list[0][match.queryIdx].pt
                (x2, y2) = kp_list[1][match.trainIdx].pt
                correspondenceList.append([x1, y1, x2, y2])
        
            corrs = np.matrix(correspondenceList)
            finalH = ransac(corrs,0.8)
            #print(finalH)
            outputImg0 = cv2.warpPerspective(img0,finalH,(800,540))
            pairs.append(outputImg0)
            match_list.append("warped "+fileNameList[x]+"("+fileNameList[y]+" as reference)")
            pairs.append(img6)
            match_list.append("reference "+fileNameList[y]+"( pair with "+fileNameList[x]+")")
            
            
    return pairs,match_list


def save_step_3(img_pairs, match_list,output_path="./output/step3"):
    """Save the intermediate result from Step 3"""
    i=0
    for eachImg in img_pairs:
        cv2.imwrite(output_path+"/"+match_list[i]+".jpg",eachImg)
        i=i+1


def up_to_step_4(imgs):
    result=[]
    """Complete the pipeline and generate a panoramic image"""
    count=0
    for i in range(len(imgs)):
        if count>int(len(imgs)/2):
            img2=imgs[i]
            result = stitchTwoImg(img2,result,1,1)
            count=count+1
            continue
            
        if i==0:
            img1 = imgs[0]
            img2 = imgs[1]
            result = stitchTwoImg(img1,img2,0,0)
            count=count+1
        else:
            img2 = imgs[i]
            result = stitchTwoImg(result,img2,1,0)
            count=count+1
        
        
    #cv2.imshow("final result",result)
    cv2.imwrite("res/final result.jpg",result)
    return result


def save_step_4(result, output_path="./output/step4"):
    """Save the intermediate result from Step 4"""
    cv2.imwrite(output_path+"/The panorama image.jpg",result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        help="compute image stitching pipeline up to this step",
        type=int
    )

    parser.add_argument(
        "input",
        help="a folder to read in the input images",
        type=str
    )

    parser.add_argument(
        "output",
        help="a folder to save the outputs",
        type=str
    )

    args = parser.parse_args()

    imgs = []
    
    fileNameList=[]
    for filename in os.listdir(args.input):
        print(filename)
        fileNameList.append(filename)
        img = cv2.imread(os.path.join(args.input, filename))
        imgs.append(img)

    if args.step == 1:
        print("Running step 1")
        modified_imgs = up_to_step_1(imgs)
        save_step_1(modified_imgs, args.output)
    elif args.step == 2:
        print("Running step 2")
        modified_imgs, match_list = up_to_step_2(imgs,fileNameList)
        save_step_2(modified_imgs, match_list, args.output)
    elif args.step == 3:
        print("Running step 3")
        img_pairs,match_list= up_to_step_3(imgs,fileNameList)
        save_step_3(img_pairs,match_list, args.output)
    elif args.step == 4:
        print("Running step 4")
        panoramic_img = up_to_step_4(imgs)
        save_step_4(panoramic_img, args.output)
