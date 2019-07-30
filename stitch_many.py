import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import random

def solve_dlt(x1,x2):

	X=[]
	for i in range(len(x1)):
		
		t1=[0.0,0.0,0.0,-1.0*x1[i][0][0],-1.0*x1[i][0][1],-1,x1[i][0][0]*x2[i][0][1],x1[i][0][1]*x2[i][0][1],x2[i][0][1]]
		t2=[1.0*x1[i][0][0],1.0*x1[i][0][1],1.0,0.0,0.0,0.0,-1.0*x1[i][0][0]*x2[i][0][0],-1.0*x1[i][0][1]*x2[i][0][0],-1.0*x2[i][0][0]]
		X.append(t1)
		X.append(t2)

	X=np.array(X,dtype='float64')
	u,d,vt=np.linalg.svd(X)
	v=np.transpose(vt)
	out=[]
	for i in range(len(v)):
		out.append(v[i][-1])
	#print(v)
	#print(v[-1:,:])
	#print("shape")
	v=np.reshape(np.array(out),(3,3))
	return v


def solve_ransac(src_pts,dst_pts):
	iterations=100
	err_thresh=1
	top_inlier_count=0
	final_H=[]
	for _ in range(iterations):
		inds=random.sample([i for i in range(len(src_pts))],5)
		x1=[]
		x2=[]
		for ind in inds:
			x1.append(src_pts[ind])
			x2.append(dst_pts[ind])

		H=solve_dlt(x1,x2)
		#H=H/H[2][2]
		n_src=[]
		projected=[]
		for i in range(len(src_pts)):
			tmp=[src_pts[i][0][0],src_pts[i][0][1],1.0]
			tmp=np.array(tmp)
			n_src.append(tmp)
			projected.append(np.dot(H,tmp.T))
		n_src=np.array(n_src,dtype='float64')
		#projected=np.dot(H,n_src.T)
		#print(projected.shape)
		inlier_count=0
		#print("P1",projected[0][0]/projected[0][2],projected[0][1]/projected[0][2])
		#print("D1",dst_pts[0][0][0],dst_pts[0][0][1])
		for i in range(len(projected)):

			e1=abs(projected[i][0]/projected[i][2]-dst_pts[i][0][0])+abs(projected[i][1]/projected[i][2]-dst_pts[i][0][1])
			print('error thresh',e1)
			if e1<err_thresh:
				inlier_count+=1
				#print("HHAHA")
		if inlier_count>top_inlier_count:
			top_inlier_count=inlier_count
			final_H=H

	return final_H

def stitch_2(f1,f2):
	MIN_MATCH_COUNT = 10
	img1 = cv.imread(f1,0)          # queryImage
	img2 = cv.imread(f2,0) # trainImage
	img1c = cv.imread(f1)
	img2c = cv.imread(f2)
	# Initiate SIFT detector
	sift = cv.xfeatures2d.SIFT_create()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# store all the good matches as per Lowe's ratio test.
	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)


	if len(good)>MIN_MATCH_COUNT:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
	    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
	    matchesMask = mask.ravel().tolist()
	    h,w = img1.shape
	    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	    dst = cv.perspectiveTransform(pts,M)
	    #img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
	    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
	    matchesMask = None


	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
	                   singlePointColor = None,
	                   matchesMask = matchesMask, # draw only inliers
	                   flags = 2)
	#print(src_pts[0])
	#print(dst_pts[0])
	#img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#plt.imshow(img3, 'gray'),plt.show()

	H=solve_ransac(dst_pts,src_pts)

	print("final homography is")
	print(H)
	#print(M)

	warp = cv.warpPerspective(img2c, H,(4*max(img1.shape[0],img2.shape[0]),4*max(img1.shape[1],img2.shape[1])))
	maxx,maxy=0,0
	for i in range(len(warp)):
		for j in range(len(warp[i])):
			if sum(warp[i][j])>0:
				if i>maxy:
					maxy=i
				if j>maxx:
					maxx=j

	for i in range(len(img1)):
		for j in range(len(img1[i])):
			if img1[i][j]>0:
				if i>maxy:
					maxy=i
				if j>maxx:
					maxx=j

	out=np.zeros((maxy,maxx,3),dtype='uint8')
	print(np.min(warp),np.max(warp))
	print(np.min(img1),np.max(img1))

	for i in range(len(out)):
		for j in range(len(out[i])):
			ans=0
			ans1=0
			ans2=0
			try:
				ans+=img1c[i][j][0]
				ans1+=img1c[i][j][1]
				ans2+=img1c[i][j][2]
				
				if sum(warp[i][j])>0:
					ans+=warp[i][j][0]
					ans1+=warp[i][j][1]
					ans2+=warp[i][j][2]
					if img1[i][j]>0:
						ans=ans/2
						ans1/=2
						ans2/=2
				
			except IndexError:
				#pass
				ans=warp[i][j][0]
				ans1=warp[i][j][1]
				ans2=warp[i][j][2]
				#print("HA")
			out[i][j][0]=ans
			out[i][j][1]=ans1
			out[i][j][2]=ans2
	cv.imwrite('temp.png',out)
	return out



def stitch_many(files):
	
	for i in range(len(files)-1):
		print(i)
		if i==0:
			tmp=stitch_2(files[i],files[i+1])
		else:
			tmp=stitch_2('temp.png',files[i+1])
	#cv.imshow('final',tmp)
import glob
imgs=glob.glob('test_images/img4*')
imgs.sort()
#stitch_many(['test_images/yosemite1.jpg','test_images/yosemite2.jpg','test_images/yosemite3.jpg','test_images/yosemite4.jpg'])
stitch_many(imgs)