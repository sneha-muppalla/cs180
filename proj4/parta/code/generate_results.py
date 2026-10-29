"""Reproducible report outputs; OpenCV only proposes/filters correspondences.
All final H estimates, interpolation, and blending use part4a.py.
"""
from pathlib import Path
import json, subprocess, sys, time
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import part4a as m
ROOT=m.ROOT
POINTS=ROOT/'points'; RESULTS=ROOT/'results'
POINTS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

def cli(*args):
 subprocess.run([sys.executable,str(Path(m.__file__)),*map(str,args)],check=True,stdout=subprocess.DEVNULL)

def pair(s,i,ref):
 a,b=[ROOT/'images'/f'set{s}_{k}.png' for k in (i,ref)]
 images=[m.load_image(p) for p in (a,b)]
 sift=cv2.SIFT_create(nfeatures=6000)
 features=[sift.detectAndCompute(np.uint8(im*255),None) for im in [cv2.cvtColor(x.astype('float32'),cv2.COLOR_RGB2GRAY) for x in images]]
 matches=cv2.BFMatcher().knnMatch(features[0][1],features[1][1],k=2)
 good=[p for p,q in matches if p.distance < .65*q.distance]
 src=np.float64([features[0][0][p.queryIdx].pt for p in good]); dst=np.float64([features[1][0][p.trainIdx].pt for p in good])
 _,mask=cv2.findHomography(src,dst,cv2.RANSAC,2.0)
 src,dst=src[mask.ravel()>0],dst[mask.ravel()>0]
 # Spatially distribute a compact, inspectable set of correspondences.
 order=np.argsort(np.linalg.norm(src-src.mean(0),axis=1))[::-1]; chosen=[]
 for j in order:
  if all(np.linalg.norm(src[j]-src[k])>65 for k in chosen):chosen.append(j)
  if len(chosen)==24:break
 if len(chosen)<8:raise ValueError('Insufficient distributed matches')
 src,dst=src[chosen],dst[chosen]
 H=m.computeH(src,dst); err=np.linalg.norm(m.transform(src,H)-dst,axis=1)
 keep=err<3; src,dst=src[keep],dst[keep]
 name=f'set{s}_{i}_to_{ref}'
 record=dict(source=str(a.resolve()),reference=str(b.resolve()),max_side=1200,source_points=src.tolist(),reference_points=dst.tolist(),selection_method='SIFT ratio test, RANSAC outlier filtering, spatial thinning; final normalized least squares computeH')
 path=POINTS/(name+'.json');path.write_text(json.dumps(record,indent=2))
 cli('pair',path,RESULTS/name)
 return path

pairs=[pair(1,1,2),pair(1,3,2),pair(2,1,3),pair(2,2,3)]
cli('mosaic',RESULTS/'mosaic_kitchen',*pairs[:2])
cli('mosaic',RESULTS/'mosaic_room',*pairs[2:])
cli('mosaic',RESULTS/'mosaic_kitchen_pair',pairs[0])

# Manually identified planar rectangles, in the 750x1000 preview coordinates.
rects=[('controls','set1_2',[[244,510],[407,505],[410,552],[244,558]],650,200),
       ('box','set2_3',[[110,643],[354,628],[354,784],[115,812]],600,400)]
for name,stem,points,w,h in rects:
 out=RESULTS/('rect_'+name);out.mkdir(exist_ok=True)
 image=m.load_image(ROOT/'images'/(stem+'.png'))
 src=np.array(points,float)*1.2;dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
 H=m.computeH(src,dst)
 record=dict(source=stem,source_points=src.tolist(),reference_points=dst.tolist(),max_side=1200,selection_method='Manually identified corners; target aspect ratio approximate')
 (out/'points.json').write_text(json.dumps(record,indent=2));np.savetxt(out/'H.txt',H)
 timings={}
 for method in ['nearest','bilinear']:
  samples=[]
  for _ in range(5):
   start=time.perf_counter();warp,mask,_=m.warp_image(image,H,method,(0,0,w-1,h-1));samples.append(time.perf_counter()-start)
  timings[method]=float(np.median(samples));m.save_image(out/(method+'.png'),warp)
 (out/'timings.json').write_text(json.dumps(timings,indent=2))
 fig,ax=plt.subplots(figsize=(5,7));ax.imshow(image);polygon=np.vstack((src,src[0]));ax.plot(*polygon.T,'r-')
 for i,(x,y) in enumerate(src,1):ax.text(x,y,str(i),color='red')
 ax.axis('off');fig.savefig(out/'original.png',bbox_inches='tight',dpi=140);plt.close(fig)
 # Same region enlarged without display smoothing so interpolation differences remain visible.
 for method in ['nearest','bilinear']:
  with Image.open(out/(method+'.png')) as im:
   x,y=im.width//3,im.height//3
   im.crop((x,y,x+90,y+90)).resize((360,360),Image.Resampling.NEAREST).save(out/(method+'_crop.png'))
print('Generated four correspondence reports, three mosaics, and two rectifications.')
