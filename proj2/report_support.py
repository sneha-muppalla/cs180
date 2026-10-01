"""Generate numerical verification and supporting report images; run from any directory."""
import ast,json,time
from pathlib import Path
import numpy as np
import cv2
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; O=P/'web_assets'; O.mkdir(exist_ok=True)
tree=ast.parse((P/'fun_with_filters.py').read_text()); ns={'np':np}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)],type_ignores=[]),'functions','exec'),ns)
rng=np.random.default_rng(180); a=rng.random((100,100)); k=rng.random((5,5)); ref=convolve2d(a,k,mode='same'); stats=[]
for name,fn in [('Four loops',ns['conv_4loops']),('Two loops',ns['conv_2loops']),('SciPy',convolve2d)]:
 ts=[]
 for _ in range(3):
  t=time.perf_counter(); out=fn(a,k,mode='same');ts.append(time.perf_counter()-t)
 stats.append([name,float(np.median(ts)*1000),float(np.max(np.abs(out-ref)))])
# Match the stack script's own parameter convention; explicitly specify sigma.
x=np.asarray(Image.open(P/'cameraman.png').convert('L'),dtype=float)/255
G1=cv2.getGaussianKernel(15,0);G=G1@G1.T
blur=convolve2d(x,G,mode='same');dx=np.array([[-1.,1.]]);dy=dx.T
ax=convolve2d(blur,dx,mode='same');ay=convolve2d(blur,dy,mode='same')
# Full support prevents truncation of the composed kernel.
kx=convolve2d(G,dx,mode='full');ky=convolve2d(G,dy,mode='full')
bx=convolve2d(x,kx,mode='same');by=convolve2d(x,ky,mode='same')
sl=np.s_[8:-8,8:-8]
for n,z in [('smooth_magnitude',np.hypot(ax,ay)),('smooth_edges',(np.hypot(ax,ay)>.18)),('full_dog_magnitude',np.hypot(bx,by)),('full_dog_edges',(np.hypot(bx,by)>.18)),('full_dog_x',kx),('full_dog_y',ky)]: plt.imsave(O/(n+'.png'),z,cmap='gray')
metrics={'runtime':stats,'dog_interior_x_error':float(np.max(np.abs(ax[sl]-bx[sl]))),'dog_interior_y_error':float(np.max(np.abs(ay[sl]-by[sl]))),'dog_full_x_error':float(np.max(np.abs(ax-bx)))}
# Regenerate a self-consistent detailed hybrid from the saved aligned inputs.
H=P/'cs180_proj2_hybrid_starter_code'
a=np.asarray(Image.open(H/'2.2_input1_aligned_jack_black.png').convert('RGB'),dtype=float)/255
b=np.asarray(Image.open(H/'2.2_input2_aligned_jack_blacksmile.png').convert('RGB'),dtype=float)/255
assert a.shape==b.shape
# Match the submitted hybrid implementation, including its even support size.
def blur_h(z,s):
 g=cv2.getGaussianKernel(int(np.ceil(6*s)),s);return cv2.filter2D(z,-1,g@g.T)
lo=blur_h(a,8);hi=b-blur_h(b,2);hy=np.clip(lo+hi,0,1)
for name,z in [('jack_low',lo),('jack_high',hi+.5),('jack_hybrid',hy)]:plt.imsave(O/(name+'.png'),np.clip(z,0,1))
for name,z in [('input1',a),('input2',b),('low',lo),('high',hi),('hybrid',hy)]:
 ft=np.log(np.maximum(np.abs(np.fft.fftshift(np.fft.fft2(z.mean(axis=2)))),1e-8));plt.imsave(O/('jack_fft_'+name+'.png'),ft,cmap='gray')
(O/'verification.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))
