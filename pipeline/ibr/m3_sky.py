"""One world-direction cloud field. Fit to masked, exposure-corrected real sky; thin wires stay on photo owners."""
from .morph_m2 import fields, rig, seq, RS, np, cv, blur, resize, GW, GH, lru_cache
from scipy.ndimage import distance_transform_edt
OUT=RS/'work/ibr/m3';OUT.mkdir(parents=True,exist_ok=True)
AW,AH=1440,256

def coordinates(r):
    return np.stack([(np.arctan2(r[...,1],r[...,0])/(2*np.pi)+.5)*AW-.5,(.64-np.arctan2(r[...,2],np.hypot(r[...,0],r[...,1])))/.78*AH-.5],-1).astype('f')

def build():
    acc=np.zeros((AH,AW,3),'f');weight=np.zeros((AH,AW),'f')
    for t in np.arange(89,143,1.25):
        refs=None
        for c in ['front','left_pillar','right_pillar']:
            i=int(np.argmin(abs(seq(c)[0]-t)));S,A,img,rows,rays,valid,_=fields(c,i);m=(A[...,5]>.96)&valid&(A[...,3]<.05);m=cv.erode(m.astype('B'),np.ones((3,3),'B')).astype(bool);rgb=resize(img,GW,GH).astype('f');
            if m.sum()<60:continue
            med=np.median(rgb[m],axis=0)
            if c=='front':refs=med
            gain=np.clip(refs/np.maximum(8,med),.55,1.65) if refs is not None else np.ones(3)
            # Remove narrow line silhouettes from the sky field. They are restored only
            # from one actual owner at their finite range, never by duplicating wire images.
            rgb=cv.morphologyEx(rgb,cv.MORPH_CLOSE,np.ones((5,5),'B'));p=coordinates(rays)[m];x=p[:,0].round().astype(int)%AW;y=p[:,1].round().astype(int);ok=(y>=0)&(y<AH);wt=np.full(ok.sum(),3 if c=='front' else 1,'f');np.add.at(weight,(y[ok],x[ok]),wt);np.add.at(acc,(y[ok],x[ok]),rgb[m][ok]*gain*wt[:,None])
    acc=np.pad(acc,((0,0),(45,45),(0,0)),mode='wrap');w=np.pad(weight,((0,0),(45,45)),mode='wrap');a=blur(acc,3.5);w=blur(w,3.5);ok=w>.04;_,nn=distance_transform_edt(~ok,return_indices=True);im=(a/np.maximum(w,.01)[...,None])[nn[0],nn[1]];img=np.uint8(blur(im,1.5)[:,45:-45].clip(0,255));cv.imwrite(str(OUT/'sky.png'),img)

@lru_cache(None)
def atlas():
    im=cv.imread(str(OUT/'sky.png'));assert im is not None;return im

def skyview(V):return cv.remap(atlas(),coordinates(V.wr),None,cv.INTER_LINEAR,borderMode=cv.BORDER_REPLICATE)
if __name__=='__main__':build()
