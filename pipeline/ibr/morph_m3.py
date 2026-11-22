"""Short-baseline moving-pillar mosaic. Lift the base eye first, solve donor translation separately."""
from .morph_m2 import (View,fields,closest,unit,grad,resize,remap,proj,blur,smooth,rig,seq,track,choose,adjusted,cut,plane,floor,rect_for,cv,np,lru_cache,OW,OH,GW,GH,L,H,XY,savej,RS)
from . import morph_sources as ms
from .m3_sky import OUT,skyview
from scipy.ndimage import distance_transform_edt,gaussian_filter1d

Y,X=XY[...,1],XY[...,0]; DEN=.00001

def instant(c,s):
    ts,ns,S,arc=seq(c);i=int(np.clip(np.searchsorted(arc,s),1,len(arc)-1));return i if abs(arc[i]-s)<abs(arc[i-1]-s) else i-1

def main_views(V,s):
    b=instant('front',s);t=seq('front')[0][b];base=[('front',b)]+[(c,closest(c,t)) for c in ['left_pillar','right_pillar']];S=seq('front')[2][b];rel=V.wr@S[:3,:3];ang=np.rad2deg(np.arctan2(rel[...,0],rel[...,2]));
    # M2's conservative upright chart is kept for the lift. Do not let a later
    # pillar change the slope of the photographed office roof in the base view.
    z=V.gr.copy();w=np.zeros_like(z)
    for c,i in base:
        sheet=V.sheet(c,i);a=np.minimum(cv.distanceTransform((sheet>=0).astype('B'),cv.DIST_L2,3)/3,1)*(1-w);z+=a*(np.maximum(sheet,V.gr)-V.gr);w+=a
    along=V.wr[...,:2]@(V.T[:2,2]/np.linalg.norm(V.T[:2,2]));u=z/np.maximum(.2,along);above=(z>V.gr+.002)&(w>.5)&(V.gr<.048)
    def fit(m,lo,hi,prior,sd):
        den=gaussian_filter1d(m.sum(0).astype('f'),sd);return (gaussian_filter1d(np.where(m,np.clip(u,lo,hi),0).sum(0),sd)+4*prior)/(den+4),den
    far,_=fit(above&(u<=.04)&(u>1/180),1/180,1/40,1/95,13);near,n=fit(above&(u>.04),1/30,1/17,1/24,18);far=far.clip(1/135,1/85);z=np.maximum(V.gr,along*(far+(near-far)*smooth((n-1)/14)*.16)).astype('f');return base,ang,z

@lru_cache(100)
def shafts(c,i):
    # Pair long nearly vertical measured line segments in the physical photograph.
    # Require near-matte support plus a measured foot below the physical horizon.
    S,A,im,*_=fields(c,i);w,h=480,310;g=cv.cvtColor(resize(im,w,h),cv.COLOR_BGR2GRAY);ls=cv.createLineSegmentDetector().detect(g)[0];out=[]
    if ls is None:return out
    ls=ls.reshape(-1,4);ls=ls[(abs(ls[:,3]-ls[:,1])>48)&(abs(ls[:,2]-ls[:,0])<.115*abs(ls[:,3]-ls[:,1]))];done=set();
    for a in ls:
        slope=(a[2]-a[0])/(a[3]-a[1]);xmid=a[0]+slope*(h/2-a[1]);grp=[b for b in ls if abs(b[0]+(b[2]-b[0])/(b[3]-b[1])*(h/2-b[1])-xmid)<10];
        if len(grp)<2:continue
        e=np.vstack([np.array(b).reshape(2,2) for b in grp]);top=e[:,1].min();bot=e[:,1].max();mx=np.median(e[:,0]);sid=round(mx/10)
        if sid in done or bot<h*.53 or top>h*.35 or bot-top<.38*h:continue
        y=np.arange(max(0,int(top)),min(h,int(bot)+1));xc=np.interp(y,[top,bot],[np.median(e[e[:,1]<top+32,0]),np.median(e[e[:,1]>bot-30,0])]);m=resize(A[...,2],w,h)[y,np.clip(xc.round().astype(int),0,w-1)];
        if (m>.2).mean()<.22:continue
        # Search the two opposite edges either side of the shaft at each source row.
        gx=cv.Sobel(g,cv.CV_32F,1,0);mk=np.zeros((h,w),'B');cent=int(round(mx));xa=np.clip(xc.astype(int)[:,None]+np.arange(-9,10),0,w-1);edge=gx[y[:,None],xa];left=np.argmax(abs(edge[:,:9]),axis=1);right=10+np.argmax(abs(edge[:,10:]),axis=1);xl=xa[np.arange(len(y)),left];xr=xa[np.arange(len(y)),right];xl=gaussian_filter1d(xl.astype('f'),2).round().astype(int);xr=gaussian_filter1d(xr.astype('f'),2).round().astype(int)
        for yy,l,r in zip(y,xl,xr):mk[yy,max(0,l-1):min(w,r+2)]=255
        foot=np.array([(xc[-1]+.5)*rig[c].width/w,(bot+3.5)*rig[c].height/h]);out.append((mk,foot));done.add(sid)
    return out

def sample_fields(q):
    S,A,img,rows,rays,v,m=fields(q['c'],q['i']);uv=(q['uv']+.5)*[GW/img.shape[1],GH/img.shape[0]]-.5;return remap(A,uv)

def native_mask(m,q):
    uv=(q['uv']+.5)*[m.shape[1]/q['img'].shape[1],m.shape[0]/q['img'].shape[0]]-.5;return remap(m,uv).astype('f')/255

def poles(V,primary):
    extra=[]
    for q in primary:
        c,i=q['c'],q['i'];S=seq(c)[2][i];
        for mask,foot in shafts(c,i):
            d=S[:3,:3]@rig[c].unproject(foot);dist=-(S[:3,3]@V.normal+V.offset)/min(-.00001,d@V.normal)
            if not 4<dist<32:continue
            P=S[:3,3]+dist*d;normal=S[:3,2].copy();normal[2]=0;den=(P-V.T[:3,3])@normal
            if den<1:continue
            iv=(V.wr@normal/den).astype('f');p=V.project(c,i,iv);m=native_mask(mask,p);old=native_mask(mask,q);p['ok'] &= (m>.43)&(iv>0)
            if p['ok'].sum()<6:continue
            q['ok'] &= old<.18;p['role']=q['role'];p['shaft']=True;p['alpha_native']=(mask,foot);extra.append(p)
    return extra

def chart(V,base,S0,c,i,Z,d,rows=False):
    # The common vertical lift in M2 is conservative. Translation ALONG the street
    # uses the measured depth in the source base ray, independently of that lift.
    S,A,img,R,_,valid,_=fields(c,i);ray=V.wr+(V.T[:3,3]-S0[:3,3])*Z[...,None];v=(ray+(S0[:3,3]-S[:3,3])*d[...,None])@S[:3,:3];p=proj(c,v,GW,GH)
    if rows:
        for _ in range(2):
            y=p[...,1].clip(0,GH-1);k=y.astype('i');a=(y-k)[...,None];l=np.minimum(k+1,GH-1)
            def rv(j):return np.einsum('hwi,hwij->hwj',ray+(S0[:3,3]-R[j,:3,3])*d[...,None],R[j,:3,:3])
            u=rv(k);v=u+(rv(l)-u)*a;p=proj(c,v,GW,GH)
    border=np.minimum.reduce([p[...,0],GW-1-p[...,0],p[...,1],GH-1-p[...,1]]);sample=remap(A,p);ok=(border>.62)&(v[...,2]>.08)&(remap(valid.astype('f'),p)>.92)&(sample[...,3]<.10);uv=((p+.5)*[img.shape[1]/GW,img.shape[0]/GH]-.5).astype('f');col=remap(img,uv);return dict(c=c,i=i,n=int(seq(c)[1][i]),uv=uv,img=img,col=col,border=border,ok=ok,sample=sample,zi=d/np.maximum(.1,v[...,2]))

def feature(col):
    l=cv.cvtColor(col,cv.COLOR_BGR2GRAY).astype('f');return (l-blur(l,2))/(5+blur(abs(l-blur(l,2)),3))

def donors(V,s,base,ang,z,primary):
    plane=(z==V.gr);missing=~np.logical_or.reduce([q['ok'] for q in primary]);bridge=cv.dilate(missing.astype('B'),np.ones((7,9),'B')).astype(bool);views=[];ids={tuple(p) for p in base}
    for dist in [2.5,6.,11.,17.,24.]:
        for k in ([1,2,0] if dist<12 else [1,2]):
            c=base[k][0];d=dist if k else -dist;i=instant(c,max(0,s+d));
            if (c,i) in ids:continue
            S0=seq(c)[2][base[k][1]];side=((ang<0) if k==1 else (ang>0)) if k else np.ones_like(bridge);mask=bridge&side;zz=np.maximum(V.gr,.025).astype('f');pp=chart(V,base,S0,c,i,z,zz)
            if (mask&pp['ok']).sum()<20:continue
            ids.add((c,i));views.append((k,i,d,mask,S0))
    if not views:return [],z
    # Per-ray plane sweep: depth prior from both source surfaces + high-pass
    # agreement between physically separated exposures and the fixed base photos.
    costs=[];ds=np.array([.004,.008,.013,.019,.028,.041,.058,.082,.12],'f');pfeat=[feature(q['col']) for q in primary]
    for level in ds:
        dsamp=np.where(plane,V.gr,level).astype('f');tot=np.zeros_like(z);wt=np.zeros_like(z);old={}
        for k,i,dist,zone,S0 in views:
            c=base[k][0];q=chart(V,base,S0,c,i,z,dsamp);sm=q['sample'];ok=q['ok']&zone;feat=feature(q['col']);dep=np.where(sm[...,2]>.30,sm[...,1],sm[...,0]);geom=abs(np.log((q['zi']+.003)/(dep+.003))).clip(0,1.8);mw=ok*(.5+.5*(sm[...,5]<.7));tot+=mw*geom*.65;wt+=mw
            for f,ok2 in [*zip(pfeat,[p['ok'] for p in primary]),*([old[k]] if k in old else [])]:
                mask=ok&ok2;tex=(np.minimum(abs(f)+abs(feat),2)*mask).astype('f');tot+=1.8*tex*np.minimum(abs(feat-f),2);wt+=1.8*tex
            old[k]=(feat,ok)
        ww=blur(wt,2);costs.append(blur(tot,2)/np.maximum(.08,ww)+.18/np.maximum(.16,ww)+.07*abs(np.log(level/.025)))
    ix=np.argmin(costs,axis=0);depth=ds[ix];valid=bridge&~plane
    den=blur(valid.astype('f'),3.5);depth=blur(depth*valid,3.5)/np.maximum(.02,den);depth=np.where(den>.04,depth,.022);# DO NOT feed a noisy winner-take-all depth image to a translating camera.
    # Fit one upright plane per side; the measured pavement is the other surface.
    # With straight poles, a rigid projection matters more than matching leaf disparity.
    zside=np.zeros_like(z)
    for sign in [-1,1]:
        m=(ang*sign>0)&bridge&(~plane)&(V.wr[...,2]<.04)&(V.wr[...,2]>-.14);vals=depth[m];val=float(np.median(vals)) if len(vals)>20 else .022
        zside[ang*sign>=0]=np.clip(val,.012,.042)
    depth=np.maximum(V.gr,zside).astype('f');qs=[]
    for k,i,dist,zone,S0 in views:
        q=chart(V,base,S0,base[k][0],i,z,depth,True);q['ok']&=zone;q['role']=10+3*[2.5,6.,11.,17.,24.].index(abs(dist))+k;q['distance']=dist;q['cameraIndex']=k;qs.append(q)
    return qs,depth

# Two-dimensional affine radiometric gauge, fitted robustly on paired smooth pixels.
def tone(q,refs,prior,old):
    weights=[];A=[];B=[];uv=q['uv'];basis=np.dstack([np.ones_like(X),uv[...,0]/q['img'].shape[1]-.5,uv[...,1]/q['img'].shape[0]-.5]);
    for col,ok in refs:
        m=q['ok']&ok&(grad(col)<11)&(grad(q['col'])<11)&(col.min(2)>16)&(q['col'].min(2)>16);n=m.sum()
        if n<18:continue
        jump=max(1,n//400);A.append(basis[m][::jump]);B.append(np.log((col[m][::jump].astype('f')+2)/(q['col'][m][::jump].astype('f')+2)));weights.append(np.full(len(A[-1]),min(1,n/180)))
    coef=prior.copy()
    if A:
        x=np.vstack(A);y=np.vstack(B);ww=np.concatenate(weights);h=ww.copy();diag=np.diag([80,130,110]);
        for _ in range(2):
            coef=np.linalg.solve((x.T*h)@x+diag,(x.T*h)@y+diag@prior);r=np.linalg.norm(y-x@coef,axis=1);h=ww*np.minimum(1,.13/np.maximum(.005,r))
    coef=np.clip(coef,-.65,.65);coef=.78*old+.22*coef if old is not None else coef;q['coef']=coef;q['gain']=np.exp(basis@coef).astype('f');q['col']=adjusted(q['col'],q['gain']);return coef

def pathcut(cost):
    # Top-to-bottom dynamic-programming path with bounded lateral motion per row.
    h,w=cost.shape;back=np.empty((h,w),'i2');a=cost[0].copy();idx=np.arange(w);moves=np.arange(-4,5)[:,None];ix=(idx+moves).clip(0,w-1)
    for y in range(1,h):
        e=a[ix]+.5*abs(moves);k=np.argmin(e,axis=0);back[y]=ix[k,idx];a=e[k,idx]+cost[y]
    path=np.empty(h,'i2');path[-1]=np.argmin(a)
    for y in range(h-2,-1,-1):path[y]=back[y+1,path[y+1]]
    return path

def add_side(lab,col,ok,q,j,left,previous):
    legal=q['ok'];only=legal&~ok;other=ok&~legal
    cum=lambda m:np.cumsum(m,axis=1);rev=lambda m:np.cumsum(m[:,::-1],axis=1)[:,::-1]
    # Prefer the base/instantaneous photo except in its narrow optical collar.
    safe=legal&(q['border']>2.5);missPrimary=(rev(safe) if left else cum(safe));
    loss=(cum(other)+rev(only)) if left else (cum(only)+rev(other));halo=cv.dilate(np.maximum(grad(q['col']),grad(col)),np.ones((5,5),'B'));agree=np.mean(abs(q['col'].astype('f')-col),axis=2);over=legal&ok
    cost=14*np.minimum(loss,120)+1.7*np.minimum(missPrimary,100)+np.where(over,2+np.minimum(halo,60)*.8+np.minimum(55,agree)*.55,22);cost+=np.where(over,2/(.3+q['border'].clip(0,10)),0)
    if previous is not None:cost+=.65*np.minimum(abs(X-previous[:,None]),45)
    path=pathcut(cost);use=legal&(~ok|((X<path[:,None]) if left else (X>=path[:,None])));lab[use]=j;col[use]=q['col'][use];ok|=use;return path

def seam_bias(labels,qs,lowref,sky,ground):
    # Remove the low-frequency component of a source switch, estimated at the SAME
    # locations in the two standalone photos. The correction cannot move an edge.
    colors={0:sky,80:ground,81:sky}
    for j,q in enumerate(qs,1):
        a=smooth((q['sample'][...,5]-.2)/.7)[...,None];colors[j]=blur((q['col']*(1-a)+sky*a).astype('f'),3)
    div=np.zeros((H,L,3),'f');
    for axis in (0,1):
        a,b=(np.s_[:-1,:],np.s_[1:,:]) if axis==0 else (np.s_[:,:-1],np.s_[:,1:]);la,lb=labels[a],labels[b];y,x=np.where((la!=lb)&(la>0)&(lb>0));ja=la[y,x];jb=lb[y,x];d=np.zeros((len(y),3),'f')
        for j in np.unique(np.r_[ja,jb]):
            m=ja==j;d[m]+=colors[j][y[m],x[m]];m=jb==j;d[m]-=colors[j][y[m],x[m]]
        d=np.clip(d,-28,28)*.84;div[y,x]-=d;div[y+(axis==0),x+(axis==1)]+=d
    ev=(.014+2-2*np.cos(np.pi*Y/H)+2-2*np.cos(np.pi*X/L)).astype('f');return np.dstack([cv.idct(cv.dct(div[...,c])/ev) for c in range(3)]).clip(-20,20)

def card(q,sl):
    uv=resize(q['uv'],OW,OH)[sl];rgb=remap(q['img'],uv);return adjusted(rgb,resize(q['gain'],OW,OH)[sl]) if 'coef' in q or ('gain' in q and q['gain'].ndim==3) else rgb

def frame(V,s,prev=None):
    base,ang,z=main_views(V,s);qs=[V.project(*p,z) for p in base];
    for j,q in enumerate(qs):q.update(role=j+1,sample=sample_fields(q),distance=0,cameraIndex=j)
    shaft=poles(V,qs)
    for q in qs:
        rot=V.project(q['c'],q['i'],np.zeros_like(z));sf=sample_fields(rot);extend=(~q['ok'])&rot['ok']&(sf[...,5]>.88)&(V.wr[...,2]>.025)
        for key in ['uv','col','border','sample']:
            v=sf if key=='sample' else rot[key];q[key][extend]=v[extend]
        q['ok']|=extend
    extra,true=donors(V,s,base,ang,z,qs);primary=qs.copy();qs+=extra;N=len(qs);bg,ga=floor(V.t,V.T,V.wr,V.gr);gnd=resize(bg,L,H);sky=skyview(V);plane=z==V.gr;refs=[(primary[0]['col'],primary[0]['ok'])];prev=prev or {};oldCoef=prev.get('coef',{});coef={};
    # Sky anchors supplement geometric overlaps at disconnected footprints. Applied
    # in native UV, so exposure correction is low-order rather than a rectangular grade.
    for j in list(range(3,N))+[1,2]:
        q=qs[j];rr=list(refs)+[(gnd,(ga>.88)&plane&(q['sample'][...,4]>.65)),(sky,(q['sample'][...,5]>.91)&(V.wr[...,2]>.045))];src=q['c'];prior=np.zeros((3,3));_,A,img,*_=fields(src,q['i']);m=(A[...,5]>.94)&(A[...,6]>.9);_,fa,fi,*_=fields(*base[0]);fm=(fa[...,5]>.94)&(fa[...,6]>.9)
        if src!='front' and m.sum()>80 and fm.sum()>80:prior[0]=.65*np.log((np.median(resize(fi,GW,GH)[fm],0)+2)/(np.median(resize(img,GW,GH)[m],0)+2));prior[2]=-prior[0]*.8
        coef[q['role']]=tone(q,rr,prior,oldCoef.get(q['role']));refs.append((q['col'],q['ok']))
    prevRole=None;histories=prev.get('hist',[]);rr=None
    if histories:
        Told=histories[-1]['T'];v=(V.wr+(V.T[:3,3]-Told[:3,3])*np.where(plane,V.gr,true)[...,None])@Told[:3,:3];rr=(v[...,:2]/np.maximum(.01,v[...,2:])*(V.f/5)+[L/2-.5,H/2-.5]).astype('f');prevRole=cv.remap(histories[-1]['role'],rr,None,cv.INTER_NEAREST,borderMode=cv.BORDER_CONSTANT)
    # Donors compete only over the gap plus a collar, biased to smallest acquisition
    # displacement. Graph edge costs keep a connected surface on one donor.
    low=np.zeros((H,L),'B');col=np.zeros((H,L,3),'B');ok=np.zeros((H,L),bool);best=np.full((H,L),-8.,'f');
    for j in sorted(range(3,N),key=lambda k:-abs(qs[k]['distance'])):
        q=qs[j];legal=q['ok'];sm=q['sample'];dobs=np.where(sm[...,2]>.3,sm[...,1],sm[...,0]);err=abs(np.log((q['zi']+.004)/(dobs+.004))).clip(0,1.8);score=2.4-.075*abs(q['distance'])+.5*np.log(np.clip(q['border']/5,.12,1))-.68*err
        if prevRole is not None:score+=.6*(prevRole==q['role'])
        dif=abs(col.astype('f')-q['col']).mean(2);bar=3+30*np.minimum(cv.dilate(np.maximum(grad(col),grad(q['col'])),np.ones((5,5),'B'))/15,2)**2+np.minimum(40,dif);use=cut(score-best,bar,legal,ok);low[use]=j+1;col[use]=q['col'][use];ok|=use;best[use]=score[use]
    paths={};old=prev.get('paths',{})
    for j,left in [(1,True),(2,False)]:paths[j]=add_side(low,col,ok,qs[j],j+1,left,old.get(j))
    # Front joins occur on opposite sides of its optical axis, including the turn
    # when one join is completely outside the viewport.
    for j,left in [(3,False),(4,True)]:
        q=dict(qs[0]);q['ok']=qs[0]['ok']&((ang<=0) if j==3 else (ang>0));paths[j]=add_side(low,col,ok,q,1,left,old.get(j))
    for q in shaft:
        parent=primary[q['role']-1];q['sample']=sample_fields(q);q['distance']=0;q['cameraIndex']=q['role']-1
        if 'coef' in parent:
            uv=q['uv'];basis=np.dstack([np.ones_like(X),uv[...,0]/q['img'].shape[1]-.5,uv[...,1]/q['img'].shape[0]-.5]);q['gain']=np.exp(basis@parent['coef']).astype('f');q['col']=adjusted(q['col'],q['gain'])
        qs.append(q);low[q['ok']]=len(qs);col[q['ok']]=q['col'][q['ok']];ok|=q['ok']
    # Global sky at infinity. A single observed source retains its high-frequency
    # wire/support texture; the common cloud field replaces the low-frequency sky.
    sp=np.zeros((H,L),'f');roles=np.zeros_like(low)
    for j,q in enumerate(qs,1):
        use=low==j;sp[use]=q['sample'][...,5][use];roles[use]=q['role'];
    if ok.any():
        _,ix=distance_transform_edt(~ok,return_indices=True);height=sp[ix[0],ix[1]];missingSky=(~ok)&((V.wr[...,2]>.10)|((height>.60)&(V.wr[...,2]>-.04)))
    else:missingSky=V.wr[...,2]>0
    # IDs 80–83 are non-photo fallback classes; actual counts are measured after
    # rasterization. Sky mask avoids painting smooth sky over a roof or pole.
    low[~ok&plane&(ga>.55)]=80;low[missingSky&(low==0)]=81
    # Narrow sub-cell boundaries, without averaging differently registered outlines.
    hard=np.zeros((OH,OW),'B');score=np.full((OH,OW),-1,'f')
    for j in np.unique(low):
        a=resize(blur((low==j).astype('f'),.65),OW,OH);win=a>score;hard[win]=j;score[win]=a[win]
    out=np.zeros((OH,OW,3),'B');patches=[];skyHi=resize(sky,OW,OH);bias=resize(seam_bias(low,qs,col,sky,gnd),OW,OH)
    for j in np.unique(hard):
        if not j:continue
        x,y,w,h=rc=rect_for(hard,j);sl=np.s_[y:y+h,x:x+w];m=hard[sl]==j
        if j==80:rgb=bg[sl];ref=rgb
        elif j==81:rgb=skyHi[sl];ref=rgb
        else:
            q=qs[j-1];ref=card(q,sl);a=resize(smooth((q['sample'][...,5]-.20)/.7),OW,OH)[sl][...,None];cloud=skyHi[sl].astype('f');delta=ref.astype('f')-blur(ref.astype('f'),8);rgb=np.uint8(np.clip(ref*(1-a)+(cloud+delta)*a+.5,0,255));
        rgb=np.uint8(np.clip(rgb.astype('f')+bias[sl]+.5,0,255));ref=np.uint8(np.clip(ref.astype('f')+bias[sl]+.5,0,255));out[sl][m]=rgb[m];patches.append((int(j),rc,ref))
    # Reproject observed pixels in previous *output* frames, rejecting mismatched
    # ranges, mixed boundaries and synthesized history. Propagate their source refs.
    obs=(hard>0)&(hard<80);hist=dict(t=V.t,T=V.T.copy(),role=roles,depth=np.where(plane,V.gr,true),rgb=out,low=low.copy(),patches=patches.copy());holes=hard==0;temporal=0;histIDs=90
    for hst in reversed(histories):
        if not holes.any():break
        d=np.where(plane,V.gr,true);v=(V.wr+(V.T[:3,3]-hst['T'][:3,3])*d[...,None])@hst['T'][:3,:3];uv=(v[...,:2]/np.maximum(.01,v[...,2:])*(V.f/5)+[L/2-.5,H/2-.5]).astype('f');oldDepth=remap(hst['depth'].astype('f'),uv);good=(abs(oldDepth-d/np.maximum(.1,v[...,2]))<.016)&(uv[...,0]>1)&(uv[...,0]<L-2)&(uv[...,1]>1)&(uv[...,1]<H-2);uu=(resize(uv,OW,OH)+.5)*5-.5
        for oj,_,_ in hst['patches']:
            if oj>=80:continue
            legal=cv.erode((hst['low']==oj).astype('B'),np.ones((3,3),'B'));use=holes&(resize((good&(remap(legal.astype('f'),uv)>.99)).astype('f'),OW,OH)>.98)
            if not use.any() or histIDs>=244:continue
            hard[use]=histIDs;xx,yy,ww,hh=rc=rect_for(hard,histIDs);sl=np.s_[yy:yy+hh,xx:xx+ww];out[sl][use[sl]]=remap(hst['rgb'],uu[sl])[use[sl]];_,r,ref=next(p for p in hst['patches'] if p[0]==oj);patches.append((histIDs,rc,remap(ref,uu[sl]-[r[0],r[1]])));temporal+=int(use.sum());holes[use]=False;histIDs+=1
    spatial=int(holes.sum());
    if spatial:
        # Harmonic low-frequency continuation with a short texture skirt. Neither
        # fallback is labelled as an observation; audit against its unfilled underlay.
        b=resize(out,L,H);mm=resize(holes.astype('f'),L,H)>.2;_,ix=distance_transform_edt(mm,return_indices=True);near=b[ix[0],ix[1]].astype('f');u=near.copy()
        for sd in (9,5,2):den=blur((~mm).astype('f'),sd);p=blur(b.astype('f')*(~mm)[...,None],sd)/np.maximum(.002,den[...,None]);a=smooth(den/.16)[...,None];u=u+(p-u)*a
        out[holes]=resize(np.uint8(u.clip(0,255)),OW,OH)[holes];hard[holes]=82;x,y,w,h=rc=rect_for(hard,82);patches.append((82,rc,bg[y:y+h,x:x+w]))
    solo=card(qs[0],np.s_[:,:]);meta=dict(t=V.t,framesize=OW*OH,temporal=temporal,spatial=spatial,skyFallback=int((hard==81).sum()),stitched=int((hard==80).sum()),bridge=int(np.count_nonzero((hard>=4)&(hard<=N))),laterPillar=sum(int((hard==j+1).sum()) for j in range(3,N) if qs[j]['cameraIndex']),pole=int(np.count_nonzero((hard>N)&(hard<80))),sources=[(q['c'],int(q['n']),q['distance']) for q in qs]);hist['rgb']=out.copy();return out,hard,patches,solo,meta,dict(hist=([a for a in histories if round(a['t']*30)%6==0 and a['t']>=V.t-1.6]+[hist])[-10:],coef=coef,paths=paths)
