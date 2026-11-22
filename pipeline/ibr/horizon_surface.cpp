// Resolve static photographic triangles by physical range and stable donor quality; no colour averaging.
#include <algorithm>
#include <cmath>
#include <cstdint>
extern "C" void paint(const float* p,int nr,int nc,const uint8_t* rgb,const float* mask,int sh,int sw,int ih,int iw,float quality,float* zb,uint8_t* out,uint8_t* alpha,int H,int W){
 for(int j=0;j<nr-1;j++)for(int i=0;i<nc-1;i++)for(int half=0;half<2;half++){
  int o=j*nc+i;int ids[3]={o,o+(half?nc+1:1),o+(half?nc:nc+1)};const float *v[3]={p+6*ids[0],p+6*ids[1],p+6*ids[2]};float x[3],y[3];bool ok=true;float lo=1e6,hi=0;
  for(int n=0;n<3;n++){ok&=v[n][2]>0 && v[n][5]>0;x[n]=v[n][0];y[n]=v[n][1];lo=std::min(lo,v[n][5]);hi=std::max(hi,v[n][5]);}if(!ok || hi-lo>8+.18*lo)continue;
  for(int n=1;n<3;n++){if(x[n]-x[0]>W/2)x[n]-=W;else if(x[n]-x[0]<-W/2)x[n]+=W;}
  int l=std::ceil(std::min({x[0],x[1],x[2]})),r=std::floor(std::max({x[0],x[1],x[2]})),t=std::max(0,(int)std::ceil(std::min({y[0],y[1],y[2]}))),b=std::min(H-1,(int)std::floor(std::max({y[0],y[1],y[2]})));if(l>r || t>b || r-l>95 || b-t>95)continue;
  float den=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2]);if(std::abs(den)<.02)continue;
  for(int Y=t;Y<=b;Y++)for(int X=l;X<=r;X++){float a=((y[1]-y[2])*(X-x[2])+(x[2]-x[1])*(Y-y[2]))/den,beta=((y[2]-y[0])*(X-x[2])+(x[0]-x[2])*(Y-y[2]))/den,c=1-a-beta;if(std::min({a,beta,c})<-.001)continue;
  int idx=Y*W+((X%W+W)%W);float z=a*v[0][2]+beta*v[1][2]+c*v[2][2];if(z<44 || z>340)continue;float score=z+quality+.045*(a*v[0][5]+beta*v[1][5]+c*v[2][5]);if(score>=zb[idx])continue;
  float u=a*v[0][3]+beta*v[1][3]+c*v[2][3],vv=a*v[0][4]+beta*v[1][4]+c*v[2][4];int U=std::clamp((int)u,0,sw-2),V=std::clamp((int)vv,0,sh-2);float fx=u-U,fy=vv-V;const float* m=mask+V*sw+U;float al=(m[0]*(1-fx)+m[1]*fx)*(1-fy)+(m[sw]*(1-fx)+m[sw+1]*fx)*fy;if(al<.48)continue;
  u=(u+.5)*iw/sw-.5;vv=(vv+.5)*ih/sh-.5;U=std::clamp((int)u,0,iw-2);V=std::clamp((int)vv,0,ih-2);fx=u-U;fy=vv-V;const uint8_t* tex=rgb+3*(V*iw+U);for(int ch=0;ch<3;ch++)out[3*idx+ch]=(uint8_t)std::clamp((tex[ch]*(1-fx)+tex[ch+3]*fx)*(1-fy)+(tex[ch+3*iw]*(1-fx)+tex[ch+3*iw+3]*fx)*fy+.5f,0.f,255.f);zb[idx]=score;alpha[idx]=255;
  }
 }
}
