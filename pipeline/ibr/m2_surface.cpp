// Perspective-correct inverse-z interpolation on a calibrated photographic relief sheet.
// Input vertices already have target (x,y,1/z) and lens/mover validity. Back folds do not own pixels.
#include <algorithm>
#include <cmath>
extern "C" void surface(const float *v,int nx,int ny,int w,int h,float *z) {
  auto tri=[&](int i,int j,int k){
    const float *a=v+4*i,*b=v+4*j,*c=v+4*k;
    if(a[3]<=0||b[3]<=0||c[3]<=0) return;
    float ux=b[0]-a[0],uy=b[1]-a[1],vx=c[0]-a[0],vy=c[1]-a[1],d=ux*vy-uy*vx;
    if(d<1e-5f) return;
    int l=std::max(0,(int)std::ceil(std::min({a[0],b[0],c[0]}))),r=std::min(w-1,(int)std::floor(std::max({a[0],b[0],c[0]})));
    int t=std::max(0,(int)std::ceil(std::min({a[1],b[1],c[1]}))),e=std::min(h-1,(int)std::floor(std::max({a[1],b[1],c[1]})));
    for(int y=t;y<=e;y++) for(int x=l;x<=r;x++) {
      float m=((x-a[0])*vy-(y-a[1])*vx)/d,n=(ux*(y-a[1])-uy*(x-a[0]))/d;
      if(m>=-.00001f&&n>=-.00001f&&m+n<=1.00001f) z[y*w+x]=std::max(z[y*w+x],a[2]+m*(b[2]-a[2])+n*(c[2]-a[2]));
    }
  };
  for(int y=0;y<ny-1;y++) for(int x=0;x<nx-1;x++){int i=y*nx+x;tri(i,i+1,i+nx);tri(i+1,i+nx+1,i+nx);}
}
