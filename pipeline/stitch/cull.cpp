#include <cmath>
#include <cstdint>
// Conservative sphere/frustum test. 4.7 sigma includes bounded optimiser growth
// for ground; use 6.7 for the larger permitted object kernels.
extern "C" int cull(const float *points,const double *T,const double *F,const int64_t *starts,const int64_t *lens,int nb,int32_t *ids){
 int n=0;for(int i=0;i<nb;i++)for(int j=starts[i],end=j+lens[i];j<end;j++){
  const float *p=points+4*j; float x=p[0]-T[3],y=p[1]-T[7],z=p[2]-T[11],r=p[3]; float dep=T[2]*x+T[6]*y+T[10]*z;
  if(dep+r<.25||dep-r>85)continue;float u=T[0]*x+T[4]*y+T[8]*z;if(u-dep*F[0]<-r*F[4]||u-dep*F[1]>r*F[5])continue;
  float v=T[1]*x+T[5]*y+T[9]*z;if(v-dep*F[2]<-r*F[6]||v-dep*F[3]>r*F[7])continue;ids[n++]=j;
 }return n;
}
