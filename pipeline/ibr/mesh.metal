#include <metal_stdlib>
using namespace metal;
struct Vert { packed_float3 p; packed_float2 uv; float cat; };
struct U { float4 x,y,z; float4 params; float4 photo,src,eye; }; // quality, temporal weight, frame tag, movers only
struct F {float4 pos [[position]];float2 uv;float z;float cat [[flat]];float3 world;};
vertex F vert(uint id [[vertex_id]],const device Vert *v [[buffer(0)]],constant U &u [[buffer(1)]]){
 auto a=v[id];float4 p=float4(float3(a.p),a.cat==3?0:1);float z=dot(u.z,p); F f;f.pos=float4(dot(u.x,p),dot(u.y,p),z-.12*(a.cat==3?.001:1.),z);f.z=z;f.uv=a.uv;f.cat=a.cat;f.world=float3(a.p);return f;
}
float cost(F f,constant U &u,texture2d<ushort> labels){
 uint2 ij=uint2(clamp(f.uv,float2(0),float2(.999999))*float2(labels.get_width(),labels.get_height()));ushort k=labels.read(ij).r;
 if(k!=ushort(f.cat) || (k>=8)!=(u.params.w>0.5))discard_fragment();
 float q=u.params.x*exp(-1.8*dot(f.uv-.5,f.uv-.5));float z=k==3?500.:f.z;
 if(k!=3){float facing=dot(normalize(f.world-u.eye.xyz),normalize(f.world-u.src.xyz));float2 dx=dfdx(f.uv)*float2(1448,938),dy=dfdy(f.uv)*float2(1448,938);float det=abs(dx.x*dy.y-dx.y*dy.x);float small=det/max(.001,sqrt(dot(dx,dx)+dot(dy,dy)));float g=smoothstep(k==2?.42:.82,k==2?.73:.97,facing)*smoothstep(k==2?.08:.15,k==2?.19:.38,small);if(k==1){float3 dir=normalize(f.world-u.src.xyz);float3 base=u.eye.xyz-u.src.xyz;float px=77.2*length(base-dir*dot(base,dir))/max(1.,distance(f.world,u.eye.xyz));g*=1.-smoothstep(14.,28.,px);}if(g<.045||f.z<1.)discard_fragment();q*=g;}
 
 return (log(max(z,.12))-.17*log(max(.0001,q)) + 2.2)/10.;
}
struct Win {float4 info [[color(0)]];float depth [[depth(any)]];};
fragment Win first(F f [[stage_in]],constant U &u [[buffer(1)]],texture2d<ushort> labels [[texture(1)]]){float d=cost(f,u,labels);return {float4(d,f.cat,f.cat==3?500.:f.z,u.params.z),d};}
fragment float4 second(F f [[stage_in]],constant U &u [[buffer(1)]],texture2d<float> tex [[texture(0)]],texture2d<ushort> labels [[texture(1)]],texture2d<float> best [[texture(2)]]){
 float d=cost(f,u,labels);float4 win=best.read(uint2(f.pos.xy));if((win.y>=8)!=(f.cat>=8)||(win.y==3)!=(f.cat==3))discard_fragment();float delta=(d-win.x)*10.;if(delta>.075)discard_fragment();
 constexpr sampler sm(coord::normalized,address::clamp_to_edge,filter::linear);float3 rgb=tex.sample(sm,f.uv).rgb;
 float3 x=clamp(rgb*u.photo.xyz,0.,1.);if(f.cat==3){float l=dot(x,float3(.2126,.7152,.0722));x=mix(float3(l*1.012,l,l*.993),x,.27);}
 float w=max(.00001,u.params.y)*pow(u.params.x,2.)*exp(-2.0*dot(f.uv-.5,f.uv-.5))*(1.-smoothstep(.025,.075,delta));return float4(w*x,w);
}
