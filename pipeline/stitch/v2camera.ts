// Emit the official chase-track pose on the default frame, and the same pure function with orbit offsets.
import * as THREE from '../../viewer/node_modules/three/build/three.module.js';
import { CHASE, chaseInputs, placeChase } from '../../viewer/src/camera/chasePose';
import { Drive } from '../../viewer/src/scene/drive';
const [file,trackFile,out]=Bun.argv.slice(2);
if (!file||!trackFile||!out) throw new Error('bun v2camera.ts DRIVE_JSON CHASE_TRACK_JSON OUTPUT_JSON');
const bytes=await Bun.file(file).arrayBuffer(),trackBytes=await Bun.file(trackFile).arrayBuffer(),drive=new Drive(JSON.parse(new TextDecoder().decode(bytes))),track=JSON.parse(new TextDecoder().decode(trackBytes));
const hash=(v:ArrayBuffer)=>new Bun.CryptoHasher('sha256').update(v).digest('hex');
const w=track.width,h=track.height;if(w!==1440||h!==900)throw new Error('expected 1440×900');
const cam=new THREE.PerspectiveCamera(track.vfov,w/h,track.near,5000);
const result:any={width:w,height:h,controller:CHASE,source:'viewer/tools/chase-track.ts and viewer/src/camera/chasePose.ts',track_sha256:hash(trackBytes),drive_sha256:hash(bytes),controller_sha256:hash(await Bun.file(new URL('../../viewer/src/camera/chasePose.ts',import.meta.url)).arrayBuffer()),sites:{}};
for(const t of (Bun.argv[5]?.split(",").map(Number) ?? [93,101,112,128])){
 const p=chaseInputs(drive,t),ref=track.frames.find((v:any)=>v.t===t);if(!ref)throw new Error('track lacks exact timestamp');const frames:Record<string,unknown>={};let pivot=new THREE.Vector3();
 for(const [key,az,el] of [...Array.from({length:96},(_,j)=>[String(j),j*360/96,CHASE.elev*180/Math.PI] as const),['raised',0,40] as const]){
  pivot=placeChase(cam,p,w/h,az*Math.PI/180,el*Math.PI/180-CHASE.elev);
  if(key==='0'){
   const pos=new THREE.Vector3().fromArray(ref.pos),q=new THREE.Quaternion().fromArray(ref.quat).normalize();
   if(cam.position.distanceTo(pos)>.00009||cam.quaternion.angleTo(q)>.000003||Math.abs(cam.fov-track.vfov)>.00009)throw new Error('pose no longer agrees with chase-track');
   cam.position.copy(pos);cam.quaternion.copy(q);cam.updateMatrixWorld();
  }
  const m=cam.matrixWorld.elements,proj=cam.projectionMatrix.elements;
  const T_world_cam=Array.from({length:4},(_,i)=>[m[i],-m[4+i],-m[8+i],m[12+i]]);
  frames[key]={az,elev:el,K:[proj[0]*w/2,proj[5]*h/2,w/2,h/2],T_world_cam};
 }
 result.sites[t]={speed:p.speed,heading:p.heading,position:[p.x,p.y,p.z],pivot:pivot.toArray(),frames};
}
await Bun.write(out,JSON.stringify(result));
