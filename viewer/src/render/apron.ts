// A soft ground under and beyond the loaded tiles: a large disc following the car in the graded road colour,
// fading into the haze exactly like the splats do, drawn before the splats and without depth writes. Holes
// and ragged edges of the map then read as unpaved haze instead of sky showing through the floor.

import * as THREE from "three";
import { MOVER_SHADOW_GLSL, type SplatGlobals } from "./splatShader";

const vert = /* glsl */ `
varying vec2 vXY;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  vXY = w.xy;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const frag = /* glsl */ `
uniform sampler2D moverTex;
uniform sampler2D contactAo;
uniform vec4 moverInfo;
uniform float groundZ;
uniform vec3 road;
uniform vec3 fogColor;
uniform vec2 centre;
uniform vec3 haze; // start, end, radius
uniform float dither;
varying vec2 vXY;
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
${MOVER_SHADOW_GLSL}
void main() {
  float d = length(vXY - centre);
  float hz = smoothstep(haze.x, haze.y, d);
  vec3 r = road;
  // movers' contact shadows where there is no ground under them but the apron
  if (moverInfo.x > 0.5) r *= rsMoverShadow(moverTex, contactAo, moverInfo.x, vec3(vXY, groundZ));
  vec3 c = mix(r, fogColor, hz * (2.0 - hz));
  float a = 1.0 - smoothstep(haze.z * 0.9, haze.z, d);
  c += (hash(gl_FragCoord.xy) - 0.5) * dither / 255.0;
  gl_FragColor = vec4(c, a);
}`;

export class Apron {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  constructor(G: SplatGlobals) {
    this.material = new THREE.ShaderMaterial({
      vertexShader: vert,
      fragmentShader: frag,
      uniforms: {
        road: { value: new THREE.Color(0.42, 0.43, 0.43) },
        fogColor: { value: new THREE.Color(0.8, 0.82, 0.84) },
        centre: { value: new THREE.Vector2() },
        haze: { value: new THREE.Vector3(65, 140, 220) },
        dither: { value: 1.5 },
        moverTex: { value: G.moverTexture },
        contactAo: { value: G.contactAo.value },
        moverInfo: { value: G.moverInfo.value },
        groundZ: { value: 0 },
      },
      transparent: true,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(new THREE.CircleGeometry(1, 96), this.material);
    this.mesh.renderOrder = -100;
    this.mesh.frustumCulled = false;
    this.mesh.name = "ground-apron";
  }

  update(x: number, y: number, z: number, road: THREE.Color, fog: THREE.Color, start: number, end: number, radius: number) {
    this.mesh.position.set(x, y, z - 0.08);
    this.mesh.scale.setScalar(radius);
    const u = this.material.uniforms;
    (u.road.value as THREE.Color).copy(road);
    (u.fogColor.value as THREE.Color).copy(fog);
    (u.centre.value as THREE.Vector2).set(x, y);
    (u.haze.value as THREE.Vector3).set(start, end, radius);
    // movers on the apron sit about on the car's road level
    u.groundZ.value = z;
  }
}
