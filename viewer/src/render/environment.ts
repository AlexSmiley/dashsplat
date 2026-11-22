// The car's image-based lighting: a prefiltered (PMREM) environment of the scene it is in. The sky dome's own
// shader (the scene's sky panorama graded like the splats, or the procedural overcast) over a road-coloured
// ground, plus the surroundings: the frame on screen (splats or the generated clip) is captured at low resolution
// a few times a second and projected back onto the directions it shows, so the windscreen and the glass roof
// reflect sky and the flanks the roadside, and the reflections move as the car drives. Linear radiance (display
// code values decoded), so a mirror-like surface shows the scene about as bright as it is on screen. A scene's
// environment map (render/sky.ts) is reflected whole, without the procedural floor.

import * as THREE from "three";
import type { Sky } from "./sky";

const CAP_W = 640;
const CAP_H = 400;

export class CarEnvironment {
  private readonly scene = new THREE.Scene();
  private readonly mat: THREE.ShaderMaterial;
  private readonly groundMat = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
  private readonly cube = new THREE.WebGLCubeRenderTarget(128, { type: THREE.HalfFloatType, generateMipmaps: false });
  private readonly cubeCam = new THREE.CubeCamera(0.1, 100, this.cube);
  private readonly pmrem: THREE.PMREMGenerator;
  private target: THREE.WebGLRenderTarget | null = null;
  private readonly cap = new THREE.WebGLRenderTarget(CAP_W, CAP_H, {
    depthBuffer: false,
    type: THREE.UnsignedByteType,
    generateMipmaps: false,
    minFilter: THREE.LinearMipmapLinearFilter,
    magFilter: THREE.LinearFilter,
  });
  private capInit = false;
  /** last build's CPU time (ms), number of captures so far */
  ms = 0;
  captures = 0;

  constructor(private readonly renderer: THREE.WebGLRenderer, private readonly sky: Sky) {
    const u = sky.material.uniforms;
    this.mat = new THREE.ShaderMaterial({
      vertexShader: sky.material.vertexShader,
      fragmentShader: sky.material.fragmentShader,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
      uniforms: {
        // the dome's uniforms (panorama, grade, haze colour, photographed sky) are shared; these are its own
        ...u,
        dither: { value: 0 },
        night: { value: 0 },
        band: { value: 0.08 },
        panoBlend: { value: 1 },
        envFloor: { value: 0.95 },
        cap: { value: this.cap.texture },
        capAmt: { value: 0 },
        capVP: { value: new THREE.Matrix4() },
        capCarFwd: { value: new THREE.Vector2(1, 0) },
        capCamFwd: { value: new THREE.Vector2(1, 0) },
        capHalfFov: { value: 0.6 },
        envLinear: { value: new URLSearchParams(location.search).get("envlin") === "0" ? 0 : 1 },
      },
    });
    this.cap.texture.colorSpace = THREE.NoColorSpace;
    // the cube is rendered y-up; the dome's +z (up) becomes the cube's +y
    const dome = new THREE.Mesh(new THREE.SphereGeometry(50, 64, 32), this.mat);
    dome.rotation.x = -Math.PI / 2;
    // the dome ignores depth: drawn first, the ground over it
    dome.renderOrder = -1;
    this.scene.add(dome);
    const g = new THREE.Mesh(new THREE.CircleGeometry(48, 64), this.groundMat);
    g.rotation.x = -Math.PI / 2;
    g.position.y = -1.2;
    this.scene.add(g);
    this.pmrem = new THREE.PMREMGenerator(renderer);
  }

  get texture(): THREE.Texture | null {
    return this.target?.texture ?? null;
  }

  /** re-render the environment (the cube, then its prefiltered mip chain into the same target) */
  build(road: THREE.Color, panoBlend: number, envFloor: number) {
    const t0 = performance.now();
    this.mat.uniforms.panoBlend.value = this.sky.environment ? 1 : panoBlend;
    this.mat.uniforms.envFloor.value = this.sky.environment ? 0 : envFloor;
    this.groundMat.color.copy(road).convertSRGBToLinear();
    this.cubeCam.update(this.renderer, this.scene);
    this.target = this.pmrem.fromCubemap(this.cube.texture, this.target as THREE.WebGLCubeRenderTarget | null);
    this.ms = performance.now() - t0;
    return this.target.texture;
  }

  /** capture the frame just drawn (call right after rendering it, before the frame ends) */
  capture(cam: THREE.PerspectiveCamera, carYaw: number) {
    const r = this.renderer;
    const gl = r.getContext() as WebGL2RenderingContext;
    if (!this.capInit) {
      r.initRenderTarget(this.cap);
      this.capInit = true;
    }
    const fb = (r.properties.get(this.cap) as { __webglFramebuffer?: WebGLFramebuffer }).__webglFramebuffer;
    const tex = (r.properties.get(this.cap.texture) as { __webglTexture?: WebGLTexture }).__webglTexture;
    if (!fb || !tex) return;
    const c = r.domElement;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, fb);
    gl.blitFramebuffer(0, 0, c.width, c.height, 0, 0, CAP_W, CAP_H, gl.COLOR_BUFFER_BIT, gl.LINEAR);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // the mip chain averages the frame down to what the cube can show (no sparkle between captures)
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
    r.resetState();
    const u = this.mat.uniforms;
    cam.updateMatrixWorld();
    (u.capVP.value as THREE.Matrix4).multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    (u.capCarFwd.value as THREE.Vector2).set(Math.cos(carYaw), Math.sin(carYaw));
    const f = cam.getWorldDirection(new THREE.Vector3());
    (u.capCamFwd.value as THREE.Vector2).set(f.x, f.y).normalize();
    u.capHalfFov.value = Math.atan(Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2) * cam.aspect) * 0.97;
    u.capAmt.value = 1;
    this.captures++;
  }

  dispose() {
    this.target?.dispose();
    this.cube.dispose();
    this.cap.dispose();
    this.pmrem.dispose();
    this.mat.dispose();
  }
}
