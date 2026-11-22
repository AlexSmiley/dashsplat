import "./style.css";
import * as THREE from "three";
import { SparkRenderer, isMobile } from "@sparkjsdev/spark";
import { Drive, loadDrive, type DriveState } from "./scene/drive";
import { cameraGeometry, CAMERA_LABELS, defaultRig, type CamGeom, type RigJson } from "./scene/rig";
import { TileManager, type MapTile, type MapJson, type RsbIndex } from "./scene/tiles";
import { ObjectManager, type ObjectsJson } from "./scene/objects";
import { SplatGlobals, gradeColour, type ViewMode } from "./render/splatShader";
import { Apron } from "./render/apron";
import { PathIndex, RoadColour, fromHalf, type WhiteBalance } from "./scene/whiteBalance";
import { Sky } from "./render/sky";
import { Car } from "./render/car";
import { PathViz, findTurns, speedLimits } from "./render/path";
import { Frustums } from "./render/frustums";
import { Generative, type GeneratedJson, type Overlay } from "./render/generative";
import { PhotoLayers, type PhotoSwitches } from "./render/photo";
import { GroundAtlas } from "./render/groundAtlas";
import { CarEnvironment } from "./render/environment";
import { VEHICLE_OCCLUSION, VehicleKit } from "./render/vehicles";
import { applyPath, parsePath, type CamPath } from "./camera/paths";
import { Director } from "./render/director";
import { CameraController, type CamMode } from "./camera/controller";
import { chaseInputs, placeChase } from "./camera/chasePose";
import { Hud, type SceneEntry, type StreetEntry } from "./ui/hud";

const params = new URLSearchParams(location.search);
const MOBILE = isMobile();
const DEFAULT_SCENE = "/scenes/dev/";
const RATES = [0.5, 1, 2, 4];
/** over the photographed backdrop, non-ground splats fade out between these distances from the car (?objFade=) */
const OBJ_FADE = (params.get("objFade") ?? "38,72").split(",").map(Number);
/** optimised scenes' haze: start, end (m), strength, sky horizon band (radians) (?far= for tuning) */
const FAR_OPT = (params.get("far") ?? "80,240,0.6,0.06").split(",").map(Number);
/** the Natural grade, one for the splats and the generated clips alike: the full white balance measured on the
 *  road (dry asphalt is neutral grey; the clip and the splats share the camera's colours, and their asphalt
 *  needs the same gains, 1.16 / 0.94 / 0.94), exposure 1.02, a gentle S-curve (0.28), saturation 1.15 in the
 *  midtones, no black point. ?wbk= (share of the white balance) and ?g2=exposure,contrast,saturation,black */
const WB_STRENGTH = Number(params.get("wbk") ?? 1);
const GRADE2 = (params.get("g2") ?? "1.02,0.28,1.15,0").split(",").map(Number);
/** the procedural cloud deck over the sky (?clouds=0 for the plain sky) */
const CLOUDS = Number(params.get("clouds") ?? 1);
/** leave out ground splats under the opaque textured ground: phones (?groundFilter=1 / 0 to force) */
const GROUND_FILTER = params.has("groundFilter") ? params.get("groundFilter") === "1" : isMobile();

function sceneUrl(s: string | null) {
  if (!s) return DEFAULT_SCENE;
  if (!s.includes("/")) return `/scenes/${s}/`;
  return s.endsWith("/") ? s : s + "/";
}

async function fetchJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

class App {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly spark: SparkRenderer;
  readonly G = new SplatGlobals();
  readonly sky = new Sky();
  readonly car = new Car("/models/model3.glb");
  readonly ctl: CameraController;
  readonly hud: Hud;
  // overcast: the car is lit by its environment (sky, surroundings, road); a faint fill only at night
  private hemi = new THREE.HemisphereLight(0xe8eef3, 0x55595c, 0);
  private env!: CarEnvironment;
  private lastCapture = 0;
  private envPending = 0;
  readonly apron: Apron;
  private colourMode: "natural" | "camera" = params.get("colour") === "camera" ? "camera" : "natural";
  private gradeAmt = 1;
  private wb: WhiteBalance | null = null;
  private roadSampler: RoadColour | null = null;
  private envKey = "";
  /** where the horizon and the car sit in the frame (fractions of the view), for checking the chase pose */
  private frameInfo: { horizon: number; carY: number; carW: number; camH: number; pitch: number } | null = null;
  private sceneKey = "";
  private drive?: Drive;
  private tiles?: TileManager;
  private objects?: ObjectManager;
  private gen?: Generative;
  /** the textured ground (ground/ground.json), and whether it is shown (?ground=0 or the menu for the splat ground) */
  private atlas?: GroundAtlas;
  private groundOn = params.get("ground") !== "0";
  private carDz = 0;
  private groundFilterOn = false;
  /** director looks for captured shots (?director=; render/director.ts), null on the normal page */
  private director: Director | null = null;
  private mapTiles: MapTile[] = [];
  /** scratch camera at the chase pose (scripted camera paths) */
  private readonly envCam = new THREE.PerspectiveCamera(50, 1, 0.1, 4000);
  /** ego car and path ribbon shown (?car=0, ?ribbon=0, capture toggles) */
  private showCar = params.get("car") !== "0";
  private showRibbon = params.get("ribbon") !== "0";
  /** a scripted camera (?campath=, capture shots): a pure function of drive time; null = the interactive camera */
  camPath: CamPath | null = parsePath(params.get("campath"));
  /** frame-exact capture (?capture=1, tools/capture.ts): the page only advances when told to; dt is the simulated
   *  step of the next frame (0 while a frame settles), envRequest re-captures the car's surroundings once */
  readonly capture = params.get("capture") === "1" ? { dt: 0, envRequest: false, maskAmt: null as number | null } : null;
  /** no mode picked (by ?mode= or the menu): Generative wherever a clip covers T, the splat view elsewhere */
  private modeAuto = !params.has("mode");
  /** clean vehicles (smooth models, the default) or the scanned splat models (?vehicles=scanned, menu) */
  private cleanVehicles = (params.get("vehicles") ?? localStorage.getItem("rs-vehicles") ?? "clean") !== "scanned";
  private vehicleKit: VehicleKit | null = null;
  /** over a generated clip: vehicle depth, then the splats in front of it (their alpha = how much hides a vehicle) */
  private occRT = new THREE.WebGLRenderTarget(4, 4, { depthBuffer: true, type: THREE.UnsignedByteType });
  private depthOnly = new THREE.MeshBasicMaterial({ colorWrite: false });
  /** photographic layers (the scene's own, or borrowed: map.json "photo", ?photo=) */
  private photo?: PhotoLayers;
  private photoSource = "";
  private photoOn: PhotoSwitches = { sky: true, backdrop: true, cutouts: true, movers: true };
  /** eased switch amounts */
  private photoEase = { sky: 0, backdrop: 0, cutouts: 0, movers: 0 };
  private photoNow = { sky: 0, backdrop: 0, cutouts: 0, movers: 0 };
  private photoEnvKey = -1;
  /** distance haze (chase view) and the sky's horizon band. Optimised scenes have real trees and buildings far
   *  out: they fade later and only most of the way into the sky, and the band is slim, so the far field does not
   *  end in a flat grey wall */
  private far = { h0: 50, h1: 140, hs: 1, band: 0.16 };
  private lastEnvBuild = 0;
  private envMs = 0;
  private overlay: Overlay = (["auto", "car", "path", "none"].includes(params.get("overlay") ?? "") ? params.get("overlay") : "auto") as Overlay;
  /** no car, shadow or path (for recording reference renders of the chase view) */
  private readonly bare = params.get("bare") === "1";
  private path?: PathViz;
  private frustums?: Frustums;
  private rigCams: CamGeom[] = cameraGeometry(defaultRig());
  private puck: THREE.Mesh;
  private T = 0;
  private playing = true;
  private rate = 1;
  private mode: ViewMode = "live";
  private night = 0;
  private nightTarget = 0;
  private memBlend = 0;
  private lastT = -1;
  private st?: DriveState;
  private fogColor = new THREE.Color(0.78, 0.81, 0.83);
  private frames: number[] = [];
  private fps = 0;
  private frameMs = 0;
  private scale = 1;
  private baseScale = 1;
  private fixedScale: number | null = params.has("dpr") ? Number(params.get("dpr")) : null;
  private lastScaleChange = 0;
  private lastHud = 0;
  private lastSceneLoad = 0;
  private dirtyParams = true;
  private brakeGlow = 0;
  private dts = new Float32Array(1200);
  private dtIdx = 0;
  sceneList: SceneEntry[] = [];

  constructor(private readonly root: HTMLElement) {
    if (import.meta.env.DEV) {
      // dev only: automation runs in an isolated world; this evaluates an expression in the page and writes the result to the DOM
      document.addEventListener("rs-eval", (e) => {
        try {
          document.body.dataset.evalOut = JSON.stringify(new Function("app", "THREE", "return (" + (e as CustomEvent<string>).detail + ")")(this, THREE));
        } catch (err) {
          document.body.dataset.evalOut = "ERR " + String(err);
        }
      });
    }
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance", stencil: false });
    this.renderer.domElement.className = "view";
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    root.append(this.renderer.domElement);
    this.baseScale = Math.min(window.devicePixelRatio || 1, MOBILE ? 2 : 2);
    this.scale = this.fixedScale ?? (MOBILE ? Math.min(this.baseScale, 1.5) : this.baseScale);
    this.renderer.setPixelRatio(this.scale);

    this.spark = new SparkRenderer({
      renderer: this.renderer,
      // anti-aliasing for sub-pixel splats (opacity-compensated), radial sort for stable rotation
      blurAmount: 0.3,
      sortRadial: true,
      minAlpha: 0.5 / 255,
      maxStdDev: Math.sqrt(8),
      enableLod: false,
    });
    this.scene.add(this.spark);
    this.scene.add(this.sky.mesh);
    this.apron = new Apron(this.G);

    // environment maps are y-up; this world is z-up
    this.scene.environmentRotation.set(Math.PI / 2, 0, 0);
    this.env = new CarEnvironment(this.renderer, this.sky);
    // the clean vehicles are built from the ego car's model once it is in
    void this.car.ready.then(() => {
      if (!this.car.model) return;
      this.vehicleKit = new VehicleKit(this.car.model);
      if (this.objects) this.objects.kit = this.vehicleKit;
      // compile the vehicles' shaders now (one of each body), so the first car to appear does not stall a frame
      const kit = this.vehicleKit;
      const warm = new THREE.Group();
      const made = (["saloon", "suv"] as const).map((kind) => kit.make({ kind, dims: [4.7, 1.9, 1.6], paint: { color: new THREE.Color(0x777777), metal: 0.4, rough: 0.3 }, parked: false }));
      for (const v of made) {
        v.update({ x: 0, y: 0, z: -1000, yaw: 0, speed: 0, accel: 0, dist: 0, yawRate: 0, fade: 0.5 }, 0, 0.016);
        warm.add(v.root);
      }
      void this.renderer.compileAsync(warm, this.ctl.camera, this.scene).catch(() => {}).finally(() => made.forEach((v) => v.dispose()));
    });
    this.rebuildEnvironment();
    this.scene.add(this.hemi);
    this.scene.add(this.apron.mesh);
    this.scene.add(this.car.root);

    const puckShape = new THREE.Shape();
    puckShape.moveTo(1, 0);
    puckShape.lineTo(-0.7, 0.62);
    puckShape.lineTo(-0.35, 0);
    puckShape.lineTo(-0.7, -0.62);
    puckShape.closePath();
    this.puck = new THREE.Mesh(new THREE.ShapeGeometry(puckShape), new THREE.MeshBasicMaterial({ color: 0x2f6bff, depthTest: false, depthWrite: false, transparent: true, toneMapped: false }));
    this.puck.renderOrder = 40;
    this.puck.visible = false;
    this.scene.add(this.puck);

    this.ctl = new CameraController(this.renderer.domElement);
    this.ctl.onModeChange = (m) => this.onCamMode(m);
    this.hud = new Hud({
      playPause: () => this.setPlaying(!this.playing),
      seek: (t) => this.seek(t),
      skip: (d) => this.seek(this.T + d),
      rate: () => this.setRate(RATES[(RATES.indexOf(this.rate) + 1) % RATES.length]),
      mode: (m) => {
        // an explicit pick holds; until then playback moves in and out of generated clips by itself
        this.modeAuto = false;
        this.setMode(m);
      },
      scene: (u) => this.loadScene(u),
      map: () => this.ctl.setMode(this.ctl.mode === "map" ? "chase" : "map"),
      night: () => this.toggleNight(),
      camera: () => this.cycleCamera(),
      cameraStep: (d) => this.stepCamera(d),
      view: (v) => (v === "cams" ? this.cycleCamera(true) : this.ctl.setMode(v)),
      vehicles: (clean) => {
        this.cleanVehicles = clean;
        localStorage.setItem("rs-vehicles", clean ? "clean" : "scanned");
        this.objects?.setClean(clean);
        this.hud.setVehicles(this.objects ? clean : null);
      },
      ground: (on) => {
        this.groundOn = on;
        this.hud.setGround(this.atlas ? on : null);
        this.dirtyParams = true;
      },
      photoLayer: (k) => {
        this.photoOn[k] = !this.photoOn[k];
        this.hud.setPhoto(this.photo ? { source: this.photoSource, on: this.photoOn } : null);
      },
      predicted: (on) => {
        if (this.gen) this.gen.predicted = on;
        if (this.atlas) this.atlas.predicted = on;
        this.hud.setPredicted(on);
      },
      overlay: (o) => {
        this.overlay = o;
        this.hud.setOverlay(o);
      },
      colour: (c) => {
        this.colourMode = c;
        this.hud.setColour(c);
        localStorage.setItem("rs-colour", c);
      },
      paint: (id) => {
        this.car.setPaint(id);
        this.hud.setPaint(id);
        localStorage.setItem("rs-paint", id);
      },
      endTrip: () => {
        this.seek(this.drive?.duration ?? 0);
        this.setPlaying(false);
      },
    });
    const paint = params.get("paint") ?? localStorage.getItem("rs-paint");
    if (paint) this.car.setPaint(paint);
    this.hud.setPaint(this.car.paintId);
    if (!params.get("colour") && localStorage.getItem("rs-colour") === "camera") this.colourMode = "camera";
    this.gradeAmt = this.colourMode === "natural" ? 1 : 0;
    this.hud.setColour(this.colourMode);
    fetchJson<{ entries: StreetEntry[] }>("/streets.json").then((s) => s && this.hud.setStreets(s.entries));
    root.append(this.hud.root);
    this.hud.setRate(this.rate);
    this.hud.setPlaying(this.playing);
    this.hud.setMode(this.mode);
    this.hud.setCamMode("chase", "");
    if (params.get("night") === "1") this.toggleNight(true);
    if (params.get("paused") === "1") this.setPlaying(false);
    if (["memory", "stitched", "generative"].includes(params.get("mode") ?? "")) this.setMode(params.get("mode") as ViewMode);
    this.hud.setOverlay(this.overlay);
    if (params.get("cam")) this.ctl.setMode(params.get("cam") as CamMode);
    if (params.get("orbit")) {
      const [a, e, z] = params.get("orbit")!.split(",").map(Number);
      this.ctl.setOrbit(a || 0, e || 0, z || 1);
    }
    if (params.get("stats") === "1") this.hud.toggleStats();

    window.addEventListener("resize", () => this.resize());
    window.addEventListener("keydown", (e) => this.onKey(e));
    this.applySize();
    (window as unknown as { __rs: unknown }).__rs = this;
    if (this.capture) {
      // the capture tool's handle: step to a drive time, ask whether everything for it is in, toggles
      const cap = this.capture;
      (window as unknown as { __capture: unknown }).__capture = {
        step: (T: number, dt: number) => {
          this.seek(T);
          cap.dt = dt;
        },
        requestEnv: () => (cap.envRequest = true),
        state: () => this.captureState(),
        setPath: (p: CamPath | null) => (this.camPath = p),
        hud: (on: boolean) => (this.hud.root.style.visibility = on ? "visible" : "hidden"),
        car: (on: boolean) => (this.showCar = on),
        ribbon: (on: boolean) => (this.showRibbon = on),
        maskAmt: (v: number | null) => (cap.maskAmt = v),
        // keyed values: "maskAmt" (the clip's Predicted overlay), "dir" (the director look's amount)
        set: (k: string, v: number) => {
          if (k === "maskAmt") cap.maskAmt = v;
          else if (k === "dir" && this.director) this.director.amount = v;
        },
      };
      this.setPlaying(false);
      // the transport shows playing (the page is only stepped, not paused)
      this.hud.setPlaying(true);
    }
  }

  /** a size or resolution change is applied at the start of the next frame, before it renders: resizing the
   *  canvas clears it, and doing that after a frame was drawn (the adaptive resolution step used to) presents
   *  that frame empty, so the page background shows for one frame */
  private resize() {
    this.needsResize = true;
  }
  private needsResize = true;
  private applySize() {
    this.needsResize = false;
    const w = this.root.clientWidth, h = this.root.clientHeight;
    this.renderer.setPixelRatio(this.scale);
    this.renderer.setSize(w, h, false);
  }

  private onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement;
    if (t && (t.tagName === "INPUT" || t.tagName === "SELECT")) return;
    const k = e.key;
    const free = this.ctl.mode === "free";
    if (k === " ") { this.setPlaying(!this.playing); e.preventDefault(); }
    else if (k === "ArrowLeft") { this.seek(this.T - (e.shiftKey ? 30 : 5)); e.preventDefault(); }
    else if (k === "ArrowRight") { this.seek(this.T + (e.shiftKey ? 30 : 5)); e.preventDefault(); }
    else if (k === "ArrowUp") { this.setRate(RATES[Math.min(RATES.length - 1, RATES.indexOf(this.rate) + 1)]); e.preventDefault(); }
    else if (k === "ArrowDown") { this.setRate(RATES[Math.max(0, RATES.indexOf(this.rate) - 1)]); e.preventDefault(); }
    else if (k === "1") this.ctl.setMode("chase");
    else if (k === "2") this.ctl.setMode(this.ctl.mode === "map" ? "chase" : "map");
    else if (k === "3") this.cycleCamera();
    else if (k === "4") this.ctl.setMode("free");
    else if (free) return;
    else if (k === "m" || k === "M") this.setMode(this.mode === "memory" ? "live" : "memory");
    else if (k === "l" || k === "L") this.setMode("live");
    else if (k === "k" || k === "K") this.setMode("stitched");
    else if (k === "n" || k === "N") this.toggleNight();
    else if (k === "f" || k === "F") this.hud.toggleStats();
    else if (k === "Escape") this.ctl.setMode("chase");
  }

  private cycleCamera(enterOnly = false) {
    if (this.ctl.mode !== "cams" || enterOnly) {
      this.ctl.camIndex = 0;
      this.ctl.setMode("cams");
    } else if (this.ctl.camIndex + 1 >= this.rigCams.length) {
      this.ctl.setMode("chase");
    } else {
      this.ctl.camIndex++;
      this.onCamMode("cams");
    }
  }
  private stepCamera(d: number) {
    const n = this.rigCams.length;
    this.ctl.camIndex = (this.ctl.camIndex + d + n) % n;
    this.onCamMode(this.ctl.mode);
  }

  private onCamMode(m: CamMode) {
    const g = this.rigCams[this.ctl.camIndex % Math.max(1, this.rigCams.length)];
    const label = m === "cams" ? `${CAMERA_LABELS[g?.name] ?? g?.name} camera · ${this.ctl.camIndex + 1}/${this.rigCams.length}` : m === "free" ? "Free camera · drag to look, WASD QE to move · Esc" : "";
    this.hud.setCamMode(m, label);
    this.car.visible = m !== "cams";
    this.dirtyParams = true;
    this.tiles?.update(this.st?.x ?? 0, this.st?.y ?? 0, this.T, 200, this.mode === "stitched", performance.now(), true);
  }

  private setPlaying(p: boolean) {
    if (p && this.drive && this.T >= this.drive.duration - 0.05) this.seek(0);
    this.playing = p;
    this.hud.setPlaying(p);
  }
  private setRate(r: number) {
    this.rate = r;
    this.hud.setRate(r);
  }
  private setMode(m: ViewMode) {
    this.mode = m;
    this.hud.setMode(m);
    this.dirtyParams = true;
    this.planTiles(true);
  }
  private toggleNight(on?: boolean) {
    this.nightTarget = on === undefined ? (this.nightTarget > 0.5 ? 0 : 1) : on ? 1 : 0;
    this.hud.setNight(this.nightTarget > 0.5);
  }
  private seek(t: number) {
    if (!this.drive) return;
    const nt = THREE.MathUtils.clamp(t, 0, this.drive.duration);
    if (Math.abs(nt - this.T) > 2) this.ctl.snap();
    this.T = nt;
    this.planTiles(true);
  }

  async loadScene(url: string) {
    const base = sceneUrl(url);
    const token = ++this.lastSceneLoad;
    this.hud.setLoading("Loading drive…");
    const [driveRes, rig, map] = await Promise.all([
      loadDrive(base + "drive.json").catch((e): Drive | null => { console.error(e); return null; }),
      fetchJson<RigJson>(base + "rig.json"),
      fetchJson<MapJson>(base + "map.json"),
    ]);
    // compact tile cache (tools/build-tile-cache.ts), only worth probing when there are tiles; it also records
    // whether the scene has objects.json, which saves a 404 on scenes without movers
    const rsb = map?.tiles?.length ? await fetchJson<RsbIndex & { objects?: boolean; generated?: boolean; photo?: boolean; ground?: boolean }>(base + "rsb/index.json") : null;
    const [objectsJson, generated] = await Promise.all([
      !rsb || rsb.objects ? fetchJson<ObjectsJson>(base + "objects.json") : null,
      // ?generated=<url of a generated.json> tries a clip on any scene before it is attached (files relative to it)
      params.get("generated") ? fetchJson<GeneratedJson>(params.get("generated")!) : !rsb || rsb.generated !== false ? fetchJson<GeneratedJson>(base + "generated.json") : null,
    ]);
    if (token !== this.lastSceneLoad) return;
    if (!driveRes) {
      this.hud.setLoading(`Could not load ${base}drive.json`);
      return;
    }
    this.hud.setScenes(this.sceneList, base);
    const url2 = new URL(location.href);
    url2.searchParams.set("scene", base.replace(/^\/scenes\/|\/$/g, ""));
    history.replaceState(null, "", url2);
    // tear down the previous scene
    this.tiles?.dispose();
    if (this.tiles) this.scene.remove(this.tiles.group);
    this.objects?.clearShadows();
    this.objects?.dispose();
    if (this.objects) this.scene.remove(this.objects.group, this.objects.vehicles);
    this.objects = undefined;
    this.gen?.dispose();
    this.gen = undefined;
    if (this.atlas) {
      this.scene.remove(this.atlas.group);
      this.atlas.dispose();
      this.atlas = undefined;
    }
    if (this.photo) {
      this.scene.remove(this.photo.group, this.photo.backdrop);
      this.photo.dispose();
      this.photo = undefined;
    }
    if (this.path) this.scene.remove(this.path.ribbon, this.path.chevrons, this.path.route);
    if (this.frustums) this.car.root.remove(this.frustums.group);

    const drive = driveRes;
    this.drive = drive;
    const rigJson = rig ?? defaultRig();
    this.rigCams = cameraGeometry(rigJson);
    this.frustums = new Frustums(rigJson);
    this.car.root.add(this.frustums.group);
    const turns = findTurns(drive);
    this.path = new PathViz(drive, turns);
    this.scene.add(this.path.ribbon, this.path.chevrons, this.path.route);
    this.hud.setDrive(drive, turns, speedLimits(drive));
    const at = (p?: string | null) => (p ? new URL(p, new URL(base, location.href)).href : null);
    await this.sky.loadPanorama(at(map?.sky), at(map?.environment));
    this.envKey = "";
    // white balance for the Natural grade: from the tile cache, from an earlier visit, or estimated live
    this.sceneKey = `${base}:${map?.tiles?.reduce((a, t) => a + t.count, 0) ?? 0}`;
    this.wb = (rsb as (RsbIndex & { grade?: WhiteBalance }) | null)?.grade ?? JSON.parse(localStorage.getItem("rs-wb:" + this.sceneKey) ?? "null");
    this.roadSampler = this.wb ? null : new RoadColour(new PathIndex(drive.x, drive.y, drive.z));
    this.G.grade2.value.set(GRADE2[0], GRADE2[1], GRADE2[2], GRADE2[3]);
    this.applyWhiteBalance();
    this.groundFilterOn = false;
    if (map && map.tiles?.length) {
      this.tiles = new TileManager(base, map, this.G, {
        workers: MOBILE ? 2 : 3,
        maxSplats: MOBILE ? 1_800_000 : 6_000_000,
        cacheBytes: MOBILE ? 80e6 : 450e6,
      }, rsb);
      this.scene.add(this.tiles.group);
      this.tiles.onInstall = (d) => this.sampleRoad(d);
      this.mapTiles = map.tiles;
      this.hud.setStatus("");
    } else {
      this.tiles = undefined;
      this.hud.setStatus("This scene has no splat tiles yet");
    }
    if (objectsJson?.objects?.length) {
      this.objects = new ObjectManager(base, objectsJson, this.G);
      this.objects.setClean(this.cleanVehicles);
      this.objects.kit = this.vehicleKit;
      this.objects.groundAt = (x, y) => this.atlas?.surfaceAt(x, y) ?? null;
      this.scene.add(this.objects.group, this.objects.vehicles);
    }
    this.hud.setVehicles(this.objects ? this.cleanVehicles : null);
    if (generated?.clips?.length) {
      const gp = params.get("generated");
      this.gen = new Generative(gp ? new URL(".", new URL(gp, location.href)).href : base, generated);
      this.gen.predicted = params.get("predicted") === "1";
    }
    const atlas = !rsb || rsb.ground !== false ? await GroundAtlas.load(base, this.G, this.renderer.capabilities.getMaxAnisotropy()) : null;
    if (token !== this.lastSceneLoad) {
      atlas?.dispose();
      return;
    }
    if (atlas) {
      this.atlas = atlas;
      this.scene.add(atlas.group);
    }
    this.hud.setGround(this.atlas ? this.groundOn : null);
    // director looks (captured shots only)
    if (this.director) this.scene.remove(this.director.group);
    this.director = params.get("director") ? new Director(params.get("director")!, drive, params) : null;
    if (this.director) {
      const D = this.director;
      this.scene.add(D.group);
      if (this.tiles && D.has("tiles")) this.tiles.presence = (t) => D.tilePresence(t, this.T);
      if (this.atlas && D.has("atlas")) this.atlas.presence = (b) => D.atlasPresence(b, this.T);
      const ps = Number(params.get("dpoint") ?? 0);
      if (ps > 0) this.G.director.value.z = ps;
    }
    const predicted = params.get("predicted") === "1";
    if (this.atlas) this.atlas.predicted = predicted;
    this.hud.setPredicted(predicted, !!this.gen?.count || !!this.atlas?.hasProvenance);
    // photographic layers: ?photo= (a scene name, a URL of an ibr.json, or off), else map.json "photo", else the
    // scene's own ibr.json, else (stitched-opt-* and live-* scenes) the newest ibr-* scene of the same drive
    const pp = params.get("photo");
    const mapPhoto = (map as (MapJson & { photo?: string }) | null)?.photo;
    const ref = (v: string) => (/\.json$|\//.test(v) ? new URL(v, new URL(base, location.href)).href : `/scenes/${v}/ibr.json`);
    const photoUrl = pp ? (/^(off|none|0)$/i.test(pp) ? null : ref(pp)) : mapPhoto ? ref(mapPhoto) : !rsb || rsb.photo !== false ? base + "ibr.json" : null;
    let photo = photoUrl ? await PhotoLayers.load(photoUrl, drive, base + "drive.json", this.G) : null;
    const sceneName = base.replace(/^\/scenes\/|\/$/g, "");
    const optimised = /^stitched-opt-/.test(sceneName);
    this.far = optimised ? { h0: FAR_OPT[0], h1: FAR_OPT[1], hs: FAR_OPT[2], band: FAR_OPT[3] } : { h0: 50, h1: 140, hs: 1, band: 0.16 };
    this.ctl.reach = optimised ? { radius: this.far.h1 + 20, ahead: 35 } : { radius: 172, ahead: 25 };
    this.sky.material.uniforms.band.value = this.far.band;
    if (!photo && !pp && map?.tiles?.length && /^(stitched-opt-|live-)/.test(sceneName)) photo = await this.autoPhoto(drive);
    if (token !== this.lastSceneLoad) {
      photo?.dispose();
      return;
    }
    if (photo) {
      this.photo = photo;
      this.scene.add(photo.group, photo.backdrop);
      const own = new URL(photo.base).pathname === new URL(base, location.href).pathname;
      this.photoSource = new URL(photo.base).pathname.replace(/^\/scenes\/|\/$/g, "") + (photo.auto ? " (auto)" : "");
      // the scene's own layers: all of them. Borrowed ones (the display is now one coherent splat scene): all off,
      // with the switches in the menu; ?layers=sky,backdrop,cutouts,movers is the former photo composition. The
      // photographed sky alone stays off too: across 93-140 s it carries ghost pole tops and a dark seam
      const ls = params.get("layers");
      const list = ls ? (ls === "all" ? ["sky", "backdrop", "cutouts", "movers"] : ls.split(",")) : own ? ["sky", "backdrop", "cutouts", "movers"] : [];
      for (const k of Object.keys(this.photoOn) as (keyof PhotoSwitches)[]) this.photoOn[k] = list.includes(k);
      for (const k of Object.keys(this.photoEase) as (keyof PhotoSwitches)[]) this.photoEase[k] = this.photoOn[k] ? 1 : 0;
      // ?cutouts=5,6,7 (or all): static kinds drawn; borrowed layers default to poles, lights and signs
      const kinds = params.get("cutouts") ?? (own ? null : "5,6,7");
      if (kinds && kinds !== "all") photo.kinds = new Set(kinds.split(",").map(Number).filter((x) => Number.isFinite(x)));
      const ids = params.get("cutoutIds");
      if (ids) photo.ids = new Set(ids.split(",").map(Number));
      // borrowed cut-outs go through the sanity checks (?sanity=0 to see them all)
      photo.sanity = !own && params.get("sanity") !== "0";
      if (params.get("audit") === "photo") {
        const ks = kinds && kinds !== "all" ? kinds.split(",").map(Number) : null;
        photo.audit(ks).then((a) => (document.body.dataset.photoAudit = JSON.stringify(a)));
      }
    }
    this.hud.setPhoto(this.photo ? { source: this.photoSource, on: this.photoOn } : null);
    this.hud.setGenerative(!!this.gen?.count);
    if (this.mode === "generative" && !this.gen?.count) this.setMode("live");
    // default start: leaving the car park, just after 80 s parked at the drive-through; optimised scenes (and
    // photo layers) start out of the turn, onto the straight past the brick office
    let t0 = params.has("t") ? Number(params.get("t")) : optimised ? 109 : this.photo ? this.photo.interval[0] + 16 : 90;
    // a generated clip starting within a few seconds of the default start: open on it, in Generative mode (as
    // when one covers the start time)
    const clipStart = this.gen?.ranges.find(([a, b]) => a > t0 && a <= t0 + 6 && b > a + 2)?.[0];
    if (!params.has("t") && clipStart !== undefined) t0 = clipStart + 0.05;
    if (!params.has("mode") && this.gen?.covering(t0)) this.setMode("generative");
    this.T = THREE.MathUtils.clamp(t0, 0, drive.duration);
    this.ctl.snap();
    this.hud.setLoading(null);
    this.planTiles(true);
  }

  /** the newest ibr-* scene recorded on the same drive (same duration and origin) whose interval lies inside it */
  private async autoPhoto(drive: Drive): Promise<PhotoLayers | null> {
    const ver = (n: string) => Number(n.match(/v(\d+)$/)?.[1] ?? 0);
    const cands = this.sceneList.filter((s) => /^ibr-/.test(s.name)).sort((a, b) => ver(b.name) - ver(a.name));
    for (const c of cands) {
      const p = await PhotoLayers.load(c.url + "ibr.json", null, null, this.G);
      if (!p) continue;
      const d = p.drive;
      const same = Math.abs(d.duration - drive.duration) < 0.5 && Math.abs(d.origin.lat - drive.origin.lat) < 1e-6 && Math.abs(d.origin.lon - drive.origin.lon) < 1e-6;
      if (same && p.interval[0] < drive.duration && p.interval[1] > 0) {
        p.auto = true;
        return p;
      }
      p.dispose();
    }
    return null;
  }

  /** road colour as the viewer shows it (graded unless Camera colours) */
  private roadColour() {
    const r = this.wb?.road ?? [0.37, 0.41, 0.4];
    return gradeColour(new THREE.Color(r[0], r[1], r[2]), this.G);
  }

  private applyWhiteBalance() {
    const g = this.wb?.gains ?? [1, 1, 1];
    const k = WB_STRENGTH;
    this.G.grade.value.set(1 + (g[0] - 1) * k, 1 + (g[1] - 1) * k, 1 + (g[2] - 1) * k, this.gradeAmt);
    this.dirtyParams = true;
    this.rebuildEnvironment();
  }

  /** feed splats from newly loaded tiles near the driven path into the road-colour estimate */
  private sampleRoad(d: { count: number; origin: [number, number, number]; extA: Uint32Array; extB: Uint32Array }) {
    const s = this.roadSampler;
    if (!s) return;
    const fa = new Float32Array(d.extA.buffer);
    const step = Math.max(1, Math.floor(d.count / 30000));
    for (let i = 0; i < d.count; i += step) {
      const i4 = i * 4;
      const b0 = d.extB[i4], b1 = d.extB[i4 + 1];
      s.add(fromHalf(b0 & 65535), fromHalf(b0 >>> 16), fromHalf(b1 & 65535), fromHalf(d.extA[i4 + 3] & 65535), fa[i4] + d.origin[0], fa[i4 + 1] + d.origin[1], fa[i4 + 2] + d.origin[2]);
    }
    if (s.count > 60000) {
      const wb = s.result();
      if (wb) {
        this.wb = wb;
        this.roadSampler = null;
        localStorage.setItem("rs-wb:" + this.sceneKey, JSON.stringify(wb));
        this.applyWhiteBalance();
      }
    }
  }

  /** PMREM of the sky over a road-coloured ground, for the car's reflections and ambient light */
  private rebuildEnvironment() {
    const road = this.roadColour();
    const key = `${this.photoEnvKey}:${this.sky.material.uniforms.useMap.value}:${this.sky.panoWeight.toFixed(2)}:${road.getHexString()}:${this.sky.horizon.getHexString()}:${this.gradeAmt}`;
    if (key === this.envKey) return;
    this.envKey = key;
    this.sky.setGrade(this.G.grade.value, this.G.grade2.value);
    this.sky.setFog(this.sky.dayHorizon((c) => gradeColour(c, this.G)));
    this.scene.environment = this.env.build(road, ...this.envParams());
    this.envMs = this.env.ms;
  }
  /** reflections show at most half a fused panorama and never get darker than the procedural sky (a photographed
   *  sky lights the car with a lower floor) */
  private envParams(): [number, number] {
    return [this.sky.panoWeight * 0.5, 0.95 - 0.4 * this.sky.material.uniforms.photoAmt.value];
  }

  private planTiles(force = false) {
    if (!this.tiles || !this.st) return;
    const f = this.ctl.focus(this.st);
    this.tiles.update(f.x, f.y, this.T, f.radius, this.mode === "stitched", performance.now(), force);
  }

  start() {
    let prev = performance.now();
    this.renderer.setAnimationLoop(() => {
      const now = performance.now();
      let dt = Math.min(0.1, (now - prev) / 1000);
      prev = now;
      // capture: eases and smoothing advance by exactly one output frame per step, nothing while it settles
      if (this.capture) {
        dt = this.capture.dt;
        this.capture.dt = 0;
      }
      this.frame(dt, now);
    });
  }

  private frame(dt: number, now: number) {
    if (this.needsResize) this.applySize();
    const drive = this.drive;
    if (!drive) {
      this.renderer.render(this.scene, this.ctl.camera);
      return;
    }
    if (this.playing && !this.hud.scrubbing) {
      this.T += dt * this.rate;
      if (this.T >= drive.duration) {
        this.T = drive.duration;
        this.setPlaying(false);
      }
    }
    const st = (this.st = drive.state(this.T, this.st));
    // eased mode and night transitions
    this.night += (this.nightTarget - this.night) * (1 - Math.exp(-dt / 0.25));
    if (Math.abs(this.night - this.nightTarget) < 0.002) this.night = this.nightTarget;
    const memTarget = this.mode === "memory" ? 1 : 0;
    this.memBlend += (memTarget - this.memBlend) * (1 - Math.exp(-dt / 0.15));
    if (Math.abs(this.memBlend - memTarget) < 0.002) this.memBlend = memTarget;

    // on the textured ground the car stands on its mesh (it can sit a few centimetres off the drive's height,
    // which would sink the tyres or float them and hide the contact shadow)
    const surf = this.atlas?.surfaceAt(st.x, st.y);
    if (surf != null && Math.abs(surf - st.z) < 0.3) this.carDz += (surf - st.z - this.carDz) * Math.min(1, dt / 0.15);
    else this.carDz *= Math.max(0, 1 - dt / 0.3);
    this.car.update(this.carDz ? { ...st, z: st.z + this.carDz } : st, this.night, dt);
    // the chase pose is a function of drive time alone (smoothed heading, height and speed; see chasePose.ts)
    this.ctl.update(dt, st, this.rigCams, this.root.clientWidth / Math.max(1, this.root.clientHeight), chaseInputs(drive, this.T, st));
    const cam = this.ctl.camera;
    // a scripted camera path overrides the interactive one (and says where to stream)
    const pathFocus = this.camPath && this.ctl.mode === "chase" ? applyPath(this.camPath, cam, this.T, chaseInputs(drive, this.T, st), this.root.clientWidth / Math.max(1, this.root.clientHeight)) : null;
    if (pathFocus) {
      cam.near = pathFocus.mapFog ? 1 : 0.1;
      cam.updateProjectionMatrix();
      cam.updateMatrixWorld();
    }
    this.frameInfo = framing(cam, st);
    this.sky.follow(cam);
    this.sky.setNight(this.night);
    // the cloud deck: over the camera's position (km), drifting slowly from the south-west
    this.sky.material.uniforms.clouds.value = CLOUDS;
    (this.sky.material.uniforms.cloudPos.value as THREE.Vector2).set(cam.position.x / 1000 + this.T * 0.004, cam.position.y / 1000 + this.T * 0.003);

    // photographic layers: inside their interval, by day, not in Memory mode; sky and backdrop in every view but
    // the map's backdrop, cut-outs and movers only near the default chase pose (gone past 30 deg of orbit or 15 deg
    // of extra height)
    const ph = this.photo;
    const P = this.photoNow;
    if (ph) {
      const sm = (x: number) => { const p = Math.min(1, Math.max(0, x)); return p * p * (3 - 2 * p); };
      const k = 1 - Math.exp(-dt / 0.15);
      for (const key of Object.keys(this.photoEase) as (keyof PhotoSwitches)[]) {
        const e = this.photoEase[key] + ((this.photoOn[key] ? 1 : 0) - this.photoEase[key]) * k;
        this.photoEase[key] = Math.abs(e - (this.photoOn[key] ? 1 : 0)) < 0.002 ? (this.photoOn[key] ? 1 : 0) : e;
      }
      const base = ph.amountAt(this.T) * (1 - this.night) * (1 - this.memBlend);
      const o = this.ctl.orbit;
      const cardView = this.ctl.mode === "chase" ? (1 - sm((Math.abs(o.az) * 180 / Math.PI - 18) / 12)) * (1 - sm((o.el * 180 / Math.PI - 7) / 8)) : 0;
      P.sky = base * this.photoEase.sky;
      P.backdrop = base * this.photoEase.backdrop * (this.ctl.mode === "map" ? 0 : 1);
      P.cutouts = base * this.photoEase.cutouts * cardView;
      P.movers = base * this.photoEase.movers * cardView;
      ph.update(this.renderer, this.T, this.playing ? this.rate : 0, cam, drive, P);
      this.sky.setPhoto(ph.sky.tex, ph.sky.w, ph.sky.ready ? P.sky : 0, ph.elevSky, ph.sky.yaw);
    } else {
      P.sky = P.backdrop = P.cutouts = P.movers = 0;
      this.sky.setPhoto([null, null, null, null], new THREE.Vector4(), 0, [-9, 64], 0);
    }
    // the car's reflections follow the photographed sky (rebuilt when its leading knot has moved a dozen on, at
    // most every 4 s: a rebuild costs about 14 ms)
    const pk = ph && P.sky > 0.5 && ph.sky.ready ? ph.skyKey : -1;
    if (pk !== this.photoEnvKey && ((pk === -1) !== (this.photoEnvKey === -1) || now - this.lastEnvBuild > 4000)) {
      this.photoEnvKey = pk;
      this.lastEnvBuild = now;
      this.rebuildEnvironment();
    }

    // fog colour: sky horizon by day (graded like the splats when it comes from a panorama, or the photographed
    // sky's horizon), near-black blue by night (display space)
    this.sky.setGrade(this.G.grade.value, this.G.grade2.value);
    const day = this.sky.dayHorizon((c) => gradeColour(c, this.G));
    if (ph && P.sky > 0 && ph.sky.ready) day.lerp(gradeColour(ph.sky.horizon.clone(), this.G, { kFloor: this.G.grade.value.w }), P.sky);
    this.fogColor.setRGB(
      day.r + (0.035 - day.r) * this.night,
      day.g + (0.045 - day.g) * this.night,
      day.b + (0.065 - day.b) * this.night,
    );
    if (this.director) this.fogColor.lerp(this.director.dark, this.director.darkness);
    this.renderer.setClearColor(this.fogColor);
    this.sky.setFog(this.fogColor);
    const focus = pathFocus ?? this.ctl.focus(st);
    const mapFog = focus.mapFog;
    const G = this.G;
    G.T.value = this.T;
    // grade toggle eases; the apron and environment follow the graded road colour
    const gTarget = this.colourMode === "natural" ? 1 : 0;
    if (this.gradeAmt !== gTarget) {
      this.gradeAmt += (gTarget - this.gradeAmt) * (1 - Math.exp(-dt / 0.12));
      if (Math.abs(this.gradeAmt - gTarget) < 0.003) this.gradeAmt = gTarget;
      G.grade.value.w = this.gradeAmt;
      this.dirtyParams = true;
      if (this.gradeAmt === gTarget) this.rebuildEnvironment();
    }
    // 0.12 s fades inside the contract's 0.25 s overlap: the finer generation is fully in before the coarser
    // one starts to fade, so a hand-over never dips in opacity (two 0.2 s fades would overlap and breathe)
    G.params.value.set(this.mode === "stitched" ? 1 : this.mode === "memory" ? 2 : 0, this.night, this.memBlend, 0.12);
    // haze: measured from the car (so orbiting does not change it), from the camera in free view, and from
    // the map centre in the map view
    const free = this.ctl.mode === "free";
    const hx = mapFog ? focus.x : free ? cam.position.x : st.x;
    const hy = mapFog ? focus.y : free ? cam.position.y : st.y;
    const F = this.far;
    let [h0, h1, hs] = mapFog ? [focus.radius * 0.55, focus.radius * 0.95, 0.75] : free ? [F.h0 + 30, F.h1 + 35, F.hs] : this.ctl.mode === "cams" ? [F.h0 + 15, F.h1 + 10, F.hs] : [F.h0, F.h1, F.hs];
    // over photographic sky and backdrop the splats fade out (showing the photographs) instead of hazing grey
    const hazeOut = mapFog ? 0 : Math.max(P.sky, P.backdrop);
    if (hazeOut > 0) {
      h0 += (60 - h0) * hazeOut;
      h1 += (118 - h1) * hazeOut;
    }
    G.fog.value.set(h0, h1, hs, hazeOut);
    // only over the photographed backdrop: trees and poles hand over to it from 38 to 72 m
    G.fog2.value.set(OBJ_FADE[0], OBJ_FADE[1], 0, 0);
    if (P.backdrop <= 0) G.fog2.value.set(1e4, 1e4 + 1, 0, 0);
    G.hazeCentre.value.set(hx, hy, 0, 0);
    G.loadFocus.value.set(focus.x, focus.y, focus.radius, mapFog ? focus.radius * 0.25 : 12);
    G.fogColor.value.set(this.fogColor.r, this.fogColor.g, this.fogColor.b);
    G.car.value.set(st.x, st.y, Math.cos(st.yaw), Math.sin(st.yaw));
    this.brakeGlow += (Math.max(0.28, this.car.brake) - this.brakeGlow) * (1 - Math.exp(-dt / 0.08));
    G.lights.value.set(1, this.brakeGlow, st.z, 0);
    const road = this.roadColour();
    const nightRoad = new THREE.Color(road.r * 0.17, road.g * 0.19, road.b * 0.27);
    road.lerp(nightRoad, this.night);
    // the apron is "unpaved haze": road colour already part-way into the haze, hazing out sooner than splats
    road.lerp(this.fogColor, 0.3);
    this.apron.update(hx, hy, st.z, road, this.fogColor, mapFog ? h0 : h0 * 0.6, mapFog ? h1 : h1 - 10, mapFog ? focus.radius : h1 + 40);
    // phones: ground-layer splats under the opaque textured ground are left out when tiles load (they are hidden
    // there anyway), so the splat budget reaches the scenery; switched off (and the ground reloaded whole) for
    // Memory mode and the splat ground
    const gf = GROUND_FILTER && !!this.atlas && this.groundOn && this.mode !== "memory";
    if (this.tiles && gf !== this.groundFilterOn) {
      this.groundFilterOn = gf;
      const atlas = this.atlas;
      this.tiles.groundFilter = gf && atlas ? (t) => atlas.filterGrid(t.bbox[0], t.bbox[1], t.bbox[3], t.bbox[4]) ?? "wait" : undefined;
      this.tiles.resetGround();
    }
    if (this.tiles) {
      this.tiles.lookahead = 2.5 + 2 * (this.playing ? this.rate : 0);
      this.tiles.update(focus.x, focus.y, this.T, focus.radius, this.mode === "stitched", now);
      this.tiles.tick(this.renderer);
    }
    if (this.objects) {
      this.objects.night = this.night;
      this.objects.fixedDt = this.capture ? dt : null;
    }
    this.objects?.update(this.T, focus.x, focus.y, focus.radius, this.mode === "stitched", now);
    // the textured ground streams around the same focus; Memory mode shows the splat ground's age colours
    const D = this.director;
    if (D) {
      // director looks: the splat shader's look, what is hidden, the overlays
      this.G.director.value.set(D.splatLook, D.amount, this.G.director.value.z, st.z);
      this.G.camPos.value.set(cam.position.x, cam.position.y, cam.position.z, 0);
      if (this.tiles) this.tiles.group.visible = !D.has("atlas");
      if (this.objects) {
        this.objects.group.visible = this.objects.group.visible && !D.has("atlas");
        this.objects.vehicles.visible = !D.has("atlas") || D.has("tracks");
      }
      this.apron.mesh.visible = !D.has("atlas");
      if (this.atlas && D.has("atlasmask")) {
        this.atlas.predicted = true;
        this.atlas.provOverride = D.amount;
      }
      D.update(this.T, cam, this.renderer.getDrawingBufferSize(new THREE.Vector2()), this.mapTiles, D.has("tracks") && this.objects ? this.objects.tracks(this.T) : null);
    }
    this.atlas?.update(this.renderer, focus.x, focus.y, cam.position, this.groundOn && this.memBlend < 0.5 && !D?.hidesAtlas, now);
    // Spark regenerates on camera motion by itself; time, shading or fades need an explicit nudge
    if (this.tiles && (this.T !== this.lastT || this.dirtyParams || this.night % 1 !== 0 || this.memBlend % 1 !== 0 || this.tiles.stats.loaded > 0)) this.tiles.touch();
    this.lastT = this.T;
    this.dirtyParams = false;

    // meshes: light and fog
    this.hemi.intensity = 0.12 * this.night;
    this.scene.environmentIntensity = 1 - 0.85 * this.night;
    if (this.path) {
      this.path.update(st.s, st.speed, true, dt, this.fogColor, [60, 250]);
      const inMap = this.ctl.mode === "map";
      this.path.route.visible = inMap;
      if (this.ctl.mode === "cams") this.path.ribbon.visible = this.path.chevrons.visible = false;
      if (inMap) this.path.updateRoute(st.s, THREE.MathUtils.clamp(cam.position.z * 0.012, 1.6, 14));
      this.puck.visible = inMap;
      if (inMap) {
        const sc = THREE.MathUtils.clamp(cam.position.z * 0.018, 2.2, 26);
        this.puck.position.set(st.x + Math.cos(st.yaw) * 1.4, st.y + Math.sin(st.yaw) * 1.4, st.z + 0.6);
        this.puck.rotation.set(0, 0, st.yaw);
        this.puck.scale.setScalar(sc);
      }
    }
    this.frustums?.setOpacity(this.memBlend * (this.ctl.mode === "cams" ? 0 : 1));
    this.car.shadow.visible = this.ctl.mode !== "cams";
    if (!this.showCar) this.car.visible = false;
    if (!this.showRibbon && this.path) this.path.ribbon.visible = this.path.chevrons.visible = false;
    if (this.bare) {
      this.car.visible = false;
      if (this.path) this.path.ribbon.visible = this.path.chevrons.visible = false;
      // a vehicle-free plate (as generated clips are made now)
      if (this.objects) this.objects.group.visible = this.objects.vehicles.visible = false;
    }

    const aspect = this.root.clientWidth / Math.max(1, this.root.clientHeight);
    // auto mode: into a clip wherever one covers T (its cross-fade), back to the splat view once the clip's
    // end has faded out
    if (this.modeAuto && this.gen) {
      const want: ViewMode = this.gen.covering(this.T) && this.nightTarget < 0.5 ? "generative" : "live";
      if (this.mode !== want && (want === "generative" || this.gen.fade === 0) && (this.mode === "live" || this.mode === "generative")) this.setMode(want);
    }
    // the clip shows at the chase pose; orbiting away fades it with the angle (gone by about 2.3 degrees)
    const o = this.ctl.orbit;
    const dev = Math.max(Math.abs(o.az), Math.abs(o.el), Math.abs(o.zoom - 1) * 0.5);
    let align = this.ctl.mode === "chase" ? 1 - THREE.MathUtils.smoothstep(dev, 0.004, 0.04) : 0;
    if (pathFocus) {
      // a scripted camera: the same fade, from its angle and distance off the chase pose (a crane dissolves the
      // clip into the splats as it lifts off)
      const ec = this.envCam;
      placeChase(ec, chaseInputs(drive, this.T, st), aspect);
      const d0 = Math.max(1, ec.position.distanceTo(new THREE.Vector3(st.x, st.y, st.z)));
      const pdev = Math.max(ec.quaternion.angleTo(cam.quaternion), (cam.position.distanceTo(ec.position) / d0) * 0.5, Math.abs(cam.fov - ec.fov) / 50);
      align = this.ctl.mode === "chase" ? 1 - THREE.MathUtils.smoothstep(pdev, 0.004, 0.04) : 0;
    }
    if (this.gen && this.capture) {
      this.gen.timeFade = true;
      this.gen.maskOverride = this.capture.maskAmt;
    }
    const genFade = this.gen?.update(dt, this.T, this.playing && !this.hud.scrubbing, this.rate, this.mode === "generative" && this.ctl.mode === "chase" && this.nightTarget < 0.5, align, cam.fov) ?? 0;
    this.gen?.setGrade(this.G.grade.value, this.G.grade2.value);
    if (this.gen && genFade > 0) this.renderGenerative(this.gen, cam, genFade, aspect);
    else this.renderer.render(this.scene, cam);
    // the car's surroundings for its reflections: the chase frame, captured four times a second; the environment
    // is rebuilt from it two frames later (reading the capture right away stalls on the frame still drawing)
    if (this.envPending > 0 && --this.envPending === 0) {
      this.scene.environment = this.env.build(this.roadColour(), ...this.envParams());
      this.envMs += (this.env.ms - this.envMs) * 0.2;
    }
    if (this.capture ? this.capture.envRequest : this.ctl.mode === "chase" && !this.bare && now - this.lastCapture > 250 && params.get("envcap") !== "0") {
      if (this.capture) this.capture.envRequest = false;
      this.lastCapture = now;
      if (pathFocus && align < 0.5) {
        // a scripted camera away from the chase pose (round the car, close to it): the surroundings as the chase
        // camera sees them (the frame on screen would show the car itself), then the frame again
        const ec = this.envCam;
        ec.near = 0.1;
        ec.far = cam.far;
        placeChase(ec, chaseInputs(drive, this.T, st), aspect);
        ec.updateProjectionMatrix();
        ec.updateMatrixWorld();
        this.renderer.render(this.scene, ec);
        this.env.capture(ec, st.yaw);
        if (this.gen && genFade > 0) this.renderGenerative(this.gen, cam, genFade, aspect);
        else this.renderer.render(this.scene, cam);
      } else this.env.capture(cam, st.yaw);
      this.envPending = 2;
    }

    // frame timing and adaptive resolution
    this.dts[this.dtIdx++ % this.dts.length] = dt * 1000;
    this.frames.push(now);
    while (this.frames.length && now - this.frames[0] > 1000) this.frames.shift();
    this.fps = this.frames.length;
    this.frameMs += (dt * 1000 - this.frameMs) * 0.05;
    if (this.fixedScale === null && now - this.lastScaleChange > 1500 && now > 4000) {
      const min = Math.max(0.75, this.baseScale * 0.5);
      // aim for the display rate: step down when frames average over 19.5 ms, try a step up only after 20 s
      if (this.frameMs > 19.5 && this.scale > min) {
        this.scale = Math.max(min, this.scale * 0.85);
        this.lastScaleChange = now + 14000;
        this.resize();
      } else if (this.frameMs < 17.2 && this.scale < this.baseScale && now - this.lastScaleChange > 6000) {
        this.scale = Math.min(this.baseScale, this.scale * 1.1);
        this.lastScaleChange = now;
        this.resize();
      }
    }
    if (now - this.lastHud > 60) {
      this.lastHud = now;
      // readable from automation without page globals
      document.body.dataset.stats = JSON.stringify(this.getStats());
      this.hud.update(st);
      const ts = this.tiles?.stats;
      const busy = ts && ts.loading + ts.queued > 0;
      if (busy) this.hud.setStatus(`Loading ${ts.loading + ts.queued} tile${ts.loading + ts.queued > 1 ? "s" : ""}`);
      else if (this.mode === "generative" && this.gen && !this.gen.ranges.some(([a, b]) => this.T >= a && this.T <= b))
        this.hud.setStatus(`Generated video: ${this.gen.ranges.map(([a, b]) => `${fmtT(a)}–${fmtT(b)}`).join(", ")}`);
      else if (this.mode === "generative" && this.gen?.predicted && this.gen.clip && !this.gen.hasMask) this.hud.setStatus("This clip has no provenance mask");
      else this.hud.setStatus("");
      if (this.hud.statsVisible) {
        const s = this.tiles?.stats;
        this.hud.setStats(
          `fps      ${this.fps}  (${this.frameMs.toFixed(1)} ms)\n` +
          `res      ${this.renderer.domElement.width}x${this.renderer.domElement.height} @${this.scale.toFixed(2)}\n` +
          `splats   ${(this.spark.activeSplats / 1e3).toFixed(0)}k drawn\n` +
          `         ${((s?.splatsLoaded ?? 0) / 1e6).toFixed(2)}M loaded\n` +
          `tiles    ${s?.loaded ?? 0}/${s?.total ?? 0}  +${(s?.loading ?? 0) + (s?.queued ?? 0)}\n` +
          `movers   ${this.objects?.loaded ?? 0}/${this.objects?.count ?? 0}\n` +
          `fetched  ${((s?.bytes ?? 0) / 1e6).toFixed(0)} MB\n` +
          `T        ${this.T.toFixed(2)} s`,
        );
      }
    }
  }

  /** Generative mode: the splat view (unless the clip covers it entirely), the clip over it, then the car and
   *  the path over the clip */
  private renderGenerative(gen: Generative, cam: THREE.PerspectiveCamera, fade: number, aspect: number) {
    const r = this.renderer;
    if (fade < 0.999 || !gen.covers(aspect, cam.fov)) r.render(this.scene, cam);
    else r.clear();
    r.autoClear = false;
    gen.draw(r, aspect, cam.fov);
    // vehicles over a clip made without them: occluded by what the splat scene has in front of them
    const veh = this.objects?.vehicles;
    const overVideo = !!veh && this.cleanVehicles && gen.vehiclesOverlay && veh.children.some((c) => c.visible);
    if (overVideo) this.vehicleOcclusion(veh!, cam);
    for (const c of veh?.children ?? []) (c.userData.vehicle as { setBlended(on: boolean): void } | undefined)?.setBlended(overVideo);
    VEHICLE_OCCLUSION.info.value.x = overVideo ? 1 : 0;
    r.clearDepth();
    const hide = ([this.spark, this.sky.mesh, this.apron.mesh, this.tiles?.group, this.objects?.group, this.atlas?.group, this.puck, this.photo?.group, this.photo?.backdrop, overVideo ? undefined : veh] as (THREE.Object3D | undefined)[]).filter((o): o is THREE.Object3D => !!o);
    const was = hide.map((o) => o.visible);
    for (const o of hide) o.visible = false;
    const carWas = this.car.root.visible;
    const ribWas = this.path?.ribbon.visible ?? false, chevWas = this.path?.chevrons.visible ?? false;
    this.car.root.visible = carWas && gen.carOver(this.overlay);
    if (this.path) {
      this.path.ribbon.visible = ribWas && this.overlay !== "none";
      this.path.chevrons.visible = chevWas && this.overlay !== "none";
    }
    r.render(this.scene, cam);
    hide.forEach((o, i) => (o.visible = was[i]));
    if (overVideo) {
      VEHICLE_OCCLUSION.info.value.x = 0;
      for (const c of veh!.children) (c.userData.vehicle as { setBlended(on: boolean): void } | undefined)?.setBlended(false);
    }
    this.car.root.visible = carWas;
    if (this.path) {
      this.path.ribbon.visible = ribWas;
      this.path.chevrons.visible = chevWas;
    }
    r.autoClear = true;
  }

  /** the occlusion of the vehicles over a clip, at half resolution: their depth, then every splat drawn against it;
   *  what lands in front of a vehicle leaves alpha behind (approximate: splats test at their centres) */
  private vehicleOcclusion(veh: THREE.Object3D, cam: THREE.PerspectiveCamera) {
    const r = this.renderer;
    const c = r.domElement;
    const w = Math.max(4, Math.round(c.width / 2)), h = Math.max(4, Math.round(c.height / 2));
    if (this.occRT.width !== w || this.occRT.height !== h) this.occRT.setSize(w, h);
    const prevTarget = r.getRenderTarget();
    const clear = r.getClearColor(new THREE.Color()), clearA = r.getClearAlpha();
    r.setRenderTarget(this.occRT);
    r.setClearColor(0x000000, 0);
    r.clear(true, true, false);
    const all = this.scene.children.slice();
    const was = all.map((o) => o.visible);
    for (const o of all) o.visible = o === veh;
    this.scene.overrideMaterial = this.depthOnly;
    r.render(this.scene, cam);
    this.scene.overrideMaterial = null;
    for (const o of all) o.visible = o === this.spark || o === this.tiles?.group;
    r.render(this.scene, cam);
    all.forEach((o, i) => (o.visible = was[i]));
    r.setRenderTarget(prevTarget);
    r.setClearColor(clear, clearA);
    VEHICLE_OCCLUSION.tex.value = this.occRT.texture;
    VEHICLE_OCCLUSION.info.value.set(1, 1 / w, 1 / c.width, 1 / c.height);
  }

  /** capture: what is still on its way for the current drive time (all false / zero = the frame is final) */
  private captureState() {
    const ts = this.tiles?.stats;
    // tiles still fading in after they loaded
    const live = (this.tiles as unknown as { live?: Map<string, { loadedAt: number }> } | undefined)?.live;
    let unfaded = 0;
    if (live) for (const l of live.values()) if (performance.now() - l.loadedAt < 470) unfaded++;
    const sp = this.spark as unknown as { sorting: boolean; sortDirty: boolean; sortTimeoutId?: unknown };
    let gen = 0;
    const g = this.gen as unknown as { players?: { video: HTMLVideoElement; texture: THREE.VideoTexture; clip: unknown }[]; active?: number; fade: number; sync: { want: number }; current?: { fps?: number } } | undefined;
    if (g?.players && g.fade > 0) {
      const p = g.players[g.active ?? 0];
      const v = p.video;
      const fps = (p.clip as { fps?: number } | null)?.fps ?? 30;
      if (v.seeking || v.readyState < 2 || Math.abs(v.currentTime - ((Math.floor(g.sync.want * fps) + 0.5) / fps)) > 0.3 / fps) gen = 1;
      // the frame the video now shows goes to the texture whether or not a frame callback fired
      p.texture.needsUpdate = true;
    }
    const movers = (this.objects as unknown as { movers?: { state: string; parts: { loadedAt: number }[] }[] } | undefined)?.movers;
    const now = performance.now();
    return {
      T: this.T,
      tiles: ts ? ts.loading + ts.queued : 0,
      unfaded,
      sorting: !!sp && (sp.sorting || sp.sortDirty || sp.sortTimeoutId !== undefined),
      atlas: this.atlas ? this.atlas.stats.pending + this.atlas.unsettled : 0,
      // still loading, or scanned parts still fading in
      movers: movers ? movers.filter((m) => m.state === "loading" || m.parts.some((p) => now - p.loadedAt < 320)).length : 0,
      kit: !this.objects || !!this.vehicleKit,
      gen,
      env: this.envPending,
      // no splats drawn (the director's atlas look): nothing to sort
      splats: !this.director?.has("atlas"),
    };
  }

  getStats() {
    return {
      fps: this.fps,
      frameMs: this.frameMs,
      scale: this.scale,
      width: this.renderer.domElement.width,
      height: this.renderer.domElement.height,
      drawn: this.spark.activeSplats,
      loaded: this.tiles?.stats.splatsLoaded ?? 0,
      tiles: this.tiles?.stats.loaded ?? 0,
      pending: (this.tiles?.stats.loading ?? 0) + (this.tiles?.stats.queued ?? 0),
      T: this.T,
      fog: this.fogColor.getHexString(),
      wb: this.wb?.gains.map((v) => +v.toFixed(3)),
      road: this.wb?.road.map((v) => +v.toFixed(3)),
      frame: this.frameInfo,
      moverShadows: this.G.moverInfo.value.x,
      ground: this.atlas && this.atlas.stats,
      groundFiltered: this.tiles?.filtered ?? 0,
      movers: this.objects?.loaded ?? 0,
      envMs: +this.envMs.toFixed(1),
      photo: this.photo && { ...this.photo.info, dropped: [...this.photo.dropped.keys()], on: Object.fromEntries(Object.entries(this.photoNow).map(([k, v]) => [k, +v.toFixed(2)])), pending: this.photo.pending },
      gen: this.gen && { fade: +this.gen.fade.toFixed(3), opacity: +this.gen.opacity.toFixed(3), clip: +this.gen.sync.clip.toFixed(3), want: +this.gen.sync.want.toFixed(3), mask: this.gen.hasMask, predicted: this.gen.predicted },
      sky: this.sky.panoStats && { lum: +this.sky.panoStats.lum.toFixed(3), blotch: +this.sky.panoStats.blotch.toFixed(3), weight: +this.sky.panoStats.weight.toFixed(2) },
      ...this.spikes(),
    };
  }

  /** frame-time spikes over the last 1200 frames */
  spikes() {
    const n = Math.min(this.dtIdx, this.dts.length);
    const a = Array.from(this.dts.subarray(0, n)).sort((x, y) => x - y);
    return { p99: a[Math.floor(n * 0.99)] ?? 0, max: a[n - 1] ?? 0, over25: a.filter((x) => x > 25).length, frames: n };
  }
  resetSpikes() {
    this.dtIdx = 0;
  }
}

const fmtT = (t: number) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

async function main() {
  const app = new App(document.getElementById("app")!);
  app.start();
  const index = await fetchJson<{ scenes: SceneEntry[] }>("/scenes/index.json");
  const scenes = index?.scenes ?? [];
  let target = sceneUrl(params.get("scene"));
  // default: the newest stitched-opt-vN (its movers edition when there is one), then the newest live-* scene with
  // tiles, then the newest stitched-*, then any scene with tiles
  if (!params.get("scene")) {
    const ver = (n: string) => Number(n.match(/v(\d+)$/)?.[1] ?? 0);
    const pick = (re: RegExp) => scenes.filter((s) => s.hasMap && re.test(s.name)).sort((a, b) => ver(b.name) - ver(a.name) || b.name.localeCompare(a.name))[0];
    // the newest optimised stitched scene first, its movers edition when there is one (stitched-opt-v5-movers-v1
    // over stitched-opt-v5)
    const opt = scenes
      .filter((s) => s.hasMap && /^stitched-opt-v\d+(-movers-v\d+)?$/.test(s.name))
      .sort((a, b) => {
        const v = (n: string) => Number(n.match(/^stitched-opt-v(\d+)/)?.[1] ?? 0);
        const mv = (n: string) => Number(n.match(/-movers-v(\d+)$/)?.[1] ?? 0);
        return v(b.name) - v(a.name) || mv(b.name) - mv(a.name);
      })[0];
    target = (opt ?? pick(/live/) ?? pick(/stitch/) ?? scenes.find((s) => s.hasMap) ?? scenes.find((s) => s.url === target))?.url ?? target;
  }
  app.sceneList = scenes;
  app.hud.setScenes(scenes, target);
  await app.loadScene(target);
}

main().catch((e) => console.error(e));

/** horizon height (from the top), car centre height and car width as fractions of the view, camera height above
 *  the road and pitch (degrees, negative = down) */
function framing(cam: THREE.PerspectiveCamera, st: { x: number; y: number; z: number; yaw: number }) {
  cam.updateMatrixWorld();
  const f = new THREE.Vector3();
  cam.getWorldDirection(f);
  const hz = new THREE.Vector3(f.x, f.y, 0).normalize().multiplyScalar(1e5).add(cam.position).project(cam);
  const c = Math.cos(st.yaw), s = Math.sin(st.yaw);
  const ctr = new THREE.Vector3(st.x + c * 1.45, st.y + s * 1.45, st.z + 0.75);
  const l = ctr.clone().add(new THREE.Vector3(-s * 0.93, c * 0.93, 0)).project(cam);
  const r = ctr.clone().add(new THREE.Vector3(s * 0.93, -c * 0.93, 0)).project(cam);
  const p = ctr.clone().project(cam);
  const k = (v: number) => +v.toFixed(3);
  return { horizon: k((1 - hz.y) / 2), carY: k((1 - p.y) / 2), carW: k(Math.abs(l.x - r.x) / 2), camH: +(cam.position.z - st.z).toFixed(2), pitch: +((Math.asin(f.z) * 180) / Math.PI).toFixed(1) };
}
