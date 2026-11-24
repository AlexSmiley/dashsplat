# Browser deployment

The original viewer is already a Vite/Three.js browser application. This branch includes a minimal **synthetic telemetry-only demo** so the app can open without processing Tesla footage or downloading model weights. The demo has no reconstructed Gaussian splats: it shows the vehicle and procedural environment.

## Local

```sh
cd viewer
npm install
npm run dev
```

Open `http://127.0.0.1:5188/?scene=demo`. To build: `npm run build`, then `npm run preview`.

## Vercel

Import the **repository root** in Vercel (do not set the root directory to `viewer`). The root `vercel.json` installs dependencies in `viewer`, builds the static Vite app, and publishes `viewer/dist`. No Python or GPU function is deployed. The demo is available at `/?scene=demo`.

## Real reconstructed scenes

The six-camera processing pipeline must run separately on a supported GPU workstation. Publish the resulting scene folder at `/scenes/<scene-name>/` and include `drive.json`, `map.json`, and all referenced binary tiles/assets. For production static hosting, place them under `viewer/public/scenes/<scene-name>/` before building (or serve them from an asset host and pass the scene directory URL in `?scene=`). Large datasets should live on external object storage/CDN, with appropriate CORS, MIME types and byte-range support. To list additional scenes in the static deploy, update `viewer/public/scenes/index.json`. The Vite dev server has its own local scenes plugin for repo-root `scenes/`; it can overshadow the bundled demo index when a separate local scenes folder exists.

The demo is not a reconstruction, not real-time AI processing, and does not accept video URLs.
