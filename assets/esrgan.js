// Browser-local super-resolution via TensorFlow.js (WebGPU, WebGL fallback).
//
// Runs Real-ESRGAN / Real-CUGAN graph models (self-hosted under /models). The
// ONNX→ort-web path can't run these (its WebGPU backend crashes on the final
// 3-channel Clip op), but TF.js WebGPU runs them fast (~16-80ms per 128² tile).
//
// Models are fixed-input (one frozen tile size per file). We slide that tile
// over the image with overlap and centre-crop-stitch the results, padding edges
// by replication. Verified model I/O contract (clean-room, not copied):
//   input  "input"    : float32 [1, T, T, 3]   NHWC, RGB, 0..1
//   output "Identity" : float32 [1, T*s, T*s, 3] NHWC, 0..1 (clamp; some overshoot)
//
// Public API:
//   SR.MODELS                                   → model registry
//   await SR.prepare(key, { onStatus, onProgress })  → { scale, backend }
//   await SR.upscaleCanvas(srcCanvas, { onProgress }) → HTMLCanvasElement (W*s × H*s)
//   SR.scale() / SR.backend() / SR.isReady() / SR.dispose()

(function () {
  const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
  const TFJS_WEBGPU = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.min.js';

  // Self-hosted, relative to /tools/*.html → resolves to site-root /models/.
  const MODEL_BASE = '../models/';
  const OVERLAP = 16;   // input-px overlap between adjacent tiles (seam guard)

  const MODELS = {
    anime_fast:   { label: '动漫/游戏 · 极速 (AnimeVideo v3)',     path: 'realesrgan/anime_fast-128',        scale: 4, tile: 128, sizeMB: 1.3 },
    general_fast: { label: '通用 · 均衡 (General x4 v3)',          path: 'realesrgan/general_fast-128',      scale: 4, tile: 128, sizeMB: 2.5 },
    anime_plus:   { label: '动漫/游戏 · 最佳 (x4plus Anime 6B)',   path: 'realesrgan/anime_plus-128',        scale: 4, tile: 128, sizeMB: 8.7 },
    cugan2x:      { label: '通用 · 2× 快速 (Real-CUGAN)',          path: 'realcugan/2x-conservative-128',    scale: 2, tile: 128, sizeMB: 2.5 },
  };

  let tf = null;
  let model = null;
  let curKey = null;
  let backendName = null;
  let preparing = null;

  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = () => res(); s.onerror = () => rej(new Error('加载失败: ' + src));
      document.head.appendChild(s);
    });
  }

  async function ensureTf(onStatus) {
    if (tf) return tf;
    onStatus?.('加载 TensorFlow.js (首次联网 ~1 MB)...');
    if (!window.tf) {
      await loadScript(TFJS);
      await loadScript(TFJS_WEBGPU);
    }
    tf = window.tf;
    // Prefer WebGPU; fall back to WebGL (slower but widely supported).
    onStatus?.('初始化 GPU 后端...');
    let ok = false;
    try { ok = await tf.setBackend('webgpu'); } catch (_) { ok = false; }
    if (ok) { await tf.ready(); backendName = 'webgpu'; }
    else {
      try { await tf.setBackend('webgl'); await tf.ready(); backendName = 'webgl'; }
      catch (e) { throw new Error('无法初始化 WebGPU/WebGL 后端: ' + (e.message || e)); }
    }
    return tf;
  }

  async function prepare(key, opts = {}) {
    const { onStatus = () => {}, onProgress = () => {} } = opts;
    if (!MODELS[key]) throw new Error('未知模型: ' + key);
    if (model && curKey === key) return { scale: MODELS[key].scale, backend: backendName };
    if (preparing) { try { await preparing; } catch (_) {} }
    preparing = (async () => {
      await ensureTf(onStatus);
      if (model && curKey !== key) { try { model.dispose(); } catch (_) {} model = null; }
      const m = MODELS[key];
      const cacheKey = 'indexeddb://sr-' + key;
      onStatus(`加载模型 ${m.label} (~${m.sizeMB}MB)...`);
      try {
        model = await tf.loadGraphModel(cacheKey);            // cached after first use
      } catch (_) {
        const url = new URL(MODEL_BASE + m.path + '/model.json', location.href).href;
        model = await tf.loadGraphModel(url, { onProgress: p => onProgress(p) });
        try { await model.save(cacheKey); } catch (_) {}
      }
      curKey = key;
      onStatus(`模型就绪 · ${backendName.toUpperCase()} · ${m.scale}×`);
      return { scale: m.scale, backend: backendName };
    })().catch(e => { preparing = null; throw e; });
    const r = await preparing;
    preparing = null;
    return r;
  }

  function isReady() { return !!model; }
  function scale() { return curKey ? MODELS[curKey].scale : 1; }
  function backend() { return backendName; }

  function dispose() {
    try { model?.dispose(); } catch (_) {}
    model = null; curKey = null;
  }

  function newCanvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

  // Tile start positions covering [0, D) with the given step, last tile clamped
  // so it stays in-bounds. For D <= T a single (padded) tile.
  function positions(D, T, step) {
    if (D <= T) return [0];
    const ps = []; const max = D - T; let p = 0;
    while (true) {
      const x = Math.min(p, max);
      if (ps.length === 0 || ps[ps.length - 1] !== x) ps.push(x);
      if (x >= max) break;
      p += step;
    }
    return ps;
  }

  // Upscale a canvas. Tiles the image, runs each tile through the model on GPU,
  // and centre-crop-stitches the scaled tiles into the output canvas.
  async function upscaleCanvas(srcCanvas, opts = {}) {
    if (!model) throw new Error('SR 未初始化, 先调用 SR.prepare()');
    const { onProgress = () => {} } = opts;
    const m = MODELS[curKey];
    const T = m.tile, s = m.scale, OV = OVERLAP, half = OV >> 1, step = T - OV;
    const W = srcCanvas.width, H = srcCanvas.height;
    const outW = W * s, outH = H * s;

    const srcCtx = srcCanvas.getContext('2d', { willReadFrequently: true });
    const out = newCanvas(outW, outH);
    const octx = out.getContext('2d');

    const xs = positions(W, T, step);
    const ys = positions(H, T, step);
    const total = xs.length * ys.length;
    let done = 0;

    const inCv = newCanvas(T, T);
    const ictx = inCv.getContext('2d');
    ictx.imageSmoothingEnabled = false;

    for (const y0 of ys) {
      const cropTop = (y0 > 0) ? half : 0;
      const cropBot = (y0 < H - T) ? half : 0;        // y0 < max → has a tile below
      const cw_h = Math.min(T, H - y0);                // valid source height in this tile
      for (const x0 of xs) {
        const cropLeft = (x0 > 0) ? half : 0;
        const cropRight = (x0 < W - T) ? half : 0;
        const cw_w = Math.min(T, W - x0);

        // crop T×T native (replicate-pad edges when the image is smaller than T)
        ictx.clearRect(0, 0, T, T);
        ictx.drawImage(srcCanvas, x0, y0, cw_w, cw_h, 0, 0, cw_w, cw_h);
        if (cw_w < T) ictx.drawImage(inCv, cw_w - 1, 0, 1, cw_h, cw_w, 0, T - cw_w, cw_h);
        if (cw_h < T) ictx.drawImage(inCv, 0, cw_h - 1, T, 1, 0, cw_h, T, T - cw_h);

        // run model: [1,T,T,3] 0..1 → [T*s,T*s,3] 0..1
        const pixels = await tf.tidy(() => {
          const x = tf.browser.fromPixels(inCv).toFloat().div(255).expandDims(0);
          const y = model.predict(x).clipByValue(0, 1).squeeze([0]);
          return y;
        });
        const rgba = await tf.browser.toPixels(pixels);   // Uint8ClampedArray RGBA, T*s × T*s
        pixels.dispose();

        const tileCv = newCanvas(T * s, T * s);
        tileCv.getContext('2d').putImageData(new ImageData(rgba, T * s, T * s), 0, 0);

        // centre-crop write region (drop the overlapped halves that a neighbour owns)
        const wValid = Math.min(T, W - x0), hValid = Math.min(T, H - y0);
        const srcX = cropLeft * s, srcY = cropTop * s;
        const srcW = (Math.min(wValid, T - cropRight) - cropLeft) * s;
        const srcH = (Math.min(hValid, T - cropBot) - cropTop) * s;
        if (srcW > 0 && srcH > 0) {
          octx.drawImage(tileCv, srcX, srcY, srcW, srcH, (x0 + cropLeft) * s, (y0 + cropTop) * s, srcW, srcH);
        }

        done++;
        onProgress(done / total);
        // let the event loop breathe so the UI stays responsive
        if ((done & 3) === 0) await new Promise(r => setTimeout(r, 0));
      }
    }
    return out;
  }

  window.SR = { MODELS, prepare, upscaleCanvas, isReady, scale, backend, dispose };
})();
