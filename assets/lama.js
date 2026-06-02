// LaMa image inpainting via ONNX Runtime Web (WebGPU, WASM fallback).
//
// Big-LaMa is a fully-convolutional inpainting network. We DON'T squash the
// whole frame to 512² (that nukes detail) — instead we slide 512² tiles only
// over the mask bounding box, run LaMa per tile at the source resolution, and
// paste each tile's output back ONLY where the mask is set. Pixels outside the
// mask are never touched.
//
// Lazily downloads ORT (~3 MB) + the LaMa model (~200 MB) on first prepare();
// the session is cached for every later inpaint() call.
//
// Public API:
//   await LaMa.prepare({ onStatus, onProgress })  → { ready, provider } | throws
//   await LaMa.inpaint(imageData, mask, { onProgress, onLog })  → ImageData
//       imageData : ImageData (full frame)
//       mask      : Uint8Array length w*h, 1 = remove / inpaint, 0 = keep
//   LaMa.isReady()
//   LaMa.dispose()
//
// Model I/O contract (Carve/LaMa-ONNX · lama_fp32.onnx):
//   image : float32 (1, 3, 512, 512)  RGB 0..1, channels-first
//   mask  : float32 (1, 1, 512, 512)  1 = hole
//   output: float32 (1, 3, 512, 512)  RGB 0..1

(function () {
  const ORT_ESM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.mjs';
  const ORT_SCRIPT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js';
  const MODEL_ID = 'Carve/LaMa-ONNX';
  const MODEL_PATH = `${MODEL_ID}/resolve/main/lama_fp32.onnx`;
  // hf-mirror.com (a widely-used HuggingFace mirror, usually fast & reachable
  // from mainland China) first, official huggingface.co as fallback. Tried in
  // order — a connection failure / non-OK response falls through to the next.
  const MODEL_HOSTS = ['https://hf-mirror.com/', 'https://huggingface.co/'];

  const TILE = 512;
  const TILE_PAD = 64;            // context around the mask the model gets to see
  const TILE_STRIDE = TILE - 128; // 128 px overlap between adjacent tiles

  let session = null;
  let provider = null;
  let preparing = null;

  function newCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  async function loadOrt(onStatus) {
    if (window.ort) return window.ort;
    onStatus?.('加载 onnxruntime-web (首次联网 ~3 MB)...');
    let mod = null;
    try {
      mod = await import(/* @vite-ignore */ ORT_ESM);
    } catch (_) {
      await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = ORT_SCRIPT;
        s.onload = resolve;
        s.onerror = () => reject(new Error('onnxruntime-web 脚本加载失败'));
        document.head.appendChild(s);
      });
    }
    if (mod && mod.default) window.ort = mod.default;
    if (!window.ort) throw new Error('onnxruntime-web 加载失败');
    return window.ort;
  }

  // Stream the model so we can show real download progress (200 MB is a long
  // wait). Tries each host in MODEL_HOSTS order, falling through to the next on
  // a connection error or non-OK status.
  async function downloadModel(onProgress, onStatus) {
    let lastErr = null;
    for (let i = 0; i < MODEL_HOSTS.length; i++) {
      const url = MODEL_HOSTS[i] + MODEL_PATH;
      const host = new URL(url).host;
      try {
        if (i > 0) onStatus?.(`换源重试:${host} ...`);
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const total = parseInt(resp.headers.get('content-length') || '0', 10);
        if (!resp.body || !total) return await resp.arrayBuffer();
        const reader = resp.body.getReader();
        const chunks = [];
        let received = 0;
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          received += value.length;
          onProgress?.(received / total);
        }
        const out = new Uint8Array(received);
        let off = 0;
        for (const c of chunks) { out.set(c, off); off += c.length; }
        return out.buffer;
      } catch (e) {
        lastErr = e;
        onStatus?.(`${host} 下载失败(${e.message})`);
      }
    }
    throw new Error(`下载 LaMa 模型失败(已试 ${MODEL_HOSTS.length} 个源): ${lastErr ? lastErr.message : '未知错误'}`);
  }

  async function prepare(opts = {}) {
    if (session) return { ready: true, provider };
    if (preparing) return preparing;
    const { onStatus = () => {}, onProgress = () => {} } = opts;
    preparing = (async () => {
      const ort = await loadOrt(onStatus);
      // Point WASM backend at a matching CDN dir (used if WebGPU is unavailable).
      try { ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/'; } catch (_) {}
      onStatus('下载 LaMa 模型 (~200 MB, 首次联网 · 优先 hf-mirror 镜像)...');
      const modelBuf = await downloadModel(onProgress, onStatus);
      onStatus('初始化推理 session...');
      const wantGpu = 'gpu' in navigator;
      const eps = wantGpu ? ['webgpu', 'wasm'] : ['wasm'];
      try {
        session = await ort.InferenceSession.create(modelBuf, { executionProviders: eps });
      } catch (e) {
        // WebGPU init can fail on some drivers — retry on pure WASM.
        if (wantGpu) {
          onStatus('WebGPU 初始化失败,回退到 WASM (慢)...');
          session = await ort.InferenceSession.create(modelBuf, { executionProviders: ['wasm'] });
          provider = 'wasm';
        } else {
          throw e;
        }
      }
      if (!provider) provider = wantGpu ? 'webgpu' : 'wasm';
      onStatus(`LaMa 就绪 (${provider})`);
      return { ready: true, provider };
    })().catch(e => { preparing = null; throw e; });
    return preparing;
  }

  function isReady() { return !!session; }

  function dispose() {
    try { session?.release?.(); } catch (_) {}
    session = null;
    provider = null;
    preparing = null;
  }

  // Run one 512² tile through LaMa. tileImg / maskTileImg are 512² ImageData.
  // Returns a Float32Array (CHW, 0..1) of the model output.
  async function runTile(tileImg, maskTileImg) {
    const ort = window.ort;
    const plane = TILE * TILE;
    const imgF = new Float32Array(3 * plane);
    const mskF = new Float32Array(plane);
    for (let i = 0; i < plane; i++) {
      imgF[i]           = tileImg.data[i * 4]     / 255;
      imgF[plane + i]   = tileImg.data[i * 4 + 1] / 255;
      imgF[plane*2 + i] = tileImg.data[i * 4 + 2] / 255;
      mskF[i]           = maskTileImg.data[i * 4] / 255;
    }
    const imgTensor = new ort.Tensor('float32', imgF, [1, 3, TILE, TILE]);
    const mskTensor = new ort.Tensor('float32', mskF, [1, 1, TILE, TILE]);
    let outputs;
    try { outputs = await session.run({ image: imgTensor, mask: mskTensor }); }
    catch (e) {
      // Some exports name the image input "img" instead of "image".
      try { outputs = await session.run({ img: imgTensor, mask: mskTensor }); }
      catch (_) { throw new Error('LaMa 模型输入接口不匹配: ' + e.message); }
    }
    const out = outputs[Object.keys(outputs)[0]].data;
    // Some LaMa exports emit 0..255 rather than 0..1 — detect and normalise.
    let mx = 0;
    for (let i = 0; i < out.length; i += 997) if (out[i] > mx) mx = out[i];
    if (mx > 1.5) for (let i = 0; i < out.length; i++) out[i] /= 255;
    return out;
  }

  // Inpaint `imageData` in-place over `mask` (1 = remove). Returns a new ImageData.
  async function inpaint(imageData, mask, opts = {}) {
    if (!session) throw new Error('LaMa 未初始化, 先调用 LaMa.prepare()');
    const { onProgress = () => {}, onLog = () => {} } = opts;
    const ort = window.ort;
    if (!ort) throw new Error('onnxruntime-web 未就绪');
    const { width: w, height: h } = imageData;

    // 1. mask bounding box
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        if (mask[row + x]) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) return new ImageData(new Uint8ClampedArray(imageData.data), w, h);

    // 2. tile grid over bbox + padding, clamped to the image
    const startX = Math.max(0, minX - TILE_PAD);
    const startY = Math.max(0, minY - TILE_PAD);
    const endX = Math.min(w, maxX + TILE_PAD + 1);
    const endY = Math.min(h, maxY + TILE_PAD + 1);

    const tiles = [];
    const seen = new Set();
    for (let ty = startY; ty < endY; ty += TILE_STRIDE) {
      for (let tx = startX; tx < endX; tx += TILE_STRIDE) {
        let x0 = tx, y0 = ty;
        let x1 = Math.min(w, x0 + TILE);
        let y1 = Math.min(h, y0 + TILE);
        if (x1 - x0 < TILE && w >= TILE) x0 = Math.max(0, x1 - TILE);
        if (y1 - y0 < TILE && h >= TILE) y0 = Math.max(0, y1 - TILE);
        const key = x0 + ',' + y0;
        if (seen.has(key)) continue;
        seen.add(key);
        let has = false;
        for (let yy = y0; yy < y1 && !has; yy++) {
          const r = yy * w;
          for (let xx = x0; xx < x1; xx++) if (mask[r + xx]) { has = true; break; }
        }
        if (!has) continue;
        tiles.push({ x0, y0, x1, y1 });
        if (x1 >= w) break;
      }
    }

    onLog?.(`mask bbox ${maxX-minX+1}×${maxY-minY+1} @ (${minX},${minY}) · ${tiles.length} 块推理`);

    const result = new Uint8ClampedArray(imageData.data);
    const srcCv = newCanvas(w, h);
    srcCv.getContext('2d').putImageData(imageData, 0, 0);

    for (let ti = 0; ti < tiles.length; ti++) {
      const { x0, y0, x1, y1 } = tiles[ti];
      const tw = x1 - x0, th = y1 - y0;
      onProgress(ti / tiles.length);

      // crop the source tile and scale to 512²
      const inCv = newCanvas(TILE, TILE);
      const ictx = inCv.getContext('2d');
      ictx.imageSmoothingQuality = 'high';
      ictx.drawImage(srcCv, x0, y0, tw, th, 0, 0, TILE, TILE);
      const tileImg = ictx.getImageData(0, 0, TILE, TILE);

      // build the matching 512² mask
      const maskLocal = new ImageData(tw, th);
      for (let yy = 0; yy < th; yy++) {
        const sr = (y0 + yy) * w, dr = yy * tw;
        for (let xx = 0; xx < tw; xx++) {
          const v = mask[sr + (x0 + xx)] ? 255 : 0;
          const pi = (dr + xx) * 4;
          maskLocal.data[pi] = v; maskLocal.data[pi+1] = v; maskLocal.data[pi+2] = v; maskLocal.data[pi+3] = 255;
        }
      }
      const mlCv = newCanvas(tw, th);
      mlCv.getContext('2d').putImageData(maskLocal, 0, 0);
      const mtCv = newCanvas(TILE, TILE);
      const mctx = mtCv.getContext('2d');
      mctx.imageSmoothingEnabled = false;
      mctx.drawImage(mlCv, 0, 0, TILE, TILE);
      const maskTileImg = mctx.getImageData(0, 0, TILE, TILE);

      const od = await runTile(tileImg, maskTileImg);

      // model output 512² → ImageData
      const plane = TILE * TILE;
      const outTileCv = newCanvas(TILE, TILE);
      const outImg = outTileCv.getContext('2d').createImageData(TILE, TILE);
      for (let i = 0; i < plane; i++) {
        outImg.data[i*4]   = Math.max(0, Math.min(255, Math.round(od[i] * 255)));
        outImg.data[i*4+1] = Math.max(0, Math.min(255, Math.round(od[plane + i] * 255)));
        outImg.data[i*4+2] = Math.max(0, Math.min(255, Math.round(od[plane*2 + i] * 255)));
        outImg.data[i*4+3] = 255;
      }
      outTileCv.getContext('2d').putImageData(outImg, 0, 0);

      // scale tile output back to source resolution
      const scaledCv = newCanvas(tw, th);
      const sctx = scaledCv.getContext('2d');
      sctx.imageSmoothingQuality = 'high';
      sctx.drawImage(outTileCv, 0, 0, tw, th);
      const scaled = sctx.getImageData(0, 0, tw, th);

      // composite ONLY where masked (tile overlaps never double-paint kept pixels)
      for (let yy = 0; yy < th; yy++) {
        const sr = (y0 + yy) * w, lr = yy * tw;
        for (let xx = 0; xx < tw; xx++) {
          const gi = sr + (x0 + xx);
          if (!mask[gi]) continue;
          const lp = (lr + xx) * 4, gp = gi * 4;
          result[gp]   = scaled.data[lp];
          result[gp+1] = scaled.data[lp+1];
          result[gp+2] = scaled.data[lp+2];
        }
      }
      // yield so the UI can paint progress
      await new Promise(r => setTimeout(r, 0));
    }
    onProgress(1);
    return new ImageData(result, w, h);
  }

  window.LaMa = { prepare, inpaint, isReady, dispose, MODEL_ID };
})();
