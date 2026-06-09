// Robust Video Matting (RVM) portrait matting via TensorFlow.js (WebGPU / WebGL).
//
// RVM is SOTA for human/portrait matting (fine hair detail). We use it for
// single images here. It's a RECURRENT model — inputs are the source frame plus
// four recurrent-state tensors (zeros on the first/only frame) and a
// downsample_ratio that controls the internal working resolution (keeps memory
// bounded, unlike fixed-1024² models that blow the WebGPU buffer limit).
//
// Why TF.js and not onnxruntime-web: RVM's ONNX uses AveragePool with ceil_mode,
// which ort-web's graph layer rejects. TF.js runs it fine.
//
// Verified model I/O (rvm_mobilenetv3_tfjs_int8, clean-room):
//   inputs : src [1,H,W,3] 0..1 RGB NHWC · r1i..r4i [1,1,1,1] zeros · downsample_ratio scalar
//   outputs: fgr, pha [1,H,W,1] 0..1 (we use pha = alpha matte), r1o..r4o
//
// Public API:
//   await RVM.prepare({ onStatus, onProgress })  → { backend }
//   await RVM.matte(srcCanvas)  → canvas (W×H, RGBA, A = alpha) — same format as
//        the tool's tensorToCanvas output, so it drops into applyMask() unchanged
//   RVM.isReady() / RVM.backend() / RVM.dispose()

(function () {
  const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
  const TFJS_WEBGPU = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu@4.22.0/dist/tf-backend-webgpu.min.js';
  const MODEL_URL = '../models/rvm-tfjs/rvm_mobilenetv3_tfjs_int8/model.json';

  let tf = null, model = null, backendName = null, preparing = null;

  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script');
      s.src = src; s.onload = () => res(); s.onerror = () => rej(new Error('加载失败: ' + src));
      document.head.appendChild(s);
    });
  }

  function newCanvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

  async function prepare(opts = {}) {
    const { onStatus = () => {}, onProgress = () => {} } = opts;
    if (model) return { backend: backendName };
    if (preparing) return preparing;
    preparing = (async () => {
      if (!window.tf) {
        onStatus('加载 TensorFlow.js (首次联网 ~1 MB)...');
        await loadScript(TFJS);
        await loadScript(TFJS_WEBGPU);
      }
      tf = window.tf;
      onStatus('初始化 GPU 后端...');
      let ok = false;
      try { ok = await tf.setBackend('webgpu'); } catch (_) { ok = false; }
      if (ok) { await tf.ready(); backendName = 'webgpu'; }
      else { await tf.setBackend('webgl'); await tf.ready(); backendName = 'webgl'; }
      onStatus('加载 RVM 模型 (~4 MB)...');
      const cacheKey = 'indexeddb://rvm-mobilenetv3';
      try { model = await tf.loadGraphModel(cacheKey); }
      catch (_) {
        const url = new URL(MODEL_URL, location.href).href;
        model = await tf.loadGraphModel(url, { onProgress: p => onProgress(p) });
        try { await model.save(cacheKey); } catch (_) {}
      }
      onStatus(`RVM 就绪 (${backendName})`);
      return { backend: backendName };
    })().catch(e => { preparing = null; throw e; });
    const r = await preparing; preparing = null; return r;
  }

  function isReady() { return !!model; }
  function backend() { return backendName; }
  function dispose() { try { model?.dispose?.(); } catch (_) {} model = null; }

  // src: a canvas (or ImageBitmap). Returns a W×H canvas whose ALPHA channel is
  // the matte (RGB left at 255), matching the tool's mask-canvas convention.
  async function matte(src) {
    if (!model) throw new Error('RVM 未初始化, 先调用 RVM.prepare()');
    const W = src.width, H = src.height;
    // Downsample so the model's internal longer side ≈ 512 — keeps quality high
    // while bounding compute/memory. Clamp to RVM's sane range.
    const ratio = Math.max(0.25, Math.min(1, 512 / Math.max(W, H)));

    const srcT = tf.tidy(() => tf.browser.fromPixels(src).toFloat().div(255).expandDims(0)); // [1,H,W,3]
    const z = tf.zeros([1, 1, 1, 1]);
    const dr = tf.scalar(ratio);
    let pha;
    try {
      const out = await model.executeAsync(
        { src: srcT, r1i: z, r2i: z, r3i: z, r4i: z, downsample_ratio: dr },
        ['pha']
      );
      pha = Array.isArray(out) ? out[0] : out;
      const data = await pha.data();           // Float32 [H*W], 0..1
      const cv = newCanvas(W, H);
      const ctx = cv.getContext('2d');
      const img = ctx.createImageData(W, H);
      const d = img.data;
      for (let i = 0; i < W * H; i++) {
        let v = data[i]; v = v < 0 ? 0 : v > 1 ? 1 : v;
        d[i*4] = 255; d[i*4+1] = 255; d[i*4+2] = 255; d[i*4+3] = Math.round(v * 255);
      }
      ctx.putImageData(img, 0, 0);
      return cv;
    } finally {
      try { srcT.dispose(); z.dispose(); dr.dispose(); if (pha && pha.dispose) pha.dispose(); } catch (_) {}
    }
  }

  window.RVM = { prepare, matte, isReady, backend, dispose };
})();
