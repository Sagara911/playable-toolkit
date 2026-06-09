// Segment Anything (SlimSAM) — click-to-segment via transformers.js.
//
// Unlike "find the main subject" matting models (RVM=human, RMBG=salient object)
// that GUESS what to cut, SAM segments exactly what you point at — far better for
// pulling a specific character out of a busy scene. SlimSAM-77 is the
// browser-sized variant the transformers.js SAM demo ships.
//
// Flow: encode the image ONCE (heavy), then each click only runs the light mask
// decoder. Foreground points (label 1) include, background points (label 0)
// exclude. Returns the highest-IoU of the 3 candidate masks.
//
// Public API:
//   await SAM.prepare({ onStatus, onProgress })
//   await SAM.setImage(canvas, { onStatus })      // encode + cache embeddings
//   await SAM.segment(points)  // points: [{x,y,fg}] in IMAGE px → { maskCanvas, score } | null
//   SAM.hasImage() / SAM.isReady() / SAM.reset() / SAM.dispose()

(function () {
  const MODEL_ID = 'Xenova/slimsam-77-uniform';
  const TRANSFORMERS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2/dist/transformers.min.js';

  let tf = null;                 // transformers module
  let model = null, processor = null, RawImage = null;
  let preparing = null;
  let cur = null;                // { image, embeddings, original_sizes, reshaped_input_sizes }

  function newCanvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }

  async function prepare(opts = {}) {
    const { onStatus = () => {}, onProgress = () => {} } = opts;
    if (model && processor) return true;
    if (preparing) return preparing;
    preparing = (async () => {
      onStatus('加载 transformers.js...');
      tf = await import(TRANSFORMERS);
      tf.env.allowLocalModels = false;
      RawImage = tf.RawImage;
      onStatus('加载 SlimSAM 模型 (首次联网)...');
      model = await tf.SamModel.from_pretrained(MODEL_ID, { progress_callback: p => onProgress(p) });
      processor = await tf.AutoProcessor.from_pretrained(MODEL_ID);
      onStatus('SAM 就绪');
      return true;
    })().catch(e => { preparing = null; throw e; });
    const r = await preparing; preparing = null; return r;
  }

  function isReady() { return !!(model && processor); }
  function hasImage() { return !!cur; }
  function reset() { cur = null; }
  function dispose() { try { model?.dispose?.(); } catch (_) {} model = null; processor = null; cur = null; }

  // Encode the image once and cache its embeddings for fast per-click decoding.
  async function setImage(canvas, opts = {}) {
    const { onStatus = () => {} } = opts;
    if (!model) throw new Error('SAM 未初始化');
    onStatus('分析图片(编码,只需一次)...');
    const url = URL.createObjectURL(await new Promise(r => canvas.toBlob(r, 'image/png')));
    let image;
    try { image = await RawImage.read(url); } finally { URL.revokeObjectURL(url); }
    const image_inputs = await processor(image);
    const embeddings = await model.get_image_embeddings(image_inputs);
    cur = {
      image,
      embeddings,
      original_sizes: image_inputs.original_sizes,
      reshaped_input_sizes: image_inputs.reshaped_input_sizes,
      W: canvas.width, H: canvas.height
    };
    onStatus('已就绪 · 点击主体');
  }

  // points: [{x,y,fg}] in IMAGE pixel coords. Returns the best mask as a canvas
  // (RGB=255, A=mask*255) — same convention the tool's applyMask() expects.
  async function segment(points) {
    if (!cur) throw new Error('SAM 还没设置图片, 先调用 setImage()');
    if (!points || !points.length) return null;
    const input_points = [[points.map(p => [p.x, p.y])]];
    const input_labels = [[points.map(p => (p.fg === false ? 0 : 1))]];
    const inputs = await processor(cur.image, { input_points, input_labels });
    const outputs = await model({
      ...cur.embeddings,
      input_points: inputs.input_points,
      input_labels: inputs.input_labels
    });
    const masks = await processor.post_process_masks(outputs.pred_masks, cur.original_sizes, cur.reshaped_input_sizes);
    const maskT = masks[0];                 // [1, nMasks, H, W] (bool)
    const dims = maskT.dims;
    const nMasks = dims[1], H = dims[2], W = dims[3];
    // pick highest-IoU candidate
    const iou = outputs.iou_scores.data;    // length nMasks
    let best = 0; for (let i = 1; i < nMasks; i++) if (iou[i] > iou[best]) best = i;
    const data = maskT.data;                // bool/uint8, length nMasks*H*W
    const off = best * H * W;
    const cv = newCanvas(W, H);
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(W, H);
    const d = img.data;
    for (let i = 0; i < W * H; i++) {
      const a = data[off + i] ? 255 : 0;
      d[i*4] = 255; d[i*4+1] = 255; d[i*4+2] = 255; d[i*4+3] = a;
    }
    ctx.putImageData(img, 0, 0);
    return { maskCanvas: cv, score: +iou[best] };
  }

  window.SAM = { prepare, setImage, segment, isReady, hasImage, reset, dispose };
})();
