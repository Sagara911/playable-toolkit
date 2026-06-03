// code-minify worker — runs best-in-class minifiers off the main thread so big
// files don't freeze the page. Loaded as a module worker: { type: 'module' }.
//   JS   → terser                 (industry standard)
//   CSS  → csso                   (collapse + shorthand + rule merging)
//   HTML → html-minifier-terser   (smart collapse + inline JS via terser + inline CSS via clean-css)
// All lazy-loaded from esm.sh on first use, then cached for the session.

let terserMod = null, cssoMod = null, htmlMinMod = null;

async function loadTerser(post) {
  if (terserMod) return;
  post({ type: 'progress', msg: '加载 terser (首次联网 ~250KB)...' });
  terserMod = await import('https://esm.sh/terser@5.36.0');
}
async function loadCsso(post) {
  if (cssoMod) return;
  post({ type: 'progress', msg: '加载 csso (首次联网)...' });
  const m = await import('https://esm.sh/csso@5.0.5');
  cssoMod = m.minify ? m : (m.default || m);
}
async function loadHtmlMin(post) {
  if (htmlMinMod) return;
  post({ type: 'progress', msg: '加载 html-minifier-terser (首次联网,含 terser+clean-css)...' });
  const m = await import('https://esm.sh/html-minifier-terser@7.2.0');
  htmlMinMod = { minify: m.minify || (m.default && m.default.minify) };
}

async function minifyCss(src, opts, post) {
  await loadCsso(post);
  // csso always collapses whitespace + shortens values + merges rules; the only
  // toggle we still honor is whether to strip all comments.
  const r = cssoMod.minify(src, {
    comments: (opts.removeComments === false) ? 'exclamation' : false,
    restructure: opts.restructure !== false
  });
  return r.css;
}

async function minifyJs(src, opts, post) {
  await loadTerser(post);
  const result = await terserMod.minify(src, {
    compress: opts.compress ? { passes: 2, drop_console: false } : false,
    mangle: !!opts.mangle,
    format: { comments: false }
  });
  if (result.error) throw new Error('Terser: ' + result.error.message);
  return result.code;
}

async function minifyHtml(src, opts, post) {
  await loadHtmlMin(post);
  const inline = opts.minifyInline !== false;
  return await htmlMinMod.minify(src, {
    collapseWhitespace: opts.collapseSpace !== false,
    conservativeCollapse: false,
    removeComments: opts.removeComments !== false,
    minifyCSS: inline,            // inline <style> + style="" via clean-css
    minifyJS: inline,             // inline <script> via terser
    removeRedundantAttributes: true,
    removeScriptTypeAttributes: true,
    removeStyleLinkTypeAttributes: true,
    collapseBooleanAttributes: true,
    useShortDoctype: true,
    keepClosingSlash: false,
    removeEmptyAttributes: true,
    sortAttributes: true,
    sortClassName: true
  });
}

self.addEventListener('message', async (e) => {
  const { id, lang, src, options } = e.data;
  const post = (msg) => self.postMessage({ id, ...msg });
  try {
    let code;
    if (lang === 'css') code = await minifyCss(src, options, post);
    else if (lang === 'js') code = await minifyJs(src, options, post);
    else if (lang === 'html') code = await minifyHtml(src, options, post);
    else throw new Error('Unknown lang: ' + lang);
    self.postMessage({ id, type: 'done', code });
  } catch (err) {
    self.postMessage({ id, type: 'error', error: err.message || String(err) });
  }
});
