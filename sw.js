// Dobby service worker — online-first with offline fallback.
// Strategy:
//   Same-origin (HTML + assets) → network-first (always try fresh, fall back to cache offline)
//   Cross-origin (esm.sh / fonts / huggingface CDNs) → pass through, never cached here
// Rationale: SWR for assets used to bite users on every deploy (copy/UI changes only
// showed after a SECOND refresh). Network-first costs one extra round-trip per request
// online but matches "push = next refresh shows it" behavior of a normal website.

const CACHE = 'dobby-v130';
const NETWORK_TIMEOUT_MS = 7000;
const IS_CLOUDFLARE_PAGES = /\.pages\.dev$/i.test(self.location.hostname);

// Pre-cache the core shell + every tool page so the site works offline immediately.
// Pre-cache only the SHELL (home + core assets + most-likely-first 12 tools).
// Other tool pages are network-first by default and end up in cache lazily
// as the user navigates to them. This change cut the pre-cache from 49 files
// to 19 so first-visit doesn't bottleneck on background SW install fetches
// (especially noticeable on China-mobile networks where Cloudflare's edge
// can be slow).
const CORE_FILES = [
  './',
  './index.html',
  './manifest.json',
  './assets/shared.css',
  './assets/shared.js',
  './assets/i18n-strings.js',
  './assets/icon.svg',
  // Popular landing tools (best-guess; revisit when analytics arrives)
  './tools/image-optimizer.html',
  './tools/png-crusher.html',
  './tools/ai-cutout.html',
  './tools/gif-tools.html',
  './tools/video-toolkit.html',
  './tools/screen-recorder.html',
  './tools/transcode.html',
  './tools/pdf-tools.html',
  './tools/file-compress.html',
  './tools/qr-gen.html',
  './tools/base64.html',
  './tools/json-tools.html'
];

// Match the URLs emitted by shared.js. Localhost and GitHub Pages still need
// physical `.html` paths; Cloudflare Pages serves the same files canonically
// without the suffix.
const CORE = CORE_FILES.map(path => (
  IS_CLOUDFLARE_PAGES && path.startsWith('./tools/')
    ? path.replace(/\.html$/i, '')
    : path
));

function toolPageAliases(request) {
  const url = new URL(request.url);
  if (!/\/tools\/[^/.]+(?:\.html)?$/i.test(url.pathname)) return [];

  const alias = new URL(url.href);
  if (/\.html$/i.test(alias.pathname)) alias.pathname = alias.pathname.replace(/\.html$/i, '');
  else alias.pathname += '.html';
  return [alias.href];
}

function fetchWithTimeout(request) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), NETWORK_TIMEOUT_MS);
  return fetch(request, { signal: controller.signal })
    .finally(() => clearTimeout(timer));
}

async function cachedFallback(request, isHtml) {
  const exact = await caches.match(request);
  if (exact) return exact;

  for (const alias of toolPageAliases(request)) {
    const cached = await caches.match(alias);
    if (cached) return cached;
  }

  return isHtml ? caches.match('./index.html') : undefined;
}

self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(CORE).catch(err => {
      console.warn('[sw] pre-cache partial failure:', err);
    }))
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(k => k !== CACHE).map(k => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // Don't intercept cross-origin (transformers.js, esm.sh, hugging face, fonts).
  if (url.origin !== location.origin) return;

  const isHtml = req.headers.get('accept')?.includes('text/html')
                 || /\.html?$/i.test(url.pathname)
                 || url.pathname.endsWith('/');

  // Network-first for ALL same-origin requests. Online: always fresh. Offline:
  // fall back to whatever's cached (HTML navigations fall back to index.html
  // so the SPA shell still loads).
  let cacheTask = Promise.resolve();
  // Only navigation documents get a deadline. Large same-origin models and
  // media may legitimately take longer than seven seconds on slower links.
  const networkTask = isHtml ? fetchWithTimeout(req) : fetch(req);
  const responseTask = networkTask.then(resp => {
      if (resp.ok) {
        const keys = [req.url, ...toolPageAliases(req)];
        cacheTask = caches.open(CACHE).then(c => Promise.all(
          keys.map(key => c.put(key, resp.clone()))
        ));
      } else if (resp.status === 404) {
        // server says gone — drop any stale cached copy so the next visit doesn't resurrect it
        const keys = [req.url, ...toolPageAliases(req)];
        cacheTask = caches.open(CACHE).then(c => Promise.all(keys.map(key => c.delete(key))));
      }
      return resp;
    }).catch(async () => (await cachedFallback(req, isHtml)) || Response.error());

  e.respondWith(responseTask);
  // Register the lifetime extension synchronously; waitUntil() can reject calls
  // made only after an asynchronous fetch has already settled.
  e.waitUntil(responseTask.then(
    () => cacheTask,
    () => cacheTask
  ).catch(err => console.warn('[sw] runtime cache failed:', err)));
});

// Allow page to trigger a manual cache update / reset.
self.addEventListener('message', (e) => {
  if (e.data === 'reset-cache') {
    caches.keys().then(keys => Promise.all(keys.map(k => caches.delete(k))));
  } else if (e.data === 'skip-waiting') {
    self.skipWaiting();
  }
});
