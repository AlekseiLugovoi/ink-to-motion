"""Camera widget: JS, HTML builder, and CSS for the Gradio app."""

CAMERA_HEAD = """
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<script>
(function() {
  function getTextboxNode(elemId) {
    var root = document.getElementById(elemId);
    if (!root) return null;
    return root.querySelector('textarea, input');
  }

  function setTextboxValue(elemId, value) {
    var node = getTextboxNode(elemId);
    if (!node) return false;
    var setter = null;
    if (node instanceof HTMLTextAreaElement) {
      setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value');
    } else {
      setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
    }
    if (!setter || !setter.set) return false;
    try { setter.set.call(node, value); } catch (e) {
      try { node.value = value; } catch (e2) { return false; }
    }
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  }

  function normalizeTemplateUrl(path) {
    if (!path) return '';
    if (/^(https?:|data:|blob:|[\\/])/i.test(path)) return path;
    return '/gradio_api/file=' + path.replace(/^[.]*[\\/]/, '');
  }

  async function getCameraStream(isMobile) {
    var tries = [];
    if (isMobile) {
      tries.push({ audio: false, video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: { ideal: 'environment' } } });
    } else {
      try {
        var devices = await navigator.mediaDevices.enumerateDevices();
        var cam = devices.find(function(d) { return d.kind === 'videoinput'; });
        if (cam && cam.deviceId) {
          tries.push({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, deviceId: { exact: cam.deviceId } } });
        }
      } catch (e) {}
      tries.push({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 } } });
    }
    tries.push({ audio: false, video: true });
    var lastErr = null;
    for (var i = 0; i < tries.length; i++) {
      try { return await navigator.mediaDevices.getUserMedia(tries[i]); } catch (e) { lastErr = e; }
    }
    throw lastErr || new Error('Camera stream unavailable');
  }

  function openNativeCameraAndLoad(widget, statusEl) {
    var targetId = widget.dataset.target || 'camera-data';
    return new Promise(function(resolve) {
      var input = document.createElement('input');
      input.type = 'file'; input.accept = 'image/*'; input.setAttribute('capture', 'environment');
      input.style.display = 'none'; document.body.appendChild(input);
      input.onchange = function(e) {
        var file = e.target.files && e.target.files[0];
        if (!file) { document.body.removeChild(input); resolve(false); return; }
        var reader = new FileReader();
        reader.onload = function(ev) { setTextboxValue(targetId, ev.target.result); document.body.removeChild(input); resolve(true); };
        reader.onerror = function() { document.body.removeChild(input); resolve(false); };
        reader.readAsDataURL(file);
      };
      input.click();
    });
  }

  function waitForVideoReady(video, timeoutMs) {
    return new Promise(function(resolve) {
      if (video.videoWidth > 0 && video.videoHeight > 0) { resolve(true); return; }
      var done = false;
      var timer = setTimeout(function() { if (!done) { done = true; resolve(false); } }, timeoutMs || 2500);
      function onReady() { if (!done && video.videoWidth > 0) { done = true; clearTimeout(timer); resolve(true); } }
      video.addEventListener('loadeddata', onReady);
    });
  }

  function stopCamera(widget) {
    if (!widget) return;
    stopDetection(widget);
    if (widget._cameraStream) { widget._cameraStream.getTracks().forEach(function(t) { t.stop(); }); widget._cameraStream = null; }
  }

  function closeModal(widget) {
    var stage = widget.querySelector('[data-role="camera-stage"]');
    var snapBtn = widget.querySelector('[data-role="camera-snap"]');
    var closeBtn = widget.querySelector('[data-role="camera-close"]');
    stopCamera(widget);
    stage.hidden = true;
    snapBtn.disabled = true;
    closeBtn.disabled = true;
    try { if (document.exitFullscreen && document.fullscreenElement) document.exitFullscreen(); } catch(e) {}
  }

  async function startCamera(widget) {
    var video = widget.querySelector('[data-role="camera-video"]');
    var status = widget.querySelector('[data-role="camera-status"]');
    var stage = widget.querySelector('[data-role="camera-stage"]');
    var snapBtn = widget.querySelector('[data-role="camera-snap"]');
    var closeBtn = widget.querySelector('[data-role="camera-close"]');
    var isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    closeModal(widget);
    stage.hidden = false;
    status.textContent = '';
    try { var el = document.documentElement; if (el.requestFullscreen) el.requestFullscreen(); } catch(e) {}
    try {
      var stream = await getCameraStream(isMobile);
      widget._cameraStream = stream;
      video.srcObject = stream;
      snapBtn.disabled = false;
      closeBtn.disabled = false;
      try { await video.play(); } catch (e) {}
      var ready = await waitForVideoReady(video, 2500);
      if (!ready && isMobile) {
        closeModal(widget);
        var loaded = await openNativeCameraAndLoad(widget, status);
        if (loaded) status.textContent = '\\u0424\\u043e\\u0442\\u043e \\u0437\\u0430\\u0433\\u0440\\u0443\\u0436\\u0435\\u043d\\u043e.';
        return;
      }
      startDetection(widget);
    } catch (err) {
      closeModal(widget);
      status.textContent = '\\u041d\\u0435 \\u0443\\u0434\\u0430\\u043b\\u043e\\u0441\\u044c \\u043e\\u0442\\u043a\\u0440\\u044b\\u0442\\u044c \\u043a\\u0430\\u043c\\u0435\\u0440\\u0443. \\u041f\\u0440\\u043e\\u0432\\u0435\\u0440\\u044c \\u0440\\u0430\\u0437\\u0440\\u0435\\u0448\\u0435\\u043d\\u0438\\u0435 \\u0431\\u0440\\u0430\\u0443\\u0437\\u0435\\u0440\\u0430 \\u0438 https.';
    }
  }

  function capturePhoto(widget) {
    var video = widget.querySelector('[data-role="camera-video"]');
    var status = widget.querySelector('[data-role="camera-status"]');
    var w = video.videoWidth || 1280, h = video.videoHeight || 720;
    if (!w || !h) { status.textContent = 'Capture failed.'; return; }
    var cx = Math.round(w * 0.22), cy = Math.round(h * 0.08);
    var cw = Math.round(w * 0.56), ch = Math.round(h * 0.84);
    var fullCanvas = document.createElement('canvas');
    fullCanvas.width = w; fullCanvas.height = h;
    try { fullCanvas.getContext('2d').drawImage(video, 0, 0, w, h); } catch (e) { return; }
    var canvas = document.createElement('canvas');
    canvas.width = cw; canvas.height = ch;
    canvas.getContext('2d').drawImage(fullCanvas, cx, cy, cw, ch, 0, 0, cw, ch);
    var dataUrl = canvas.toDataURL('image/jpeg', 0.92);
    var targetId = widget.dataset.target || 'camera-data';
    setTextboxValue(targetId, dataUrl);
    closeModal(widget);
  }

  function bindCameraWidget(widget) {
    if (!widget || widget.dataset.bound === '1') return;
    widget.dataset.bound = '1';
    var openBtn = widget.querySelector('[data-role="camera-open"]');
    if (openBtn) openBtn.addEventListener('click', function() { startCamera(widget); });
    widget.querySelector('[data-role="camera-snap"]').addEventListener('click', function() { capturePhoto(widget); });
    widget.querySelector('[data-role="camera-close"]').addEventListener('click', function() { closeModal(widget); });
    widget.querySelector('[data-role="camera-stage"]').addEventListener('click', function(e) {
      if (e.target.dataset.role === 'camera-stage') closeModal(widget);
    });
    window.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(widget); });
    window.addEventListener('beforeunload', function() { stopCamera(widget); });
  }

  function setFrameColor(widget, found) {
    var rects = widget.querySelectorAll('.camera-frame-overlay svg rect[stroke]');
    var color = found ? 'rgba(34,197,94,0.95)' : 'rgba(180,180,180,0.8)';
    rects.forEach(function(r) { r.setAttribute('stroke', color); });
  }

  function startDetection(widget) {
    var video = widget.querySelector('[data-role="camera-video"]');
    var status = widget.querySelector('[data-role="camera-status"]');
    var stage = widget.querySelector('[data-role="camera-stage"]');
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    widget._detectRunning = true;
    widget._detectInflight = false;
    async function loop() {
      while (widget._detectRunning) {
        await new Promise(function(r) { setTimeout(r, 1200); });
        if (!widget._detectRunning || stage.hidden || !video.videoWidth || widget._detectInflight) continue;
        canvas.width = 192;
        canvas.height = Math.max(108, Math.round(video.videoHeight * 192 / video.videoWidth));
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        try {
          widget._detectInflight = true;
          var resp = await fetch('/api/detect_aruco', { method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({frame: canvas.toDataURL('image/jpeg', 0.35)}) });
          if (resp.ok) {
            var data = await resp.json();
            if (widget._detectRunning) {
              var count = Number(data.count || 0);
              setFrameColor(widget, count >= 1);
              if (status) status.textContent = count >= 1 ? ('ArUco: ' + count) : '';
            }
          }
        } catch(e) {} finally { widget._detectInflight = false; }
      }
    }
    loop();
  }

  function stopDetection(widget) { widget._detectRunning = false; widget._detectInflight = false; setFrameColor(widget, false); }

  function bindAllCameraWidgets() { document.querySelectorAll('.camera-widget').forEach(bindCameraWidget); }
  window.addEventListener('load', bindAllCameraWidgets);
  new MutationObserver(bindAllCameraWidgets).observe(document.documentElement, { childList: true, subtree: true });

  // --- Auto mode: ?char=XXX opens camera immediately ---
  var params = new URLSearchParams(window.location.search);
  var autoChar = params.get('char');
  if (autoChar) {
    window.__inkAutoChar = autoChar;
    (function waitForAutoWidget() {
      function tryOpen() {
        var w = document.querySelector('.camera-widget[data-target="camera-data-auto"]');
        if (w && w.dataset.bound === '1') { startCamera(w); return true; }
        return false;
      }
      if (tryOpen()) return;
      new MutationObserver(function(_, obs) { if (tryOpen()) obs.disconnect(); })
        .observe(document.documentElement, { childList: true, subtree: true });
    })();
  }

  // --- QR code generation ---
  var _qrScriptLoaded = false;
  function ensureQRScript() {
    return new Promise(function(resolve) {
      if (_qrScriptLoaded || window.qrcode) { _qrScriptLoaded = true; resolve(); return; }
      var s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/qrcode-generator@1.4.4/qrcode.min.js';
      s.onload = function() { _qrScriptLoaded = true; resolve(); };
      document.head.appendChild(s);
    });
  }

  function renderQR(canvas, url, size) {
    var qr = qrcode(0, 'M');
    qr.addData(url);
    qr.make();
    var modules = qr.getModuleCount();
    var cellSize = Math.floor(size / modules);
    canvas.width = cellSize * modules;
    canvas.height = cellSize * modules;
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#000';
    for (var r = 0; r < modules; r++)
      for (var c = 0; c < modules; c++)
        if (qr.isDark(r, c)) ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
  }

  var _qrGenerating = false;
  function generateAllQR() {
    if (_qrGenerating) return;
    var pending = document.querySelectorAll('[data-qr-char]:not([data-qr-done])');
    if (!pending.length) return;
    _qrGenerating = true;
    ensureQRScript().then(function() {
      document.querySelectorAll('[data-qr-char]:not([data-qr-done])').forEach(function(el) {
        el.dataset.qrDone = '1';
        var charId = el.dataset.qrChar;
        var url = window.location.origin + window.location.pathname + '?char=' + charId;
        var canvas = document.createElement('canvas');
        renderQR(canvas, url, 140);
        el.style.display = 'flex';
        el.style.justifyContent = 'center';
        el.style.alignItems = 'center';
        canvas.style.display = 'block';
        el.innerHTML = '';
        el.appendChild(canvas);
      });
      _qrGenerating = false;
    });
  }

  window.addEventListener('load', generateAllQR);
  new MutationObserver(function() { generateAllQR(); }).observe(document.documentElement, { childList: true, subtree: true });

  // --- Set char_id from URL into Gradio textbox ---
  if (autoChar) {
    (function waitForCharBox() {
      function trySet() { return setTextboxValue('auto-char-id', autoChar); }
      if (trySet()) return;
      new MutationObserver(function(_, obs) { if (trySet()) obs.disconnect(); })
        .observe(document.documentElement, { childList: true, subtree: true });
    })();
  }
})();
</script>
"""


def make_camera_panel(data_target="camera-data"):
    mid = data_target.replace("-", "_")
    return f"""
<div class="camera-widget" data-target="{data_target}">
  <div class="camera-status" data-role="camera-status"></div>
  <div class="camera-stage" data-role="camera-stage" hidden>
    <div class="camera-stage-inner">
      <div class="camera-frame">
        <video data-role="camera-video" autoplay playsinline muted></video>
        <div class="camera-frame-overlay">
          <svg viewBox="0 0 100 100" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <mask id="fmask_{mid}">
                <rect width="100" height="100" fill="white"/>
                <rect x="22" y="8" width="56" height="84" rx="3" ry="3" fill="black"/>
              </mask>
            </defs>
            <rect width="100" height="100" fill="rgba(0,0,0,0.5)" mask="url(#fmask_{mid})"/>
            <rect x="22" y="8" width="56" height="84" rx="3" ry="3"
                  fill="none" stroke="rgba(180,180,180,0.8)" stroke-width="0.4"/>
          </svg>
        </div>
      </div>
      <button type="button" class="camera-close-x" data-role="camera-close" disabled>&#10005;</button>
      <div class="camera-hint">\u041f\u043e\u043c\u0435\u0441\u0442\u0438 \u0440\u0438\u0441\u0443\u043d\u043e\u043a \u0432 \u0440\u0430\u043c\u043a\u0443 \u0438 \u0441\u0434\u0435\u043b\u0430\u0439 \u0441\u043d\u0438\u043c\u043e\u043a</div>
      <button type="button" class="camera-shutter" data-role="camera-snap" disabled><span></span></button>
    </div>
  </div>
</div>
"""


CSS = """
#main-wrap { max-width: 640px !important; margin: auto !important; }
#camera-data, #camera-data-auto, #auto-char-id { display: none !important; }
.camera-widget { display: grid; gap: 12px; }
.camera-widget .camera-status { font-size: 13px; color: #4b5563; }
.camera-widget .camera-stage {
  position: fixed; inset: 0; z-index: 9999; background: #000;
  display: none; align-items: center; justify-content: center;
}
.camera-widget .camera-stage:not([hidden]) { display: flex; }
.camera-widget .camera-stage-inner { width: 100%; height: 100%; position: relative; }
.camera-widget .camera-frame { position: absolute; inset: 0; overflow: hidden; background: #000; }
.camera-widget .camera-frame video {
  position: absolute; inset: 0; width: 100%; height: 100%;
  object-fit: cover; pointer-events: none;
}
.camera-widget .camera-frame-overlay { position: absolute; inset: 0; pointer-events: none; }
.camera-widget .camera-frame-overlay svg { width: 100%; height: 100%; }
.camera-widget .camera-close-x {
  position: absolute; top: 16px; right: 16px; z-index: 10;
  width: 44px; height: 44px; border-radius: 50%; border: none;
  background: rgba(0,0,0,0.5); color: white; font-size: 22px;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
}
.camera-widget .camera-close-x[disabled] { opacity: 0.55; cursor: not-allowed; }
.camera-widget .camera-hint {
  position: absolute; bottom: 2%; left: 22%; width: 56%;
  text-align: center; color: white; font-size: 16px; z-index: 10;
  text-shadow: 0 1px 4px rgba(0,0,0,0.7); pointer-events: none;
}
.camera-widget .camera-shutter {
  position: absolute; top: 50%; right: 4%; transform: translateY(-50%);
  z-index: 10; width: 68px; height: 68px; border-radius: 50%;
  border: 4px solid white; background: transparent; cursor: pointer;
  padding: 0; display: flex; align-items: center; justify-content: center;
}
.camera-widget .camera-shutter span {
  display: block; width: 54px; height: 54px; border-radius: 50%; background: white;
}
.camera-widget .camera-shutter:active span { background: #ccc; }
.camera-widget .camera-shutter[disabled] { opacity: 0.55; cursor: not-allowed; }
@media (orientation: portrait) {
  .camera-widget .camera-stage-inner {
    display: flex; align-items: center; justify-content: center;
  }
  .camera-widget .camera-stage-inner::before {
    content: "\u041f\u043e\u0432\u0435\u0440\u043d\u0438 \u0442\u0435\u043b\u0435\u0444\u043e\u043d \u0433\u043e\u0440\u0438\u0437\u043e\u043d\u0442\u0430\u043b\u044c\u043d\u043e";
    color: white; font-size: 20px; text-align: center;
  }
  .camera-widget .camera-frame,
  .camera-widget .camera-close-x,
  .camera-widget .camera-hint,
  .camera-widget .camera-shutter { display: none; }
}
.char-card {
  border: 2px solid #e5e7eb; border-radius: 12px; padding: 12px;
  text-align: center; cursor: pointer; transition: border-color 0.2s;
}
.char-card:hover { border-color: #f97316; }
.char-card img { max-height: 120px; }
.qr-center-wrap {
  width: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 8px 0 4px;
}
.qr-side-panel {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100%;
}
.qr-side-panel > div {
  width: 100%;
}
.preset-media {
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  overflow: hidden;
  background: #fff;
}
.preset-media .block-title {
  color: #f97316;
  font-weight: 700;
}
.preset-media img,
.preset-media video {
  object-fit: contain !important;
}
.preset-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 8px;
}
.preset-card {
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  background: #fff;
  padding: 12px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.preset-card img {
  width: 100%;
  height: 150px;
  object-fit: contain;
  border-radius: 10px;
  background: linear-gradient(180deg, #fff, #f8fafc);
}
.preset-title {
  font-size: 13px;
  font-weight: 700;
  color: #111827;
  margin-bottom: 8px;
}
.preset-path {
  font-size: 12px;
  color: #6b7280;
  margin-top: 8px;
  word-break: break-word;
}
.preset-empty,
.preset-empty-block {
  min-height: 150px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #9ca3af;
  border: 1px dashed #d1d5db;
  border-radius: 10px;
  background: #f8fafc;
  padding: 16px;
}
@media (max-width: 640px) {
  .preset-grid { grid-template-columns: 1fr; }
}
"""
