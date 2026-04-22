(function() {
  if (window.__inkCameraInitialized) return;
  window.__inkCameraInitialized = true;

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

  async function getCameraStream(isMobile) {
    var tries = [];
    if (isMobile) {
      tries.push({ audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: 'environment' } } });
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
        if (loaded) status.textContent = 'Фото загружено.';
        return;
      }
      startDetection(widget);
    } catch (err) {
      closeModal(widget);
      status.textContent = 'Не удалось открыть камеру. Проверь разрешение браузера и https.';
    }
  }

  function capturePhoto(widget) {
    var video = widget.querySelector('[data-role="camera-video"]');
    var status = widget.querySelector('[data-role="camera-status"]');
    var w = video.videoWidth || 1280, h = video.videoHeight || 720;
    if (!w || !h) { status.textContent = 'Capture failed.'; return; }
    var cx = Math.round(w * 0.17), cy = Math.round(h * 0.08);
    var cw = Math.round(w * 0.66), ch = Math.round(h * 0.84);
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

  function setFrameColor(widget, count) {
    var rects = widget.querySelectorAll('.camera-frame-overlay svg rect[stroke]');
    var badge = widget.querySelector('[data-role="camera-badge"]');
    var snap = widget.querySelector('[data-role="camera-snap"]');
    var strokeColor, badgeBg, badgeText;
    if (count >= 4) {
      strokeColor = 'rgba(34,197,94,0.95)';
      badgeBg = 'rgba(34,197,94,0.92)';
      badgeText = 'OK — фоткай!';
    } else if (count >= 1) {
      strokeColor = 'rgba(245,158,11,0.95)';
      badgeBg = 'rgba(245,158,11,0.9)';
      badgeText = 'Метки: ' + count + '/4';
    } else {
      strokeColor = 'rgba(239,68,68,0.95)';
      badgeBg = 'rgba(239,68,68,0.9)';
      badgeText = 'Ищу метки…';
    }
    rects.forEach(function(r) { r.setAttribute('stroke', strokeColor); });
    if (badge) { badge.textContent = badgeText; badge.style.background = badgeBg; }
    if (snap) snap.style.borderColor = count >= 4 ? 'rgba(34,197,94,0.95)' : 'white';
  }

  var _detector = null;
  var _detectorInitError = null;
  function getDetector() {
    if (_detector) return _detector;
    if (typeof AR === 'undefined' || !AR.Detector) return null;
    try { _detector = new AR.Detector({ dictionaryName: 'ARUCO_4X4_1000' }); }
    catch(e) { _detectorInitError = (e && e.message) ? e.message : String(e); }
    return _detector;
  }

  function setDebug(widget, text) {
    var dbg = widget.querySelector('[data-role="camera-debug"]');
    if (dbg) dbg.textContent = text;
  }

  function startDetection(widget) {
    var video = widget.querySelector('[data-role="camera-video"]');
    var stage = widget.querySelector('[data-role="camera-stage"]');
    var badge = widget.querySelector('[data-role="camera-badge"]');
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d', { willReadFrequently: true });
    widget._detectRunning = true;
    setDebug(widget, 'AR=' + (typeof AR !== 'undefined' ? 'ok' : 'missing') +
             ' known=' + JSON.stringify(window.__arucoKnownIds || []));
    async function loop() {
      while (widget._detectRunning) {
        await new Promise(function(r) { setTimeout(r, 400); });
        if (!widget._detectRunning || stage.hidden || !video.videoWidth) continue;
        var detector = getDetector();
        if (!detector) {
          var msg = (typeof AR === 'undefined')
            ? 'detector: scripts not loaded (AR undefined)'
            : (_detectorInitError ? 'detector init error: ' + _detectorInitError
                                  : 'detector: initializing…');
          if (badge) badge.textContent = 'Загрузка детектора…';
          setDebug(widget, msg);
          continue;
        }
        // Crop to the visible guide frame (same % as capturePhoto)
        var vw = video.videoWidth, vh = video.videoHeight;
        var sx = Math.round(vw * 0.17), sy = Math.round(vh * 0.08);
        var sw = Math.round(vw * 0.66), sh = Math.round(vh * 0.84);
        var targetW = 480;
        var scale = targetW / sw;
        canvas.width = targetW;
        canvas.height = Math.max(1, Math.round(sh * scale));
        ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
        try {
          var t0 = performance.now();
          var imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          var markers = detector.detect(imgData);
          var dt = Math.round(performance.now() - t0);
          var known = window.__arucoKnownIds || [];
          var knownSet = {}; for (var k = 0; k < known.length; k++) knownSet[known[k]] = 1;
          var seen = {}, count = 0, ids = [];
          for (var i = 0; i < markers.length; i++) {
            var id = markers[i].id;
            ids.push(id);
            if (knownSet[id] && !seen[id]) { seen[id] = 1; count++; }
          }
          setFrameColor(widget, count);
          setDebug(widget,
            'video=' + vw + 'x' + vh +
            ' crop=' + canvas.width + 'x' + canvas.height +
            ' det=' + dt + 'ms' +
            ' raw=' + markers.length + ' known=' + count +
            (ids.length ? ' ids=' + ids.join(',') : ''));
        } catch(e) {
          setDebug(widget, 'detect error: ' + ((e && e.message) ? e.message : e));
        }
      }
    }
    loop();
  }

  function stopDetection(widget) { widget._detectRunning = false; setFrameColor(widget, 0); }

  function bindAllCameraWidgets() { document.querySelectorAll('.camera-widget').forEach(bindCameraWidget); }
  var _bindTimer = 0;
  function bindAllCameraWidgetsDebounced() { clearTimeout(_bindTimer); _bindTimer = setTimeout(bindAllCameraWidgets, 300); }
  window.addEventListener('load', bindAllCameraWidgets);
  new MutationObserver(bindAllCameraWidgetsDebounced).observe(document.documentElement, { childList: true, subtree: true });

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
        var url = window.location.origin + window.location.pathname;
        if (charId) url += '?char=' + charId;
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

  var _qrTimer = 0;
  function generateAllQRDebounced() { clearTimeout(_qrTimer); _qrTimer = setTimeout(generateAllQR, 300); }
  window.addEventListener('load', generateAllQR);
  new MutationObserver(generateAllQRDebounced).observe(document.documentElement, { childList: true, subtree: true });

})();
