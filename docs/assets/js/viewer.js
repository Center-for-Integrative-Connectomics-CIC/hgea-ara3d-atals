// HGEA on the 3D ARA: slices (coronal / sagittal / horizontal) and a 3D view of the domains.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const EDITION = new URLSearchParams(location.search).get('edition') === 'ccfv3bbp' ? 'ccfv3bbp' : 'ara';   // ?edition=ccfv3bbp
const DATA = new URL(EDITION === 'ccfv3bbp' ? '../../data/ccfv3bbp/' : '../../data/', import.meta.url).href;

// ───────── data ─────────
async function gz(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
const json = (name) => fetch(DATA + name).then((r) => r.json());

let META, ONTO, NISSL, ARA, LAB = {}, NA, ND, NL;
const S = { view: 'coronal', layer: 'full', nissl: true, outlines: false, opacity: 0.7, on: new Set(),
  pos: { coronal: 313, sagittal: 140, horizontal: 160 }, zoom: 1, tx: 0, ty: 0, nissl3d: 'volume', density: 0.5, shell: false, svg: false, svgAlpha: 0.8 };
let DOMRGB = new Uint8Array(256 * 3), DOMNAME = {};

async function boot() {
  [META, ONTO] = await Promise.all([json('meta.json'), json('ontology.json')]);
  [NA, ND, NL] = META.grid.shape;
  const [n, a, f, p] = await Promise.all(['nissl', 'ara', 'hgea_full', 'hgea_plates'].map((k) => gz(META.files[k].file)));
  NISSL = new Uint8Array(n); ARA = new Uint16Array(a); LAB.full = new Uint8Array(f); LAB.plates = new Uint8Array(p);
  for (const d of META.domains) {
    const rgb = [1, 3, 5].map((k) => parseInt(d.colour.slice(k, k + 2), 16));
    for (const id of [d.id, d.id + 100]) DOMRGB.set(rgb, id * 3);
    DOMNAME[d.id] = d.name; DOMNAME[d.id + 100] = `${d.name} (left)`;
    S.on.add(d.id);
  }
  S.pos.coronal = META.plates[Math.floor(META.plates.length / 2)].z;
  const hp = LAB.full; let lrs = 0, lrn = 0, dvs = 0;
  for (let i = 0; i < hp.length; i += 97) if (hp[i]) { const r = i % (ND * NL); lrs += Math.min(r % NL, NL - 1 - (r % NL)); dvs += Math.floor(r / NL); lrn++; }
  if (lrn) { S.pos.sagittal = Math.round(lrs / lrn); S.pos.horizontal = Math.round(dvs / lrn); }
  S.opacity = $('opacity').value / 100; ui(); init3D(); setView('coronal');
  $('loading').style.opacity = 0;
}

// ───────── panel ─────────
function ui() {
  $('domains').innerHTML = META.domains.map((d) => `<div class="dom" data-id="${d.id}"><span class="sw" style="background:${d.colour}"></span>${d.name}</div>`).join('');
  $('domains').querySelectorAll('.dom').forEach((el) => el.addEventListener('click', () => {
    const id = +el.dataset.id; S.on.has(id) ? S.on.delete(id) : S.on.add(id); el.classList.toggle('off', !S.on.has(id)); changed();
  }));
  $('allon').onclick = () => { META.domains.forEach((d) => S.on.add(d.id)); syncDoms(); changed(); };
  $('alloff').onclick = () => { S.on.clear(); syncDoms(); changed(); };
  seg('layer', (v) => { S.layer = v; changed(); });
  seg('view', setView);
  $('nissl').onchange = () => { S.nissl = $('nissl').checked; redraw(); };
  $('outlines').onchange = () => { S.outlines = $('outlines').checked; redraw(); };
  $('opacity').oninput = () => { S.opacity = $('opacity').value / 100; if (hgPts) { hgPts.material.opacity = S.opacity; hgPts.material.transparent = S.opacity < 1; need3D = true; } redraw(); };
  seg('nissl3d', (v) => { S.nissl3d = v; update3DNissl(); });
  $('svg').onchange = async () => {
    S.svg = $('svg').checked; $('svgrow').style.display = S.svg ? 'flex' : 'none';
    if (S.svg) await loadSvgs();
    updateSvg3D(); redraw();
  };
  $('svgop').oninput = () => { S.svgAlpha = $('svgop').value / 100; if (svgPlanes) svgPlanes.forEach((m) => { m.material.opacity = S.svgAlpha; }); need3D = true; redraw(); };
  $('density').oninput = () => { S.density = $('density').value / 100; update3DNissl(); };
  $('shell').onchange = () => { S.shell = $('shell').checked; if (shell) shell.visible = S.shell; need3D = true; };
  $('pos').oninput = () => { S.pos[pl()] = +$('pos').value; planesDirty = true; redraw(); };
  $('prev').onclick = () => jump(-1); $('next').onclick = () => jump(1);
  addEventListener('keydown', (e) => { if (e.target.tagName === 'INPUT') return; if (e.key === ']') jump(1); if (e.key === '[') jump(-1); });
  if (META.accuracy_rows) {   // CCFv3BBP edition: rows baked with the data
    $('acc').innerHTML = '<tr><th>measure</th><th class="n">value</th></tr>' + META.accuracy_rows.map(([m, v]) => `<tr><td>${m}</td><td class="n">${v}</td></tr>`).join('');
  } else {
    const A = META.accuracy, L = A.lopo_label_accuracy, D = A.lopo_hgea_dice;
    $('acc').innerHTML = `<tr><th>measure</th><th class="n">value</th></tr>
      <tr><td>placement, ARA lines (median)</td><td class="n">${A.placement_line_median_um} µm</td></tr>
      <tr><td>placement, hippocampus</td><td class="n">${A.placement_hpf_median_um} µm</td></tr>
      <tr><td>fill, ${A.fill_guide}-guided · label accuracy</td><td class="n">${L[A.fill_guide]}</td></tr>
      <tr><td>fill · HGEA overlap (Dice)</td><td class="n">${D[A.fill_guide]}</td></tr>
      <tr><td>plain interpolation, for comparison</td><td class="n">${L.none}</td></tr>`;
  }
  if (META.accuracy_note) $('accnote').textContent = META.accuracy_note;
  if (META.method) $('method').textContent = META.method;
  if (META.title) { const [a, b] = META.title.split(' · '); $('brand').innerHTML = `${a} · <em>${b}</em>`; document.title = META.title; }
  if (META.subtitle) $('subtitle').textContent = META.subtitle;
  if (META.svg === false) $('svgwrap').style.display = 'none';   // the SVG plates are drawn on the 3D ARA slices only
  document.querySelectorAll('#edition button').forEach((b) => {
    b.classList.toggle('active', b.dataset.v === EDITION);
    b.onclick = () => { if (b.dataset.v !== EDITION) location.search = b.dataset.v === 'ara' ? '' : `?edition=${b.dataset.v}`; };
  });
  $('sources').innerHTML = META.sources.map((s) => `· ${s}`).join('<br>');
}
function syncDoms() { $('domains').querySelectorAll('.dom').forEach((el) => el.classList.toggle('off', !S.on.has(+el.dataset.id))); }
function seg(id, fn) {
  document.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle('active', x === b)); fn(b.dataset.v);
  }));
}
function changed() { redraw(); if (S.view === '3d') build3D(); else dirty3D = true; }

// ───────── slices ─────────
const pl = () => (S.view === 'cut3d' ? 'coronal' : S.view);   // slice plane behind the current view
const is3D = () => S.view === '3d' || S.view === 'cut3d';
const PLANES = { coronal: { n: () => NA, W: () => NL, H: () => ND }, sagittal: { n: () => NL, W: () => NA, H: () => ND },
  horizontal: { n: () => ND, W: () => NL, H: () => NA } };
function vox(plane, k, u, v) {   // -> grid index
  if (plane === 'coronal') return (k * ND + v) * NL + u;
  if (plane === 'sagittal') return (u * ND + v) * NL + k;
  return (v * ND + k) * NL + u;
}
function setView(v) {
  S.view = v;
  const is3 = is3D();
  $('gl').style.display = is3 ? 'block' : 'none'; $('slice').style.display = is3 ? 'none' : 'block';
  $('slicebar').style.display = v === '3d' ? 'none' : 'flex';
  $('prev').style.display = $('next').style.display = pl() === 'coronal' ? '' : 'none';
  $('hint').textContent = is3 ? 'drag rotate · scroll zoom · right-drag pan' + (v === 'cut3d' ? ' · [ ] plates' : '') : 'scroll zoom · drag pan · [ ] plates';
  if (v !== '3d') { $('pos').max = PLANES[pl()].n() - 1; $('pos').value = S.pos[pl()]; }
  if (!is3) { S.zoom = 1; S.tx = S.ty = 0; }
  else { if (dirty3D || !hgPts) build3D(); if (v === 'cut3d') aimCut(); update3DNissl(); }
  $('sec3d').style.display = is3 ? 'block' : 'none';
  resize(); redraw();
}
function jump(step) {
  if (pl() !== 'coronal') return;
  const z = S.pos.coronal, zs = META.plates.map((p) => p.z);
  const t = step > 0 ? zs.find((p) => p > z) : [...zs].reverse().find((p) => p < z);
  if (t != null) { S.pos.coronal = t; $('pos').value = t; redraw(); }
}
function sliceImage(plane = pl(), k = S.pos[pl()], tissueAlpha = 255) {
  const P = PLANES[plane], W = P.W(), H = P.H(), lab = LAB[S.layer];
  const img = new ImageData(W, H), px = img.data, al = S.opacity;
  for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
    const i = vox(plane, k, u, v), o = (v * W + u) * 4, a = ARA[i];
    let r = 0, g = 0, b = 0, A = a ? tissueAlpha : 0;
    if (a) { const t = S.nissl ? 25 + NISSL[i] * 0.85 : 46; r = g = b = t; }
    const h = lab[i];
    if (h && S.on.has(h % 100)) { r = r * (1 - al) + DOMRGB[h * 3] * al; g = g * (1 - al) + DOMRGB[h * 3 + 1] * al; b = b * (1 - al) + DOMRGB[h * 3 + 2] * al; A = 255; }
    if (S.outlines && a && ((u + 1 < W && ARA[vox(plane, k, u + 1, v)] !== a) || (v + 1 < H && ARA[vox(plane, k, u, v + 1)] !== a))) { r = 110; g = 170; b = 230; }
    px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = A;
  }
  const cv = document.createElement('canvas'); cv.width = W; cv.height = H; cv.getContext('2d').putImageData(img, 0, 0);
  return cv;
}
function fit() {
  const cv = $('slice'), P = PLANES[S.view], W = P.W(), H = P.H();
  const s = Math.min(cv.width / W, cv.height / H) * 0.94 * S.zoom;
  return { s, ox: (cv.width - W * s) / 2 + S.tx, oy: (cv.height - H * s) / 2 + S.ty };
}
function redraw() {
  if (!META) return;
  where();
  if (S.view === 'cut3d') $('posl').textContent = `coronal ${S.pos.coronal} · ${(S.pos.coronal * 0.025).toFixed(2)} mm`;
  if (is3D()) { if (S.view === 'cut3d') { updateCut(); update3DNissl(); } else updateSvg3D(); need3D = true; return; }
  const cv = $('slice'), ctx = cv.getContext('2d'), P = PLANES[S.view];
  ctx.clearRect(0, 0, cv.width, cv.height);
  const { s, ox, oy } = fit();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(sliceImage(), ox, oy, P.W() * s, P.H() * s);
  if (S.svg && SVG) drawSvg2D(ctx, s, ox, oy);
  $('posl').textContent = `${S.view} ${S.pos[S.view]} · ${(S.pos[S.view] * 0.025).toFixed(2)} mm`;
}
function where() {
  const el = $('where');
  if (pl() !== 'coronal') { el.innerHTML = `${S.layer === 'full' ? 'filled 3D' : 'plate slices only'} · ${S.view === '3d' ? '3D view' : `${S.view} cut`}`; return; }
  const z = S.pos.coronal, p = META.plates.find((q) => q.z === z), tag = S.view === 'cut3d' ? ' · 3D cut' : '';
  if (p) {
    const fit = EDITION === 'ccfv3bbp' ? `median AP of the plate in CCFv3BBP (its sheet spans ${p.ap_spread_vox} slices)` : `ARA line fit ${p.line_median_um} µm (hippocampus ${p.hpf_median_um} µm)`;
    el.innerHTML = `<b>HGEA level ${p.level}</b> · Bregma ${p.bregma_mm.toFixed(2)} mm<br>${fit}${tag}`; return;
  }
  const a = [...META.plates].reverse().find((q) => q.z < z), b = META.plates.find((q) => q.z > z);
  el.innerHTML = (a && b ? `slice ${z}: <b>filled</b> between levels ${a.level} and ${b.level}` : `slice ${z}: outside the plate series`) + tag;
}

// pan, zoom, hover
(() => {
  const cv = $('slice'); let drag = null;
  cv.addEventListener('wheel', (e) => { e.preventDefault(); const f = e.deltaY < 0 ? 1.15 : 1 / 1.15; S.zoom = Math.max(0.5, Math.min(12, S.zoom * f)); redraw(); }, { passive: false });
  cv.addEventListener('mousedown', (e) => { drag = { x: e.clientX, y: e.clientY, tx: S.tx, ty: S.ty }; });
  addEventListener('mouseup', () => { drag = null; });
  cv.addEventListener('mousemove', (e) => {
    if (drag) { S.tx = drag.tx + (e.clientX - drag.x) * devicePixelRatio; S.ty = drag.ty + (e.clientY - drag.y) * devicePixelRatio; redraw(); return; }
    const r = cv.getBoundingClientRect(), { s, ox, oy } = fit(), P = PLANES[S.view];
    const u = Math.floor(((e.clientX - r.left) * devicePixelRatio - ox) / s), v = Math.floor(((e.clientY - r.top) * devicePixelRatio - oy) / s);
    const tip = $('tip');
    if (u < 0 || v < 0 || u >= P.W() || v >= P.H()) { tip.style.display = 'none'; return; }
    const i = vox(S.view, S.pos[S.view], u, v), o = ONTO[ARA[i]];
    const hf = LAB.full[i], hp = LAB.plates[i];
    tip.innerHTML = `<b>${o.acronym}</b> ${o.name}` + (hf ? `<br>HGEA (filled): <b>${DOMNAME[hf]}</b>` : '') + (hp ? `<br>HGEA (plate): <b>${DOMNAME[hp]}</b>` : '');
    tip.style.display = 'block'; tip.style.left = `${e.clientX - r.left + 14}px`; tip.style.top = `${e.clientY - r.top + 14}px`;
  });
  cv.addEventListener('mouseleave', () => { $('tip').style.display = 'none'; });
})();

// ───────── 3D ─────────
let cutFace = null, clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
let renderer, scene, camera, controls, hgPts = null, shell = null, dirty3D = true, need3D = true, vol = null, planes = null, planesDirty = true;
const toScene = (a, d, l) => [(l - NL / 2) * 0.025, -(d - ND / 2) * 0.025, -(a - NA / 2) * 0.025];
function init3D() {
  renderer = new THREE.WebGLRenderer({ canvas: $('gl'), antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); renderer.localClippingEnabled = true;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(35, 1, 0.01, 200); camera.position.set(9, 6, 8);
  controls = new OrbitControls(camera, $('gl')); controls.enableDamping = true;
  controls.target.set(...toScene(META.plates[16].z, 140, NL / 2)); controls.addEventListener('change', () => { need3D = true; });
  // brain shell: boundary voxels of the annotation, every 3rd voxel
  const pos = [];
  for (let a = 0; a < NA; a += 3) for (let d = 0; d < ND; d += 3) for (let l = 0; l < NL; l += 3) {
    const i = (a * ND + d) * NL + l; if (!ARA[i]) continue;
    if ((l + 3 < NL && !ARA[i + 3]) || (l >= 3 && !ARA[i - 3]) || (d + 3 < ND && !ARA[i + 3 * NL]) || (d >= 3 && !ARA[i - 3 * NL])) pos.push(...toScene(a, d, l));
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  shell = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.03, color: 0x8fa3b0, transparent: true, opacity: 0.12, depthWrite: false }));
  shell.visible = S.shell; scene.add(shell);
  scene.add(new THREE.AmbientLight(0xffffff, 0.8));
  const key = new THREE.DirectionalLight(0xffffff, 0.6); key.position.set(4, 8, 6); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.3); fill.position.set(-5, -3, -4); scene.add(fill);
  new ResizeObserver(resize).observe($('stage'));
  (function loop() { requestAnimationFrame(loop); controls.update(); if (need3D && is3D()) { need3D = false; if (vol && vol.visible) vol.material.uniforms.uCam.value.copy(vol.worldToLocal(camera.position.clone())); renderer.render(scene, camera); } })();
}
function build3D() {   // voxel-exact surfaces: every face between a shown domain voxel and a different label (as napari draws labels)
  if (hgPts) { scene.remove(hgPts); hgPts.geometry.dispose(); }
  const lab = LAB[S.layer], show = (h) => h && S.on.has(h % 100);
  const step = [ND * NL, NL, 1], dims = [NA, ND, NL];
  let n = 0;
  const count = (i, c, j, inside) => { if (!inside || lab[j] !== c) n++; };
  for (let a = 0; a < NA; a++) for (let d = 0; d < ND; d++) for (let l = 0; l < NL; l++) {
    const i = (a * ND + d) * NL + l, c = lab[i]; if (!show(c)) continue;
    const q = [a, d, l];
    for (let ax = 0; ax < 3; ax++) { count(i, c, i + step[ax], q[ax] + 1 < dims[ax]); count(i, c, i - step[ax], q[ax] > 0); }
  }
  const pos = new Float32Array(n * 12), nor = new Int8Array(n * 12), col = new Float32Array(n * 12), idx = new Uint32Array(n * 6);
  const lin = (x) => { const c = x / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };   // sRGB legend colours -> linear, as three.js expects
  let f = 0;
  const V = 0.025, cx = NL / 2, cy = ND / 2, cz = NA / 2;
  // a face on axis ax at voxel boundary, side s (+1 / -1); corners in grid units (a, d, l)
  const face = (a, d, l, ax, sgn, c) => {
    const o = [a, d, l]; o[ax] += sgn > 0 ? 1 : 0;
    const u = (ax + 1) % 3, w = (ax + 2) % 3, cs = [[0, 0], [1, 0], [1, 1], [0, 1]];
    for (let k = 0; k < 4; k++) {
      const p = [o[0], o[1], o[2]]; p[u] += cs[k][0]; p[w] += cs[k][1];
      const vi = (f * 4 + k) * 3;
      pos[vi] = (p[2] - cx) * V; pos[vi + 1] = -(p[1] - cy) * V; pos[vi + 2] = -(p[0] - cz) * V;
      const nv = [0, 0, 0]; nv[ax] = sgn;   // grid normal -> scene normal (x = LR, y = -DV, z = -AP)
      nor[vi] = nv[2] * 127; nor[vi + 1] = -nv[1] * 127; nor[vi + 2] = -nv[0] * 127;
      col[vi] = lin(DOMRGB[c * 3]); col[vi + 1] = lin(DOMRGB[c * 3 + 1]); col[vi + 2] = lin(DOMRGB[c * 3 + 2]);
    }
    const b0 = f * 4, ii = f * 6;
    // winding so the face is front-facing along its normal (parity of the axis permutation decides)
    if ((sgn > 0) === (ax !== 1)) { idx.set([b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3], ii); } else { idx.set([b0, b0 + 2, b0 + 1, b0, b0 + 3, b0 + 2], ii); }
    f++;
  };
  for (let a = 0; a < NA; a++) for (let d = 0; d < ND; d++) for (let l = 0; l < NL; l++) {
    const i = (a * ND + d) * NL + l, c = lab[i]; if (!show(c)) continue;
    const q = [a, d, l];
    for (let ax = 0; ax < 3; ax++) {
      if (!(q[ax] + 1 < dims[ax]) || lab[i + step[ax]] !== c) face(a, d, l, ax, 1, c);
      if (!(q[ax] > 0) || lab[i - step[ax]] !== c) face(a, d, l, ax, -1, c);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3, true));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  hgPts = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: S.opacity < 1, opacity: S.opacity,
    clippingPlanes: S.view === 'cut3d' ? [clip] : [] }));
  scene.add(hgPts); dirty3D = false; need3D = true;
}
function resize() {
  const st = $('stage'), w = st.clientWidth, h = st.clientHeight;
  if (renderer) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); need3D = true; }
  const sc = $('slice'); sc.width = w * devicePixelRatio; sc.height = h * devicePixelRatio; redraw();
}

// 3D Nissl: volume rendering (front-to-back compositing, 50 µm) or three orthogonal slices (25 µm)
const VS = `out vec3 vPos; void main() { vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const FS = `precision highp float; precision highp sampler3D;
in vec3 vPos; out vec4 outColor;
uniform sampler3D uVol; uniform vec3 uCam; uniform float uDensity; uniform float uClipZ;
vec2 hitBox(vec3 o, vec3 d) { vec3 inv = 1.0 / d, t0 = (vec3(-0.5) - o) * inv, t1 = (vec3(0.5) - o) * inv;
  vec3 tmin = min(t0, t1), tmax = max(t0, t1); return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z)); }
void main() {
  vec3 dir = normalize(vPos - uCam); vec2 t = hitBox(uCam, dir); t.x = max(t.x, 0.0);
  if (t.x >= t.y) discard;
  float dt = 1.0 / 320.0; vec4 acc = vec4(0.0);
  for (float s = t.x; s < t.y; s += dt) {
    vec3 p = uCam + dir * s;
    if (p.z > uClipZ) continue;
    float v = texture(uVol, vec3(p.x + 0.5, 0.5 - p.y, 0.5 - p.z)).r;
    float a = smoothstep(0.08, 0.6, v) * uDensity * 0.08;
    vec3 c = vec3(0.35 + 0.9 * v);
    acc.rgb += (1.0 - acc.a) * a * c; acc.a += (1.0 - acc.a) * a;
    if (acc.a > 0.97) break;
  }
  outColor = acc;
}`;
function buildVolume() {
  const W = NL >> 1, H = ND >> 1, D = NA >> 1, v = new Uint8Array(W * H * D);
  for (let z = 0; z < D; z++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0;
    for (let dz = 0; dz < 2; dz++) for (let dy = 0; dy < 2; dy++) { const b = ((2 * z + dz) * ND + 2 * y + dy) * NL + 2 * x; s += NISSL[b] + NISSL[b + 1]; }
    v[(z * H + y) * W + x] = s >> 3;
  }
  const tex = new THREE.Data3DTexture(v, W, H, D);
  tex.format = THREE.RedFormat; tex.minFilter = tex.magFilter = THREE.LinearFilter; tex.unpackAlignment = 1; tex.needsUpdate = true;
  const mat = new THREE.ShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: VS, fragmentShader: FS, side: THREE.BackSide,
    transparent: true, depthWrite: false, uniforms: { uVol: { value: tex }, uCam: { value: new THREE.Vector3() }, uDensity: { value: S.density }, uClipZ: { value: 1.0 } } });
  vol = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), mat);
  vol.scale.set(NL * 0.025, ND * 0.025, NA * 0.025); vol.renderOrder = -1;
  scene.add(vol);
}
function planeMesh(plane) {
  const P = PLANES[plane], W = P.W(), H = P.H(), k = S.pos[plane], data = new Uint8Array(W * H * 4);
  for (let v = 0; v < H; v++) for (let u = 0; u < W; u++) {
    const i = vox(plane, k, u, v), o = (v * W + u) * 4, g = Math.min(255, 30 + NISSL[i] * 1.1);
    data[o] = data[o + 1] = data[o + 2] = g; data[o + 3] = ARA[i] ? Math.round(255 * Math.min(1, 0.25 + S.density)) : 0;
  }
  const tex = new THREE.DataTexture(data, W, H); tex.flipY = false; tex.magFilter = THREE.LinearFilter; tex.needsUpdate = true;
  const m = new THREE.Mesh(new THREE.PlaneGeometry(W * 0.025, H * 0.025),
    new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
  // plane texture: u right, v down; place each plane at its cut
  if (plane === 'coronal') { m.position.set(0, 0, -(k - NA / 2) * 0.025); m.scale.y = -1; }
  else if (plane === 'sagittal') { m.rotation.y = Math.PI / 2; m.position.set((k - NL / 2) * 0.025, 0, 0); m.scale.y = -1; }
  else { m.rotation.x = -Math.PI / 2; m.position.set(0, -(k - ND / 2) * 0.025, 0); m.scale.y = -1; }
  return m;
}
// coronal 3D: the atlas cut at the coronal position; the cut face shows that slice exactly as in the 2D view
function updateCut() {
  if (!scene) return;
  const k = S.pos.coronal, zc = -(k - NA / 2) * 0.025;
  clip.constant = zc - 0.0125;   // mesh keeps everything posterior to the slice; the cut face shows the slice itself
  if (hgPts) hgPts.material.clippingPlanes = [clip];
  const tex = new THREE.CanvasTexture(sliceImage('coronal', k, 150));   // tissue see-through, domains solid
  tex.magFilter = THREE.NearestFilter; tex.colorSpace = THREE.SRGBColorSpace;
  if (!cutFace) {
    cutFace = new THREE.Mesh(new THREE.PlaneGeometry(NL * 0.025, ND * 0.025), new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide }));
    scene.add(cutFace);
  } else { cutFace.material.map.dispose(); cutFace.material.map = tex; }
  cutFace.visible = true; cutFace.position.set(0, 0, zc);
  need3D = true;
}
function aimCut() {
  updateCut();
  const zc = -(S.pos.coronal - NA / 2) * 0.025;
  controls.target.set(0, 0.3, zc - 0.8); camera.position.set(8.5, 6, zc + 13); controls.update();
}

function update3DNissl() {
  if (!scene) return;
  updateSvg3D();
  const cut = S.view === 'cut3d', clips = cut ? [clip] : [];
  if (!cut) { if (cutFace) cutFace.visible = false; if (hgPts) hgPts.material.clippingPlanes = []; }
  if (S.nissl3d === 'volume' && !vol) buildVolume();
  if (vol) {
    vol.visible = S.nissl3d === 'volume'; vol.material.uniforms.uDensity.value = S.density;
    vol.material.uniforms.uClipZ.value = cut ? (clip.constant / vol.scale.z) : 1.0;   // world z of the cut -> box-local z
  }
  if (planes && (planesDirty || S.nissl3d !== 'planes')) { planes.forEach((m) => { scene.remove(m); m.geometry.dispose(); m.material.map.dispose(); }); planes = null; }
  if (S.nissl3d === 'planes' && !planes) { planes = ['coronal', 'sagittal', 'horizontal'].map(planeMesh); planes.forEach((m) => scene.add(m)); planesDirty = false; }
  if (planes) planes.forEach((m, k) => { m.material.clippingPlanes = clips; m.visible = !(cut && k === 0); });   // the cut face replaces the coronal slice
  if (shell) shell.material.clippingPlanes = clips;
  need3D = true;
}

// ───────── Bienkowski SVG plates (original drawings, aligned by the same transform the build used) ─────────
let SVG = null, svgPlanes = null, svgLoading = null;
const svgClip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0);
const R = 2;   // raster resolution for 3D and for sagittal / horizontal crossings: R pixels per 25 µm voxel
function loadSvgs() {
  if (svgLoading) return svgLoading;
  $('where').innerHTML += '<br>loading the 33 HGEA SVG plates…';
  svgLoading = Promise.all(META.plates.map((p) => new Promise((ok, bad) => {
    const im = new Image(); im.onload = () => ok(im); im.onerror = () => bad(new Error(p.svg)); im.src = DATA + p.svg;
  }))).then((imgs) => {
    SVG = META.plates.map((p, k) => {
      const [A, B, C, D, E, F] = p.svg_affine, cv = document.createElement('canvas');
      cv.width = NL * R; cv.height = ND * R;
      const ctx = cv.getContext('2d'), o = p.brain_outline;
      ctx.beginPath();   // keep the drawing on the brain: drops the legend, inset, axes and labels around the plate
      for (let j = 0; j < o.length; j += 2) ctx[j ? 'lineTo' : 'moveTo'](R * (o[j] + 0.5), R * (o[j + 1] + 0.5));
      ctx.closePath(); ctx.clip();
      ctx.setTransform(R * A, R * D, R * B, R * E, R * (C + 0.5), R * (F + 0.5));   // SVG px -> raster px (voxel centres at +0.5)
      ctx.drawImage(imgs[k], 0, 0);
      return { plate: p, img: imgs[k], raster: cv, data: ctx.getImageData(0, 0, cv.width, cv.height).data };
    });
    where();
  });
  return svgLoading;
}
function drawSvg2D(ctx, s, ox, oy) {
  ctx.save(); ctx.globalAlpha = S.svgAlpha;
  if (S.view === 'coronal') {
    const e = SVG.find((x) => x.plate.z === S.pos.coronal);
    if (e) {
      const [A, B, C, D, E, F] = e.plate.svg_affine;
      ctx.imageSmoothingEnabled = true;
      ctx.setTransform(s * A, s * D, s * B, s * E, ox + s * (C + 0.5), oy + s * (F + 0.5));
      ctx.drawImage(e.img, 0, 0);
    }
  } else {   // a 1-voxel strip of each plate's drawing at its slice
    const P = PLANES[S.view], W = P.W(), H = P.H(), k = S.pos[S.view], img = new ImageData(W, H), px = img.data, RW = NL * R;
    for (const e of SVG) {
      const z = e.plate.z;
      for (let t = 0; t < (S.view === 'sagittal' ? ND : NL); t++) {
        const x = S.view === 'sagittal' ? k : t, y = S.view === 'sagittal' ? t : k;
        const si = ((y * R + (R >> 1)) * RW + x * R + (R >> 1)) * 4;
        if (e.data[si + 3] < 20) continue;
        const u = S.view === 'sagittal' ? z : t, v = S.view === 'sagittal' ? t : z, o = (v * W + u) * 4;
        px[o] = e.data[si]; px[o + 1] = e.data[si + 1]; px[o + 2] = e.data[si + 2]; px[o + 3] = 255;
      }
    }
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H; cv.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(cv, ox, oy, W * s, H * s);
  }
  ctx.restore();
}
function updateSvg3D() {
  if (!scene) return;
  if (S.svg && SVG && !svgPlanes) {
    svgPlanes = SVG.map((e) => {
      const tex = new THREE.CanvasTexture(e.raster); tex.colorSpace = THREE.SRGBColorSpace;
      const m = new THREE.Mesh(new THREE.PlaneGeometry(NL * 0.025, ND * 0.025),
        new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: S.svgAlpha, side: THREE.DoubleSide, depthWrite: false }));
      m.position.set(0, 0, -(e.plate.z - NA / 2) * 0.025 + 0.003);   // just anterior of the slice, so it draws over the cut face
      scene.add(m); return m;
    });
  }
  if (!svgPlanes) return;
  const cut = S.view === 'cut3d';
  svgClip.constant = clip.constant + 0.025 + 0.004;   // in coronal 3D keep the plate at the cut and everything posterior
  svgPlanes.forEach((m) => { m.visible = S.svg; m.material.clippingPlanes = cut ? [svgClip] : []; });
  need3D = true;
}

boot().catch((e) => { $('loading').textContent = `could not load: ${e.message}`; console.error(e); });
