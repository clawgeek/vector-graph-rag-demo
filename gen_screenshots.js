const puppeteer = require('/Users/sang/.workbuddy/binaries/node/workspace/node_modules/puppeteer-core');
const fs = require('fs');
const path = require('path');
const os = require('os');

const DIR = '/Users/sang/WorkBuddy/文章发布/vector-graph-rag-demo';
const SHOTS = path.join(DIR, 'screenshots');

// 把 ASCII 箭头换成 unicode，避免内容里出现 > 干扰 HTML；不做整体转义，
// 颜色用正则直接注入 <span> 标签（原始文本里没有 < > 字符）。
function prep(s) {
  return s.replace(/-->/g, '→').replace(/\r/g, '');
}

function html(title, body) {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0b0e14;font-family:"SF Mono",Menlo,Consolas,monospace;padding:34px 40px;}
  .win{background:#0d1117;border:1px solid #1f2630;border-radius:10px;overflow:hidden;box-shadow:0 10px 40px rgba(0,0,0,.5)}
  .bar{background:#161b22;padding:10px 16px;display:flex;align-items:center;gap:8px;border-bottom:1px solid #1f2630}
  .dot{width:12px;height:12px;border-radius:50%}
  .r{background:#ff5f56}.y{background:#ffbd2e}.g{background:#27c93f}
  .t{margin-left:12px;color:#8b949e;font-size:13px}
  pre{color:#c9d1d9;font-size:15px;line-height:1.55;padding:22px 24px;white-space:pre-wrap;word-break:break-word}
  .hl{color:#58a6ff}.ok{color:#3fb950}.ac{color:#ff7a18}
  </style></head><body><div class="win"><div class="bar">
    <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
    <span class="t">${title}</span></div><pre>${body}</pre></div></body></html>`;
}

(async () => {
  let idx = prep(fs.readFileSync(path.join(SHOTS, 'out_index.txt'), 'utf8'))
    .replace(/（[^）]*Milvus[^）]*）/g, '<span class="ac">$&</span>')
    .replace(/实体向量库[^|]*/g, '<span class="ok">$&</span>');
  let q = prep(fs.readFileSync(path.join(SHOTS, 'out_query.txt'), 'utf-8'))
    .replace(/答案[:：].*/g, m => `<span class="ok">${m}</span>`)
    .replace(/检索路径[:：].*/g, m => `<span class="hl">${m}</span>`)
    .replace(/Q\d.*/g, m => `<span class="ac">${m}</span>`);

  const idxHtml = path.join(SHOTS, '_idx.html');
  const qHtml = path.join(SHOTS, '_q.html');
  fs.writeFileSync(idxHtml, html('demo · index — vector graph build', idx));
  fs.writeFileSync(qHtml, html('demo · query — multi-hop reasoning', q));

  const browser = await puppeteer.launch({
    executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    headless: 'new',
    userDataDir: path.join(os.tmpdir(), 'vgshot_' + Date.now()),
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu', '--force-color-profile=srgb']
  });

  const jobs = [[idxHtml, '01_index.png'], [qHtml, '02_query.png']];
  for (const [file, out] of jobs) {
    const p = await browser.newPage();
    await p.setViewport({ width: 1040, height: 900, deviceScaleFactor: 2 });
    await p.goto('file://' + file, { waitUntil: 'networkidle0' });
    await new Promise(r => setTimeout(r, 250));
    await p.screenshot({ path: path.join(SHOTS, out), fullPage: true });
    await p.close();
    console.log('saved', out);
  }
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
