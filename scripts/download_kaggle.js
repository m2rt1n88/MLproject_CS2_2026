// Minimal Kaggle dataset downloader.
// Works around the broken Windows schannel TLS stack in this sandbox by using
// Node's bundled OpenSSL instead of the OS crypto store.
//
// Usage: node scripts/download_kaggle.js <owner/dataset> <output-file>
const https = require("https");
const fs = require("fs");
const path = require("path");

const [, , ref, outPath] = process.argv;
if (!ref || !outPath) {
  console.error("usage: node download_kaggle.js <owner/dataset> <output-file>");
  process.exit(2);
}

const startUrl = `https://www.kaggle.com/api/v1/datasets/download/${ref}`;

function download(url, redirectsLeft, cb) {
  https
    .get(url, { headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" } }, (res) => {
      const { statusCode, headers } = res;

      if ([301, 302, 303, 307, 308].includes(statusCode) && headers.location) {
        res.resume();
        if (redirectsLeft <= 0) return cb(new Error("too many redirects"));
        const next = new URL(headers.location, url).href;
        return download(next, redirectsLeft - 1, cb);
      }

      if (statusCode !== 200) {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (c) => (body += c.slice(0, 500)));
        res.on("end", () => cb(new Error(`HTTP ${statusCode}: ${body.trim()}`)));
        return;
      }

      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      const file = fs.createWriteStream(outPath);
      res.pipe(file);
      file.on("finish", () => file.close(() => cb(null, fs.statSync(outPath).size)));
      file.on("error", cb);
      res.on("error", cb);
    })
    .on("error", cb);
}

download(startUrl, 8, (err, size) => {
  if (err) {
    console.error("FAILED:", err.message);
    process.exit(1);
  }
  console.log(`OK ${outPath} ${(size / 1048576).toFixed(2)} MiB`);
});
