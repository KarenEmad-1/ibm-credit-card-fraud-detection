// Build the self-contained frontend for the Docker image: node build.mjs <outDir>
//
// The dashboard has no compile step: the source in this folder runs as-is in the
// browser, with Preact/htm, D3 and the fonts coming from CDNs (see the import map in
// index.html). For the image, this script makes the same files work with no CDN:
//
//   vendor/preact-htm.js   htm's Preact standalone bundle (Preact + hooks + htm), copied
//   vendor/d3.js           D3 bundled into one minified ES module by esbuild
//   vendor/fonts.css       JetBrains Mono + VT323 (latin subsets), from @fontsource
//   index.html             import map, modulepreload and font links pointed at vendor/
//
// Versions come from package-lock.json, so the build is repeatable. The build fails if
// any http(s) URL is left in the output.
import { build } from "esbuild";
import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";

const out = process.argv[2] ?? "dist";
const vendor = join(out, "vendor");
// Some packages (htm) don't export package.json, so locate them in node_modules directly
const pkgDir = (name) => join("node_modules", ...name.split("/"));

rmSync(out, { recursive: true, force: true });
mkdirSync(join(vendor, "fonts"), { recursive: true });

// 1. App source, unchanged
for (const f of ["index.html", "styles.css", "js"]) cpSync(f, join(out, f), { recursive: true });

// 2. Preact + htm: the exact file the CDN serves (htm/preact/standalone.module.js)
cpSync(join(pkgDir("htm"), "preact", "standalone.module.js"), join(vendor, "preact-htm.js"));

// 3. D3: all of d3's sub-packages bundled into a single ES module
await build({
  stdin: { contents: 'export * from "d3";', resolveDir: process.cwd(), loader: "js" },
  bundle: true, format: "esm", minify: true, target: "es2020", legalComments: "eof",
  outfile: join(vendor, "d3.js"), logLevel: "warning",
});

// 4. Fonts: the weights styles.css uses, latin subset only
const fonts = [["@fontsource/jetbrains-mono", ["400", "500", "700", "800"]], ["@fontsource/vt323", ["400"]]];
let css = "/* JetBrains Mono and VT323, SIL Open Font License 1.1, via @fontsource */\n";
for (const [name, weights] of fonts) {
  for (const w of weights) {
    const src = readFileSync(join(pkgDir(name), `latin-${w}.css`), "utf8");
    css += src.replace(/url\(\.\/files\/([^)]+)\)/g, (_, file) => {
      cpSync(join(pkgDir(name), "files", file), join(vendor, "fonts", file));
      return `url(./fonts/${file})`;
    }) + "\n";
  }
}
writeFileSync(join(vendor, "fonts.css"), css);

// 5. Point index.html at the local copies
const indexPath = join(out, "index.html");
let html = readFileSync(indexPath, "utf8");
const swaps = [
  [/"htm\/preact\/standalone": "https:[^"]+"/, '"htm/preact/standalone": "./vendor/preact-htm.js"'],
  [/"d3": "https:[^"]+"/, '"d3": "./vendor/d3.js"'],
  [/<link rel="modulepreload" href="https:[^"]+">/, '<link rel="modulepreload" href="vendor/preact-htm.js">'],
  [/ *<link rel="preconnect" href="https:\/\/fonts\.googleapis\.com">\n/, ""],
  [/<link href="https:\/\/fonts\.googleapis\.com[^"]+" rel="stylesheet">/, '<link rel="stylesheet" href="vendor/fonts.css">'],
];
for (const [pattern, replacement] of swaps) {
  if (!pattern.test(html)) throw new Error(`index.html: expected to find ${pattern}`);
  html = html.replace(pattern, replacement);
}
writeFileSync(indexPath, html);

// 6. Nothing may still load from the network
const walk = (d) => readdirSync(d).flatMap((f) => (statSync(join(d, f)).isDirectory() ? walk(join(d, f)) : [join(d, f)]));
const remote = walk(out)
  .filter((f) => /\.(html|js|css)$/.test(f) && !f.endsWith("d3.js"))   // d3.js only has URLs in licence/comments
  .filter((f) => /(src|href)="https?:|from\s*["']https?:|url\(["']?https?:|"https?:\/\/[^"]+\.m?js"/.test(readFileSync(f, "utf8")))
  .map((f) => relative(out, f));
if (remote.length) throw new Error(`remote URLs left in: ${remote.join(", ")}`);
console.log(`frontend built in ${out}: ${walk(out).length} files`);
