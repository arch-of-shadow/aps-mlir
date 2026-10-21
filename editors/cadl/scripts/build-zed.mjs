// Zed requires a git repository + revision, even for local development.
// Snapshot only this grammar into an ignored local repository; never commit or
// modify the user's main working tree. No remote publication is required.
import { execFileSync } from 'node:child_process';
import { mkdir, cp, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const snapshot = path.join(root, 'build/grammar-repo');
const extension = path.join(root, 'build/zed');
await mkdir(snapshot, { recursive: true });
await mkdir(extension, { recursive: true });
for (const name of ['grammar.js', 'tree-sitter.json', 'package.json', 'src']) {
  await cp(path.join(root, 'tree-sitter-cadl', name), path.join(snapshot, name), { recursive: true });
}
await cp(path.join(root, 'LICENSE'), path.join(snapshot, 'LICENSE'));
await mkdir(path.join(snapshot, 'queries'), { recursive: true });
await cp(path.join(root, 'zed/languages/cadl/highlights.scm'), path.join(snapshot, 'queries/highlights.scm'));
const metadata = JSON.parse(await readFile(path.join(snapshot, 'tree-sitter.json'), 'utf8'));
metadata.grammars[0].highlights = 'queries/highlights.scm';
await writeFile(path.join(snapshot, 'tree-sitter.json'), JSON.stringify(metadata, null, 2) + '\n');
function git(args) {
  return execFileSync('git', args, { cwd: snapshot, encoding: 'utf8' }).trim();
}
if (!existsSync(path.join(snapshot, '.git'))) git(['init', '--quiet']);
git(['add', 'grammar.js', 'tree-sitter.json', 'package.json', 'src', 'queries', 'LICENSE']);
const staged = git(['diff', '--cached', '--name-only']);
if (staged || !existsSync(path.join(snapshot, '.git/HEAD')) || !git(['log', '-1', '--format=%H'])) {
  git(['-c', 'user.name=CADL editor build', '-c', 'user.email=cadl-build@localhost',
    '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'Snapshot CADL editor grammar']);
}
const revision = git(['rev-parse', 'HEAD']);
const repository = pathToFileURL(snapshot).href;
const template = await readFile(path.join(root, 'zed/extension.toml.in'), 'utf8');
await writeFile(path.join(extension, 'extension.toml'), template
  .replace('__GRAMMAR_REPOSITORY__', repository).replace('__GRAMMAR_REVISION__', revision));
await cp(path.join(root, 'zed/languages'), path.join(extension, 'languages'), { recursive: true });
await cp(path.join(root, 'LICENSE'), path.join(extension, 'LICENSE'));
await cp(path.join(root, 'README.md'), path.join(extension, 'README.md'));
console.log(`Zed dev extension: ${extension}\nGrammar revision: ${revision}`);
console.log('In Zed, run "zed: install dev extension" and select that directory.');
