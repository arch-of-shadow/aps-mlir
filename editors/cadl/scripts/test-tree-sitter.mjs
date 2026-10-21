import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readdirSync, mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const repo = path.resolve(root, '../..');
const grammar = path.join(root, 'tree-sitter-cadl');
const cli = path.join(root, 'node_modules/.bin/tree-sitter');
mkdirSync(path.join(root, 'build'), { recursive: true });
const config = path.join(root, 'build/tree-sitter-config.json');
writeFileSync(config, JSON.stringify({ 'parser-directories': [root] }));
function run(args) {
  return execFileSync(cli, args, { cwd: grammar, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
}
function cadlFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    if (entry.name === 'wont_pass') return [];
    const full = path.join(dir, entry.name);
    return entry.isDirectory() ? cadlFiles(full) : entry.name.endsWith('.cadl') ? [full] : [];
  });
}
process.stdout.write(run(['test']));
const files = [path.join(root, 'fixtures/highlight.cadl'), ...['examples', 'tutorial/cadl', 'tests/hardfloat']
  .flatMap(dir => cadlFiles(path.join(repo, dir)))];
run(['parse', '--quiet', '--config-path', config, ...files]);
console.log(`Tree-sitter: ${files.length} CADL files parsed without errors.`);
const fixture = path.join(root, 'fixtures/highlight.cadl');
for (const query of ['highlights', 'brackets', 'indents', 'outline']) {
  const queryPath = path.join(root, `zed/languages/cadl/${query}.scm`);
  const output = run(['query', '--config-path', config, queryPath, fixture]);
  assert.match(output, /capture:/, `${query} query matched nothing`);
  if (query === 'highlights') {
    for (const [scope, text] of [
      ['function', 'example'], ['attribute', 'opcode'], ['attribute', 'unroll'],
      ['type', 'bf16'], ['number', '1.0e+2_f32'], ['variable.special', '_burst_read'],
      ['function.builtin', 'bf16_add'], ['function.builtin', 'fp32_fma_flags'],
      ['function.builtin', '$signed'], ['boolean', 'true'],
    ]) {
      assert.ok(output.split('\n').some(line => line.includes(`- ${scope},`) && line.endsWith(`\`${text}\``)),
        `missing ${scope} capture for ${text}`);
    }
  }
}
// ABI 15 works with current Zed and tree-sitter 0.25+.
assert.match(readFileSync(path.join(grammar, 'src/parser.c'), 'utf8'), /#define LANGUAGE_VERSION 15/);
console.log('Zed: all four queries compiled and matched; highlight assertions passed.');
