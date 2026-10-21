import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import tm from 'vscode-textmate';
import onig from 'vscode-oniguruma';

const root = new URL('../', import.meta.url);
const wasm = await readFile(new URL('../node_modules/vscode-oniguruma/release/onig.wasm', import.meta.url));
await onig.loadWASM(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength));
const grammarPath = new URL('vscode/syntaxes/cadl.tmLanguage.json', root);
const registry = new tm.Registry({
  onigLib: Promise.resolve({
    createOnigScanner: patterns => new onig.OnigScanner(patterns),
    createOnigString: text => new onig.OnigString(text),
  }),
  loadGrammar: async scope => scope === 'source.cadl'
    ? tm.parseRawGrammar(await readFile(grammarPath, 'utf8'), fileURLToPath(grammarPath)) : null,
});
const grammar = await registry.loadGrammar('source.cadl');
let checks = 0;
function check(source, text, scope, state = tm.INITIAL) {
  const result = grammar.tokenizeLine(source, state);
  const start = source.indexOf(text);
  assert.notEqual(start, -1);
  const tokens = result.tokens.filter(t => t.startIndex < start + text.length && t.endIndex > start);
  assert.ok(tokens.length && tokens.every(t => t.scopes.includes(scope)),
    `${JSON.stringify(text)} in ${JSON.stringify(source)}: expected ${scope}, got ${JSON.stringify(tokens)}`);
  checks++;
  return result.ruleStack;
}
for (const literal of [
  "7'b0101011", "8'SB1010_u8", "8'o77", "8'SO77", "32'd1000", "32'sD42_i32",
  "64'shFFFFFFFF_u64", "8'HFF", '0xFF', '0sxFF_i32', '0Sb101_u8', '0So77_u8',
  '42', '42_u32', '1_f32', '1_bf16', '1.25_bf16', '1.25e-2_f32', '2E+3_f64',
]) check(`let x = -${literal};`, literal, 'constant.numeric.cadl');
for (const type of ['bf16', 'f32', 'f64', 'i256', 'u32cc', 'usize', 'Instance']) {
  check(`let x: ${type} = 0;`, type, 'storage.type.cadl');
}
for (const builtin of ['$signed', '$unsigned', '$f32', '$f64', '$int', '$uint', 'bitcast']) {
  check(`${builtin}(x);`, builtin, 'support.function.cadl');
}
check('bitcast<u16>(h);', 'u16', 'storage.type.cadl');
for (const name of ['bf16_add', 'fp32_fma_flags', 'bf16_from_i32', 'fp32_narrow_flags']) {
  check(`${name}(x);`, name, 'support.function.cadl');
}
for (const name of ['_irf', '_mem', '_burst_read', '_burst_write', '_csr']) {
  check(`${name}[x];`, name, 'support.variable.cadl');
}
check('rtype example(rs1: u5) {', 'example', 'entity.name.function.cadl');
check('helper(x);', 'helper', 'entity.name.function.cadl');
check('#[partition_dim_array([0, 1])]', 'partition_dim_array', 'entity.name.tag.cadl');
check('#[opcode(7\'b0101011)]', "7'b0101011", 'constant.numeric.cadl');
check('# [opcode(0)]', 'opcode', 'entity.name.tag.cadl');
check('[[unroll(4)]]', 'unroll', 'entity.name.tag.cadl');
check('a[0 +: 16];', '+:', 'keyword.operator.cadl');
check('"flow #[opcode] // 1.0_f32"', 'flow', 'string.quoted.double.cadl');
check('// rtype example 32\'hFF', 'rtype', 'comment.line.double-slash.cadl');
const block = check('/* flow', 'flow', 'comment.block.cadl');
check('1.0_f32 */', '1.0_f32', 'comment.block.cadl', block);
check('my_f32_value = helper2(2);', 'my_f32_value', 'variable.other.cadl');
check('let bf16_addition = 0;', 'bf16_addition', 'variable.other.cadl');

// An attribute's nested array must not close the attribute too early.
let state = grammar.tokenizeLine('#[partition_dim_array([0, 1])]', tm.INITIAL).ruleStack;
check('rtype after_array() {', 'after_array', 'entity.name.function.cadl', state);
state = tm.INITIAL;
for (const line of (await readFile(new URL('fixtures/highlight.cadl', root), 'utf8')).split('\n')) {
  state = grammar.tokenizeLine(line, state).ruleStack;
}
assert.ok(state.equals(grammar.tokenizeLine('', tm.INITIAL).ruleStack),
  'fixture left an unclosed TextMate region');

// Keep reserved-word highlighting in sync with the authoritative Lark terminals.
const lark = await readFile(new URL('../../cadl_frontend/grammar.lark', root), 'utf8');
for (const [, keyword] of lark.matchAll(/^KW_\w+: "(\w+)"/gm)) {
  const expected = ['true', 'false'].includes(keyword) ? 'constant.language.boolean.cadl'
    : keyword === 'bitcast' ? 'support.function.cadl'
    : keyword === 'Instance' || ['static', 'regfile', 'register', 'let'].includes(keyword) ? 'storage.type.cadl'
    : ['flow', 'rtype'].includes(keyword) ? 'storage.type.function.cadl' : 'keyword.control.cadl';
  check(keyword, keyword, expected);
}
console.log(`VS Code: ${checks} scope assertions passed.`);
registry.dispose();
