"""Drive actual adaptation, split instruction emission, RTL and Spike together."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'build/tools/aps-scaleisa-opt/aps-scaleisa-opt'
NAME = 'au_generated_candidate_0__direct'
KERNEL = f'''func.func @{NAME}(%a:i32,%b:i32,%c:i32,%d:i32,%e:i32)->i32
    attributes {{au.generated=true}} {{
  %x=arith.muli %a,%b:i32
  %y=arith.muli %x,%c:i32
  %z=arith.muli %d,%e:i32
  %r=arith.addi %y,%z:i32
  return %r:i32
}}'''
PROGRAM = '''module {
  func.func @main()->i32 {
    %a=arith.constant 2:i32
    %b=arith.constant 3:i32
    %c=arith.constant 4:i32
    %d=arith.constant 5:i32
    %e=arith.constant 6:i32
    %r=func.call @NAME_site_0(%a,%b,%c,%d,%e):(i32,i32,i32,i32,i32)->i32
    return %r:i32
  }
  KERNEL
}'''.replace('NAME', NAME).replace('KERNEL', KERNEL.replace('@'+NAME+'(', '@'+NAME+'_site_0('))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='aps-scaleisa-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.source = self.path / 'input.mlir'
        self.source.write_text(PROGRAM)
        profile = json.loads((ROOT/'hardware/performance/rocket-rv32.json').read_text())
        profile['hardware'] = json.loads((ROOT/'examples/resource_ihp130.json').read_text())
        profile['hardware']['_profile'] = {'clock_ns': 4}
        self.profile = self.path/'target-profile.json'
        self.profile.write_text(json.dumps(profile))
        self.cache = self.path / 'discovery'
        for name in ['final', 'selection', 'uarch/scf']:
            (self.cache / name).mkdir(parents=True)
        (self.cache/'final/selected.mlir').write_text(PROGRAM)
        (self.cache/f'uarch/scf/{NAME}.input.mlir').write_text('module {'+KERNEL+'}')
        (self.cache/'stage-report.json').write_text(json.dumps({
            'status': 'completed', 'metrics': {}, 'config': {
                'input_mlir': str(self.source), 'area_budget': 500,
                'selection_method': 'greedy', 'discovery_stages': ['linalg','affine','scf']}}))
        report_path = self.cache/'stage-report.json'
        report = json.loads(report_path.read_text())
        report['config']['target_profile_sha256'] = hashlib.sha256(self.profile.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report))
        (self.cache/'selection/selection-report.json').write_text(json.dumps({
            'selected_area': 10, 'selected_impls': [
                {'function_name': NAME, 'origin_stage': 'scf'}]}))
        self.out = self.path / 'out'

    def run_driver(self, *args, success=True):
        result = subprocess.run([
            str(DRIVER), str(self.source), '--discovery-dir='+str(self.cache),
            '--target-profile='+str(self.profile),
            '--clock=4',
            '--output-dir='+str(self.out), *map(str, args)],
            cwd=ROOT, env=os.environ.copy(), capture_output=True,
            text=True, timeout=180)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_default_profile_keeps_asap7_physical_delays(self):
        contracts = ROOT/'examples/resource_ihp130.json'
        original = contracts.read_bytes()
        result = subprocess.run([str(DRIVER), str(self.source),
            '--resource='+str(contracts), '--clock=8',
            '--discovery-dir='+str(self.cache), '--output-dir='+str(self.out)],
            cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Discovery target profile does not match', result.stderr)
        physical = json.loads((ROOT/'scaleisa/resources/scaleisa-hector/resource.json').read_text())
        aps = json.loads(original)
        profile = json.loads((self.out/'target-profile.json').read_text())['hardware']
        backend = json.loads((self.out/'resource.json').read_text())
        for name in ('addi', 'muli', 'divsi', 'remsi'):
            self.assertEqual(profile[name]['delay'], physical[name]['delay'])
            self.assertEqual(backend[name]['delay'], physical[name]['delay'])
            self.assertEqual(profile[name]['area'], physical[name]['area'])
            self.assertEqual(profile[name]['latency'], aps[name]['latency'])
        self.assertEqual(profile['_profile']['process'], 'ASAP7 RVT TT')
        self.assertEqual(profile['_profile']['clock_ns'], 8)
        self.assertEqual(contracts.read_bytes(), original)

    def test_default_clock_comes_from_target_profile(self):
        profile = json.loads(self.profile.read_text())
        profile['hardware']['_profile']['clock_ns'] = 7
        self.profile.write_text(json.dumps(profile))
        config_file = self.cache/'stage-report.json'
        config = json.loads(config_file.read_text())
        config['config']['target_profile_sha256'] = hashlib.sha256(self.profile.read_bytes()).hexdigest()
        config_file.write_text(json.dumps(config))
        (self.cache/'final/selected.mlir').write_text(
            'module { func.func @main() -> i32 { %v = arith.constant 5 : i32\n return %v : i32 } }')
        (self.cache/'selection/selection-report.json').write_text(json.dumps({
            'selected_area': 0, 'selected_impls': []}))
        result = subprocess.run([str(DRIVER), str(self.source),
            '--target-profile='+str(self.profile), '--discovery-dir='+str(self.cache),
            '--output-dir='+str(self.out)], cwd=ROOT, env=os.environ.copy(),
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual(json.loads((self.out/'report.json').read_text())['status'], 'passed')

    def test_rejects_mismatched_selection_budget(self):
        result = self.run_driver('--area-budget=501', success=False)
        self.assertIn('configuration does not match', result.stderr)
        self.assertEqual(json.loads((self.out/'report.json').read_text())['status'], 'failed')
        self.assertFalse((self.out/'program.o').exists())

    def test_rejects_stale_target_profile(self):
        profile = json.loads(self.profile.read_text())
        profile['interface']['command_cycles'] += 1
        self.profile.write_text(json.dumps(profile))
        result = self.run_driver(success=False)
        self.assertIn('Discovery target profile does not match', result.stderr)
        self.assertFalse((self.out/'program.o').exists())

    def test_software_selection_compiles_without_synthesizing_rtl(self):
        program = 'module { func.func @main() -> i32 { %v = arith.constant 5 : i32\n return %v : i32 } }'
        (self.cache/'final/selected.mlir').write_text(program)
        (self.cache/'selection/selection-report.json').write_text(json.dumps({
            'selected_area': 0, 'selected_impls': []}))
        self.run_driver()
        report = json.loads((self.out/'report.json').read_text())
        self.assertFalse(report['hardware_selected'])
        self.assertEqual(report['compiled']['ci_sites'], 0)
        self.assertTrue((self.out/'program.o').exists())
        self.assertFalse((self.out/'design.sv').exists())

    def test_rejects_mismatched_discovery_stages(self):
        result = self.run_driver('--discovery-stages=scf', success=False)
        self.assertIn('Discovery stages do not match', result.stderr)
        self.assertFalse((self.out/'program.o').exists())

    def test_single_cycle_response_uses_current_destination(self):
        kernel = f'''func.func @{NAME}(%a:i32,%b:i32)->i32
            attributes {{au.generated=true}} {{
          %r=arith.addi %a,%b:i32
          return %r:i32
        }}'''
        program = f'''module {{
          func.func @main()->i32 {{
            %a=arith.constant 2:i32
            %b=arith.constant 3:i32
            %r=func.call @{NAME}_site_0(%a,%b):(i32,i32)->i32
            return %r:i32
          }}
          {kernel.replace('@'+NAME+'(', '@'+NAME+'_site_0(')}
        }}'''
        self.source.write_text(program)
        (self.cache/'final/selected.mlir').write_text(program)
        (self.cache/f'uarch/scf/{NAME}.input.mlir').write_text('module {'+kernel+'}')
        caller = self.path/'caller.c'
        caller.write_text('volatile int test_status;\nextern int selected_main(void);\n'
                          'int main(void){return selected_main()!=5;}\n')
        # The RTL runner also verifies that the response tag matches the
        # instruction's nonzero destination, as Rocket's scoreboard requires.
        self.run_driver('--emit=elf', '--entry=_start', '--cosim',
            '--rename-main=selected_main', '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        report = json.loads((self.out/'report.json').read_text())
        self.assertEqual(report['cosimulation'], 'passed')
        samples = json.loads((self.out/'ci-runtime.json').read_text())['instructions']
        self.assertEqual(sum(sample['calls'] for sample in samples), 1)
        self.assertTrue(all(sample['min_cycles'] > 0 for sample in samples))
        self.assertEqual(report['ci_cycle_comparison'][0]['total_cycles'], samples[0]['total_cycles'])

    def test_missing_implementation_stops_before_compilation(self):
        (self.cache/f'uarch/scf/{NAME}.input.mlir').unlink()
        self.run_driver(success=False)
        self.assertEqual(json.loads((self.out/'report.json').read_text())['status'], 'failed')
        self.assertFalse((self.out/'program.o').exists())

    def test_split_program_executes_against_generated_rtl(self):
        profile = json.loads(self.profile.read_text())
        profile['interface']['max_extra_csrs'] = 1
        self.profile.write_text(json.dumps(profile))
        path = self.cache/'stage-report.json'
        config = json.loads(path.read_text())
        config['config']['target_profile_sha256'] = hashlib.sha256(self.profile.read_bytes()).hexdigest()
        path.write_text(json.dumps(config))
        caller = self.path/'caller.c'
        caller.write_text('volatile int test_status;\nextern int selected_main(void);\n'
                          'int main(void){return selected_main()!=54;}\n')
        self.run_driver(
            '--max-extra-csrs=1',
            '--emit=elf', '--entry=_start', '--cosim', '--rename-main=selected_main',
            '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        report = json.loads((self.out/'report.json').read_text())
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['cosimulation'], 'passed')
        self.assertEqual(report['compiled']['ci_sites'], 1)
        self.assertEqual(report['compiled']['ci_instructions'], 3)
        self.assertEqual(self.source.read_text(), PROGRAM)
        self.assertIn(NAME+'_site_0', (self.out/'selected.mlir').read_text())
        self.assertTrue((self.out/'program.elf').read_bytes().startswith(b'\x7fELF'))

    def test_observer_rejects_ambiguous_allocations(self):
        program = PROGRAM.replace('func.func @main()->i32 {',
            'func.func @main()->i32 {\n'
            ' %left=memref.alloca():memref<8xi32>\n'
            ' %right=memref.alloca():memref<8xi32>')
        self.source.write_text(program)
        (self.cache/'final/selected.mlir').write_text(program)
        result = self.run_driver('--observe-function=check_arrays',
                                 '--observe-type=memref<8xi32>', success=False)
        self.assertIn('exactly one entry allocation', result.stderr)

    def test_profile_csr_limit_matches_export_configuration(self):
        result = self.run_driver('--max-extra-csrs=1', success=False)
        self.assertIn('max_extra_csrs must match', result.stderr)

    def test_observer_selects_repeated_types_in_requested_order(self):
        program = '''module {func.func @main()->i32 {
          %left=memref.alloca():memref<1xi32>
          %right=memref.alloca():memref<1xi32>
          %a=arith.constant 11:i32
          %b=arith.constant 22:i32
          %zero=arith.constant 0:i32
          affine.store %a,%left[0]:memref<1xi32>
          affine.store %b,%right[0]:memref<1xi32>
          return %zero:i32
        }}'''
        (self.cache/'final/selected.mlir').write_text(program)
        (self.cache/'selection/selection-report.json').write_text(json.dumps({
            'selected_area': 0, 'selected_impls': []}))
        caller = self.path/'caller.c'
        caller.write_text('volatile int test_status, calls, errors;\n'
            'extern int selected_main(void);\n'
            'void check_arrays(int *right,int *left){++calls;errors=(*right!=22||*left!=11);}\n'
            'int main(void){return selected_main()||calls!=1||errors;}\n')
        self.run_driver('--observe-function=check_arrays',
            '--observe-allocation=1', '--observe-allocation=0',
            '--rename-main=selected_main', '--emit=elf', '--entry=_start', '--cosim',
            '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        self.assertEqual(json.loads((self.out/'report.json').read_text())['software_simulation'], 'passed')

    def test_observer_rejects_out_of_range_allocation(self):
        result = self.run_driver('--observe-function=check_arrays',
                                 '--observe-allocation=9', success=False)
        self.assertIn('outside the entry allocation list', result.stderr)

    def test_requested_pipeline_fallback_is_reported_and_correct(self):
        kernel = f'''func.func @{NAME}(%a:memref<8xi32>,%out:memref<1xi32>)
          attributes {{au.generated=true}} {{
          %zero=arith.constant 0:i32
          affine.for %i=0 to 8 {{
            %v=affine.load %a[%i]:memref<8xi32>
            %yes=arith.cmpi sgt,%v,%zero:i32
            scf.if %yes {{affine.store %v,%out[0]:memref<1xi32>}}
          }} {{pipeline=0:i64}}
          return
        }}'''
        program = f'''module {{func.func @main()->i32 {{
          %a=memref.alloca():memref<8xi32>
          %out=memref.alloca():memref<1xi32>
          %four=arith.constant 4:i32
          affine.store %four,%out[0]:memref<1xi32>
          affine.for %i=0 to 8 {{
            %iv=arith.index_cast %i:index to i32
            %v=arith.subi %iv,%four:i32
            affine.store %v,%a[%i]:memref<8xi32>
          }}
          func.call @{NAME}_site_0(%a,%out):(memref<8xi32>,memref<1xi32>)->()
          %r=affine.load %out[0]:memref<1xi32>
          return %r:i32
        }} {kernel.replace('@'+NAME+'(', '@'+NAME+'_site_0(')} }}'''
        self.source.write_text(program)
        (self.cache/'final/selected.mlir').write_text(program)
        directory = self.cache/'uarch/affine'
        directory.mkdir()
        (directory/(NAME+'.input.mlir')).write_text('module {'+kernel+'}')
        path = self.cache/'selection/selection-report.json'
        report = json.loads(path.read_text())
        report['selected_impls'][0]['origin_stage'] = 'affine'
        path.write_text(json.dumps(report))
        caller = self.path/'caller.c'
        caller.write_text('volatile int test_status;\nextern int selected_main(void);\n'
                          'int main(void){return selected_main()!=3;}\n')
        self.run_driver('--rename-main=selected_main', '--emit=elf', '--entry=_start', '--cosim',
            '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        result = json.loads((self.out/'report.json').read_text())
        self.assertEqual(result['cosimulation'], 'passed')
        self.assertEqual(result['backend_pipeline_fallback_count'], 1)
        self.assertEqual(result['backend_pipeline_outcomes'][0]['realized_mode'], 'sequential')

    def memory_program(self, alias=False, banked=False, single_bank=False):
        kernel = f'''func.func @{NAME}(%a:memref<8xi32>,%b:memref<16xi32>)
          attributes {{au.generated=true}} {{
          %one=arith.constant 1:i32
          affine.for %i=0 to 8 {{
            %v=affine.load %a[%i]:memref<8xi32>
            %w=arith.addi %v,%one:i32
            affine.store %w,%b[%i]:memref<16xi32>
          }}
          return
        }}'''
        if single_bank:
            kernel = kernel.replace('au.generated=true',
                'au.generated=true, partition_dim_array_0=[0:i64], '
                'partition_factor_array_0=[1:i64], partition_cyclic_array_0=[true], '
                'partition_dim_array_1=[0:i64], partition_factor_array_1=[1:i64], '
                'partition_cyclic_array_1=[true]')
        if banked:
            kernel = kernel.replace('au.generated=true',
                'au.generated=true, partition_dim_array_0=[0:i64], '
                'partition_factor_array_0=[2:i64], partition_cyclic_array_0=[true], '
                'partition_dim_array_1=[0:i64], partition_factor_array_1=[2:i64], '
                'partition_cyclic_array_1=[true]')
            kernel = kernel.replace('          }\n          return',
                                    '          } {unroll=2:i64, pipeline=0:i64}\n          return')
        main = f'''func.func @main()->i32 {{
          %a=memref.alloca():memref<8xi32>
          %b=memref.alloca():memref<16xi32>
          %zero=arith.constant 0:i32
          %seven=arith.constant 7:i32
          %sentinel=arith.constant 324508639:i32
          affine.for %i=0 to 8 {{
            %idx=arith.index_cast %i:index to i32
            %v=arith.muli %idx,%seven:i32
            affine.store %v,%a[%i]:memref<8xi32>
          }}
          affine.for %i=0 to 16 {{
            affine.store %sentinel,%b[%i]:memref<16xi32>
          }}
          func.call @{NAME}_site_0(%a,%b):(memref<8xi32>,memref<16xi32>)->()
          return %zero:i32
        }}'''
        if alias:
            kernel = kernel.replace('16xi32', '8xi32')
            main = main.replace('16xi32', '8xi32').replace('0 to 16', '0 to 8')
            main = main.replace('_site_0(%a,%b)', '_site_0(%a,%a)')
        program = 'module {'+main+kernel.replace('@'+NAME+'(', '@'+NAME+'_site_0(')+'}'
        self.source.write_text(program)
        (self.cache/'final/selected.mlir').write_text(program)
        directory = self.cache/'uarch/affine'
        directory.mkdir()
        (directory/(NAME+'.input.mlir')).write_text('module {'+kernel+'}')
        path = self.cache/'selection/selection-report.json'
        report = json.loads(path.read_text())
        report['selected_impls'][0]['origin_stage'] = 'affine'
        path.write_text(json.dumps(report))

    def test_aliased_writable_operands_are_rejected(self):
        self.memory_program(alias=True)
        result = self.run_driver(success=False)
        self.assertIn('scaleisa.noalias', result.stderr)
        self.assertEqual(json.loads((self.out/'report.json').read_text())['status'], 'failed')

    def global_memory_program(self, alias=False):
        self.memory_program(alias=alias)
        for path in (self.source, self.cache/'final/selected.mlir'):
            text = path.read_text()
            b_size = 8 if alias else 16
            text = text.replace('module {',
                'module {memref.global @global_a : memref<8xi32> = uninitialized\n'
                f'memref.global @global_b : memref<{b_size}xi32> = uninitialized\n', 1)
            text = text.replace('%a=memref.alloca():memref<8xi32>',
                                '%a=memref.get_global @global_a:memref<8xi32>')
            text = text.replace(f'%b=memref.alloca():memref<{b_size}xi32>',
                                f'%b=memref.get_global @global_b:memref<{b_size}xi32>')
            path.write_text(text)

    def test_distinct_global_buffers_are_proven_and_correct(self):
        self.global_memory_program()
        caller = self.path/'global-memory.c'
        caller.write_text('''volatile int test_status;
          extern int selected_main(void),global_a[8],global_b[16];
          int main(void){
            if(selected_main())return 1;
            for(int i=0;i<8;++i)if(global_a[i]!=7*i || global_b[i]!=7*i+1)return 2;
            for(int i=8;i<16;++i)if(global_b[i]!=324508639)return 3;
            return 0;
          }
        ''')
        self.run_driver('--emit=elf', '--entry=_start', '--cosim',
            '--rename-main=selected_main', '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        report = json.loads((self.out/'report.json').read_text())
        self.assertEqual(report['cosimulation'], 'passed')
        self.assertIn(NAME, report['noalias_proofs'])

    def test_repeated_global_writable_operand_is_rejected(self):
        self.global_memory_program(alias=True)
        result = self.run_driver(success=False)
        self.assertIn('scaleisa.noalias', result.stderr)

    def test_memory_kernel_preserves_outputs_and_canaries(self):
        self.check_memory_kernel()

    def test_single_bank_hint_preserves_nonzero_dma_inputs_and_outputs(self):
        self.check_memory_kernel(single_bank=True)
        scheduled = (self.out/'design.scheduled-tor.mlir').read_text()
        self.assertIn('count(2), cyclic(1)', scheduled)

    def test_nested_affine_indices_preserve_dma_banks_and_values(self):
        kernel = f'''func.func @{NAME}(%a:memref<8x3xi32>) attributes {{au.generated=true}} {{
          %one=arith.constant 1:i32
          affine.for %i=0 to 8 {{
            affine.for %j=0 to 3 {{
              %v=affine.load %a[%i,%j]:memref<8x3xi32>
              %w=arith.addi %v,%one:i32
              affine.store %w,%a[%i,%j]:memref<8x3xi32>
            }} {{pipeline=0:i64}}
          }}
          return
        }}'''
        program = f'''module {{func.func @main(%a:memref<8x3xi32>)->i32 {{
          func.call @{NAME}_site_0(%a):(memref<8x3xi32>)->()
          %zero=arith.constant 0:i32
          return %zero:i32
        }}{kernel.replace('@'+NAME+'(', '@'+NAME+'_site_0(')}}}'''
        self.source.write_text(program)
        (self.cache/'final/selected.mlir').write_text(program)
        directory = self.cache/'uarch/affine'
        directory.mkdir()
        (directory/(NAME+'.input.mlir')).write_text('module {'+kernel+'}')
        path = self.cache/'selection/selection-report.json'
        report = json.loads(path.read_text())
        report['selected_impls'][0]['origin_stage'] = 'affine'
        path.write_text(json.dumps(report))
        caller = self.path/'nested.c'
        caller.write_text('''volatile int test_status;
          int guarded[32] __attribute__((aligned(64)));
          extern int selected_main(int *);
          int main(void){
            for(int i=0;i<32;++i)guarded[i]=11*i+7;
            if(selected_main(guarded+4))return 1;
            for(int i=0;i<32;++i)if(guarded[i]!=11*i+7+(i>=4&&i<28))return 2;
            return 0;
          }
        ''')
        self.run_driver('--cosim', '--emit=elf', '--entry=_start',
            '--rename-main=selected_main', '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        self.assertIn('count(2), cyclic(1)', (self.out/'design.scheduled-tor.mlir').read_text())

    def test_scalar_reference_ci_imports_and_writes_back(self):
        kernel = f'''func.func @{NAME}(%a:memref<i32>)->i32
          attributes {{au.generated=true}} {{
          %one=arith.constant 1:i32
          %v=memref.load %a[]:memref<i32>
          %r=arith.addi %v,%one:i32
          memref.store %r,%a[]:memref<i32>
          return %r:i32
        }}'''
        program = f'''module {{func.func @main()->i32 {{
          %a=memref.alloca():memref<i32>
          %v=arith.constant 41:i32
          memref.store %v,%a[]:memref<i32>
          %r=func.call @{NAME}_site_0(%a):(memref<i32>)->i32
          %stored=memref.load %a[]:memref<i32>
          %sum=arith.addi %r,%stored:i32
          %expected=arith.constant 84:i32
          %error=arith.subi %sum,%expected:i32
          return %error:i32
        }}{kernel.replace('@'+NAME+'(', '@'+NAME+'_site_0(')}}}'''
        self.source.write_text(program)
        (self.cache/'final/selected.mlir').write_text(program)
        (self.cache/f'uarch/scf/{NAME}.input.mlir').write_text('module {'+kernel+'}')
        caller = self.path/'scalar.c'
        caller.write_text('volatile int test_status; extern int selected_main(void); '
                          'int main(void){return selected_main();}')
        self.run_driver('--cosim', '--emit=elf', '--entry=_start',
            '--rename-main=selected_main', '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))

    def test_odd_word_buffer_transfers_preserve_surrounding_canaries(self):
        for count in (3, 5):
            with self.subTest(words=count):
                typ = f'memref<{count}xi32>'
                kernel = f'''func.func @{NAME}(%a:{typ}) attributes {{au.generated=true}} {{
                  %one=arith.constant 1:i32
                  affine.for %i=0 to {count} {{
                    %v=affine.load %a[%i]:{typ}
                    %r=arith.addi %v,%one:i32
                    affine.store %r,%a[%i]:{typ}
                  }}
                  return
                }}'''
                program = f'''module {{
                  func.func @main(%a:{typ})->i32 {{
                    func.call @{NAME}_site_0(%a):({typ})->()
                    %zero=arith.constant 0:i32
                    return %zero:i32
                  }}
                  {kernel.replace('@'+NAME+'(', '@'+NAME+'_site_0(')}
                }}'''
                self.source.write_text(program)
                (self.cache/'final/selected.mlir').write_text(program)
                (self.cache/f'uarch/scf/{NAME}.input.mlir').write_text('module {'+kernel+'}')
                caller = self.path/'tail.c'
                caller.write_text(f'''volatile int test_status;
                  int guarded_storage[{count+8}] __attribute__((aligned(64)));
                  extern int selected_main(int *);
                  int main(void) {{
                    for(int i=0;i<{count+8};++i)guarded_storage[i]=100+i;
                    if(selected_main(guarded_storage+4))return 1;
                    for(int i=0;i<{count+8};++i)
                      if(guarded_storage[i]!=100+i+(i>=4 && i<{count+4}))return 2;
                    return 0;
                  }}''')
                self.run_driver('--cosim', '--emit=elf', '--entry=_start',
                    '--rename-main=selected_main', '--link-source='+str(caller),
                    '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
                    '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))

    def test_banked_unrolled_pipeline_preserves_outputs_and_canaries(self):
        self.check_memory_kernel(banked=True)
        hardware = (self.out/'hardware.mlir').read_text()
        self.assertIn('partition_factor_array = [2]', hardware)
        self.assertIn('pipeline = 1', hardware)
        self.assertIn('_issue_rule', (self.out/'design.cmt2.mlir').read_text())

    def test_large_banked_copy_loops_preserve_nonzero_data_and_canaries(self):
        self.memory_program(banked=True)
        for path in (self.source, self.cache/'final/selected.mlir',
                     self.cache/f'uarch/affine/{NAME}.input.mlir'):
            text = path.read_text().replace('8xi32', '4100xi32').replace('16xi32', '8204xi32')
            text = re.sub(r'0 to 8\b', '0 to 4100', text)
            text = re.sub(r'0 to 16\b', '0 to 8204', text)
            path.write_text(text)
        caller = self.path/'large-memory.c'
        caller.write_text('''volatile int test_status;
          extern int selected_main(void);
          static int bad,calls;
          void check_arrays(int *a,int *b){
            ++calls;
            for(int i=0;i<4100;++i)if(a[i]!=7*i)bad=1;
            for(int i=0;i<4100;++i)if(b[i]!=7*i+1)bad=1;
            for(int i=4100;i<8204;++i)if(b[i]!=324508639)bad=1;
          }
          int main(void){int r=selected_main();return r || bad || calls!=1;}
        ''')
        self.run_driver('--emit=elf', '--entry=_start', '--cosim',
            '--rename-main=selected_main', '--observe-function=check_arrays',
            '--observe-type=memref<4100xi32>', '--observe-type=memref<8204xi32>',
            '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        self.assertEqual(json.loads((self.out/'report.json').read_text())['cosimulation'], 'passed')
        self.assertLess((self.out/'design.cmt2.mlir').stat().st_size, 3_000_000)

    def check_memory_kernel(self, banked=False, single_bank=False):
        self.memory_program(banked=banked, single_bank=single_bank)
        caller = self.path/'memory.c'
        caller.write_text('''volatile int test_status;
          extern int selected_main(void);
          static int bad,calls;
          void check_arrays(int *a,int *b){
            ++calls;
            for(int i=0;i<8;++i)if(a[i]!=7*i || b[i]!=7*i+1)bad=1;
            for(int i=8;i<16;++i)if(b[i]!=324508639)bad=1;
          }
          int main(void){int r=selected_main();return r || bad || calls!=1;}
        ''')
        self.run_driver(
            '--emit=elf', '--entry=_start', '--cosim', '--rename-main=selected_main',
            '--observe-function=check_arrays', '--observe-type=memref<8xi32>',
            '--observe-type=memref<16xi32>', '--link-source='+str(caller),
            '--link-source='+str(ROOT/'tests/scaleisa-compiler/fixtures/start.S'),
            '--link-arg=-T'+str(ROOT/'tests/scaleisa-compiler/fixtures/ram.ld'))
        report = json.loads((self.out/'report.json').read_text())
        self.assertEqual(report['cosimulation'], 'passed')
        self.assertIn(NAME, report['noalias_proofs'])


if __name__ == '__main__':
    unittest.main()
