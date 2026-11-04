"""The installed CMT2 HardFloat package must run independently of APS."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_installed_package_is_self_contained(tmp_path, monkeypatch):
    circt = tmp_path / 'standalone-circt'
    library = circt / 'lib/Dialect/Cmt2/ModuleLibrary'
    library.mkdir(parents=True)
    (library / 'manifest.yaml').write_text('modules: [{name: Reg, path: firrtl/reg}]\n')
    installer = runpy.run_path(str(ROOT / 'thirdparty/HardFloat/install.py'))
    package = installer['install'](circt)
    import yaml
    manifest = yaml.safe_load((circt / 'build/share/circt/ModuleLibrary/manifest.yaml').read_text())
    modules = {entry['name']: entry for entry in manifest['modules']}
    assert {'Reg', 'HardFloat', 'HardFloat32'} == set(modules)
    runtime_library = circt / 'build/share/circt/ModuleLibrary'
    assert (runtime_library / modules['Reg']['path']).resolve() == library / 'firrtl/reg'
    assert (runtime_library / modules['HardFloat']['path'] / 'build.sh').is_file()
    assert not (library / 'firrtl/hardfloat_ip').exists()
    builder = runpy.run_path(str(package / 'build-hardfloat-ip.py'))
    assert builder['INSTALLED']
    assert builder['ROOT'] == package
    assert builder['CIRCT'] == circt
    builder['verify_sources']()
    for name in installer['WRAPPERS']:
        assert (package / (name + '.sv')).read_bytes() == (
            ROOT / 'thirdparty/HardFloat' / (name + '.sv')).read_bytes()

    output = tmp_path / 'generated'
    output.mkdir()
    monkeypatch.chdir(output)
    globals_ = builder['main'].__globals__

    # Exercise packaging and resources without running synthesis tools.
    def emit_firrtl(name, bundle, specialization):
        assert bundle.is_relative_to(package)
        assert specialization.is_relative_to(package)
        Path(name + '.mlir').write_text('module {}\n')

    def unexpected_subprocess(*args, **kwargs):
        raise AssertionError('Installed package called an external builder')

    monkeypatch.setitem(globals_, 'emit_firrtl', emit_firrtl)
    monkeypatch.setattr(builder['subprocess'], 'run', unexpected_subprocess)
    monkeypatch.setattr(sys, 'argv', ['build-hardfloat-ip.py', '--resource-output=resource.json'])
    builder['main']()
    assert (output / 'resource.json').is_file()
    assert (output / 'HardFloat32_op0_l3_q5.mlir').is_file()
    assert str(ROOT) not in (output / 'HardFloat32_op0_l3_q5.f').read_text()

    monkeypatch.setattr(sys, 'argv', ['build-hardfloat-ip.py', 'width=16', 'op=5'])
    builder['main']()
    assert (output / 'HardFloat_w16_o5_p0_l3_q5.mlir').is_file()
    assert str(ROOT) not in (output / 'HardFloat_w16_o5_p0_l3_q5.f').read_text()
