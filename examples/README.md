# APS examples

Run commands from the repository root with Pixi.

| Location | Contents |
|---|---|
| [aquas/](aquas/) | CADL, memory, graphics, cryptography, and floating-point examples |
| [megg-compiler/](megg-compiler/) | CADL instruction descriptions and C callers for Megg compilation |
| [aspdac26-tutorial/](aspdac26-tutorial/) | ASP-DAC 2026 scripts, CADL/C inputs, and interactive compiler tutorial |
| [aps-scaleisa/](aps-scaleisa/) | 16 cases with discovery, hardware/software generation, independent validation, and optional Rocket measurements |

Shared configuration stays here: [interface.json](interface.json),
[resource_ihp130.json](resource_ihp130.json), [resource.json](resource.json), and
[scaleisa-aps-model.json](scaleisa-aps-model.json).

For a first CADL instruction, follow the [synthesis guide](../docs/SYNTHESIS/README.md).
For the application-driven examples:

```sh
make -C examples/aps-scaleisa list
make -C examples/aps-scaleisa CASE=bitcount run
```

Generated outputs belong under `build/` or an ignored output directory.

Each group provides a Makefile: `make -C examples/aquas help`,
`make -C examples/megg-compiler help`,
`make -C examples/aspdac26-tutorial help`, and
`make -C examples/aps-scaleisa help`.
