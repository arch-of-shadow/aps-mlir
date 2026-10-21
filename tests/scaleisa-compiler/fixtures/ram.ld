OUTPUT_ARCH(riscv)
ENTRY(_start)
MEMORY {
  RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 16M
}
SECTIONS {
  .text : ALIGN(16) {
    KEEP(*(.text.start))
    *(.text .text.*)
  } > RAM
  .rodata : ALIGN(16) { *(.rodata .rodata.*) } > RAM
  .data : ALIGN(16) { *(.data .data.* .sdata .sdata.*) } > RAM
  .tohost : ALIGN(64) { KEEP(*(.tohost)) } > RAM
  .bss (NOLOAD) : ALIGN(16) {
    *(.bss .bss.* .sbss .sbss.*)
    *(COMMON)
  } > RAM
}
