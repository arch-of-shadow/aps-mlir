.section .bss
.balign 16
stack_bottom:
.skip 131072
stack_top:
.section .text.start
.globl _start
_start:
  la sp, stack_top
  # Enable floating-point state before the C caller uses RV32F instructions.
  li t0, 0x6000
  csrs mstatus, t0
  csrw fcsr, zero
  call main
  la t0, test_status
  sw a0, 0(t0)
  # Spike HTIF exit: 1 means success, (status << 1) | 1 reports failure.
  slli a0, a0, 1
  ori a0, a0, 1
  la t0, tohost
  sw zero, 4(t0)
  sw a0, 0(t0)
1:
  wfi
  j 1b

.section .tohost,"aw",@progbits
.balign 64
.globl tohost
tohost:
  .dword 0
.balign 64
.globl fromhost
fromhost:
  .dword 0
