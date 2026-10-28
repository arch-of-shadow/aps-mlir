#include <stdint.h>
// Octree's selected ABI: opcode 43/funct7 0 zeros a 16384-word array;
// funct7 1 multiplies two i32 operands. Run against the exported octree RTL.
extern "C" { volatile int test_status; }
alignas(64) static uint32_t values[16384];
static void clear_au(){
 asm volatile("fence rw,rw\n.insn r 43,2,0,zero,%0,zero\nfence rw,rw" :: "r"(values) : "memory");
}
extern "C" int main(){
 for(int pass=0;pass<2;++pass){
  for(int i=0;i<16384;++i)values[i]=7*i+pass+1;
  clear_au();
  for(int i=0;i<16384;++i)if(values[i])return 1+pass;
 }
 uint32_t a=7,b=13,r;
 asm volatile(".insn r 43,7,1,%0,%1,%2":"=r"(r):"r"(a),"r"(b):"memory");
 return r==91?0:3;
}
