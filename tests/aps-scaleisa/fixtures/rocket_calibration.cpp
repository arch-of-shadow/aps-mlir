#include <stdint.h>
volatile int test_status;
extern "C" volatile uint64_t tohost,fromhost;
static uint32_t cycles(){uint32_t x;asm volatile("fence rw,rw; csrr %0,mcycle":"=r"(x)::"memory");return x;}
static void print(const char *label,uint32_t value){
 char text[80];int n=0;while(*label)text[n++]=*label++;text[n++]='=';
 char digits[10];int k=0;do{digits[k++]='0'+value%10;value/=10;}while(value);while(k)text[n++]=digits[--k];text[n++]='\n';
 alignas(64) volatile uint64_t req[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};asm volatile("fence rw,rw":::"memory");tohost=(uintptr_t)req;while(!fromhost){}fromhost=0;asm volatile("fence rw,rw":::"memory");
}
__attribute__((noinline)) static uint32_t run_empty(){
 uint32_t x=123456789u,y=3u;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\n\naddi t0,t0,-1\nbnez t0,1b":"+r"(x):"r"(y):"t0","memory");return cycles()-start;
 }
__attribute__((noinline)) static uint32_t run_alu(){
 uint32_t x=123456789u,y=3u;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\nadd %0,%0,%1\nadd %0,%0,%1\nadd %0,%0,%1\nadd %0,%0,%1\nadd %0,%0,%1\nadd %0,%0,%1\nadd %0,%0,%1\nadd %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+r"(x):"r"(y):"t0","memory");return cycles()-start;
 }
__attribute__((noinline)) static uint32_t run_mul(){
 uint32_t x=123456789u,y=3u;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+r"(x):"r"(y):"t0","memory");return cycles()-start;
 }
__attribute__((noinline)) static uint32_t run_div32(){
 uint32_t x=123456789u,y=3u;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\ndivu %0,%0,%1\ndivu %0,%0,%1\ndivu %0,%0,%1\ndivu %0,%0,%1\ndivu %0,%0,%1\ndivu %0,%0,%1\ndivu %0,%0,%1\ndivu %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+r"(x):"r"(y):"t0","memory");return cycles()-start;
 }
__attribute__((noinline)) static uint32_t run_fp32(){
 float x=1.25f,y=1.00001f;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\nfadd.s %0,%0,%1\nfadd.s %0,%0,%1\nfadd.s %0,%0,%1\nfadd.s %0,%0,%1\nfadd.s %0,%0,%1\nfadd.s %0,%0,%1\nfadd.s %0,%0,%1\nfadd.s %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+f"(x):"f"(y):"t0","memory");return cycles()-start;
 }
__attribute__((noinline)) static uint32_t run_fpmul32(){
 float x=1.25f,y=1.00001f;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\nfmul.s %0,%0,%1\nfmul.s %0,%0,%1\nfmul.s %0,%0,%1\nfmul.s %0,%0,%1\nfmul.s %0,%0,%1\nfmul.s %0,%0,%1\nfmul.s %0,%0,%1\nfmul.s %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+f"(x):"f"(y):"t0","memory");return cycles()-start;
 }
__attribute__((noinline)) static uint32_t run_fpsub32(){
 float x=1.25f,y=1.00001f;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\nfsub.s %0,%0,%1\nfsub.s %0,%0,%1\nfsub.s %0,%0,%1\nfsub.s %0,%0,%1\nfsub.s %0,%0,%1\nfsub.s %0,%0,%1\nfsub.s %0,%0,%1\nfsub.s %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+f"(x):"f"(y):"t0","memory");return cycles()-start;
 }
__attribute__((noinline)) static uint32_t run_divf32(){
 float x=1.25f,y=1.00001f;
 uint32_t start=cycles();asm volatile("li t0,128\n1:\nfdiv.s %0,%0,%1\nfdiv.s %0,%0,%1\nfdiv.s %0,%0,%1\nfdiv.s %0,%0,%1\nfdiv.s %0,%0,%1\nfdiv.s %0,%0,%1\nfdiv.s %0,%0,%1\nfdiv.s %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+f"(x):"f"(y):"t0","memory");return cycles()-start;
 }
extern "C" int main(){
 run_empty();uint32_t empty=run_empty();
 run_alu();uint32_t alu=run_alu();
 run_mul();uint32_t mul=run_mul();
 run_div32();uint32_t div32=run_div32();
 run_fp32();uint32_t fp32=run_fp32();
 run_fpmul32();uint32_t fpmul32=run_fpmul32();
 run_fpsub32();uint32_t fpsub32=run_fpsub32();
 run_divf32();uint32_t divf32=run_divf32();
 print("APS_CAL_empty",empty);
 print("APS_CAL_alu",alu);
 print("APS_CAL_mul",mul);
 print("APS_CAL_div32",div32);
 print("APS_CAL_fp32",fp32);
 print("APS_CAL_fpmul32",fpmul32);
 print("APS_CAL_fpsub32",fpsub32);
 print("APS_CAL_divf32",divf32);
return 0;}
