#include <stdint.h>
volatile int test_status;
extern "C" volatile uint64_t tohost,fromhost;
static uint32_t cycles(){uint32_t x;asm volatile("fence rw,rw; csrr %0,mcycle":"=r"(x)::"memory");return x;}
static void print(const char *label,uint32_t value){
 char text[80];int n=0;while(*label)text[n++]=*label++;text[n++]='=';
 char digits[10];int k=0;do{digits[k++]='0'+value%10;value/=10;}while(value);while(k)text[n++]=digits[--k];text[n++]='\n';
 alignas(64) volatile uint64_t req[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};asm volatile("fence rw,rw":::"memory");tohost=(uintptr_t)req;while(!fromhost){}fromhost=0;asm volatile("fence rw,rw":::"memory");
}

static uint32_t results[3];
__attribute__((noinline)) static uint32_t run_empty(){
 uint32_t x=123456789u,y=3u;
 uint32_t start=cycles();
 asm volatile("li t0,128\n1:\n\n\n\n\n\n\n\n\naddi t0,t0,-1\nbnez t0,1b":"+r"(x):"r"(y):"t0","memory");
 uint32_t elapsed=cycles()-start;results[0]=x;return elapsed;
 }
__attribute__((noinline)) static uint32_t run_native(){
 uint32_t x=123456789u,y=3u;
 uint32_t start=cycles();
 asm volatile("li t0,128\n1:\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\nmul %0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+r"(x):"r"(y):"t0","memory");
 uint32_t elapsed=cycles()-start;results[1]=x;return elapsed;
 }
__attribute__((noinline)) static uint32_t run_ci(){
 uint32_t x=123456789u,y=3u;
 uint32_t start=cycles();
 asm volatile("li t0,128\n1:\n.insn r 43,7,1,%0,%0,%1\n.insn r 43,7,1,%0,%0,%1\n.insn r 43,7,1,%0,%0,%1\n.insn r 43,7,1,%0,%0,%1\n.insn r 43,7,1,%0,%0,%1\n.insn r 43,7,1,%0,%0,%1\n.insn r 43,7,1,%0,%0,%1\n.insn r 43,7,1,%0,%0,%1\naddi t0,t0,-1\nbnez t0,1b":"+r"(x):"r"(y):"t0","memory");
 uint32_t elapsed=cycles()-start;results[2]=x;return elapsed;
 }
extern "C" int main(){
 run_empty();uint32_t empty=run_empty();
 run_native();uint32_t native=run_native();
 run_ci();uint32_t ci=run_ci();
 print("APS_CAL_empty",empty);print("APS_CAL_native_mul",native);print("APS_CAL_ci_mul",ci);
 return results[1]==results[2]?0:1;
}
