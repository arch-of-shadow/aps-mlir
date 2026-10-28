static void audit_report();
#include <stdint.h>
static const uint32_t inputs[64][1]={{0u},{1u},{4294967295u},{2147483648u},{2147483647u},{1431655765u},{2863311530u},{16843009u},{4050898376u},{2401453953u},{787661114u},{3431086835u},{1785836716u},{136392293u},{2817566750u},{1145053655u},{3800013712u},{2160006473u},{519999234u},{3208513723u},{1568506484u},{4223466541u},{2583459302u},{935063455u},{3564857688u},{1916461841u},{284843210u},{2931414659u},{1333350460u},{3979921909u},{2348303278u},{691518823u},{3345430304u},{1689694425u},{46541458u},{2687870027u},{1046814212u},{3753154493u},{2105807222u},{464751407u},{3110274280u},{1462927009u},{4106352730u},{2461102611u},{811658188u},{3459278213u},{1812979518u},{237983991u},{2825835184u},{1250839657u},{3838690850u},{2267889627u},{560773524u},{3276550989u},{1569434886u},{4289406655u},{2657788024u},{940186161u},{3670643690u},{1957236131u},{392726364u},{2974286101u},{1394047694u},{3978753159u}};
volatile int test_status,checked_outputs,validation_calls,mismatches;
volatile int first_mismatch=-1;
static uint32_t bits(float x){uint32_t b;__builtin_memcpy(&b,&x,4);return b;}
static int32_t reference(int i){uint32_t x=inputs[i][0]; int32_t n=0; while(x){x&=x-1;++n;} return n;}
extern "C" int scalar_selected_main();
extern "C" int32_t scalar_results[64];
static void scalar_validate(){
 int32_t *values=scalar_results;
 ++validation_calls;
 for(int i=0;i<64;++i){
  if(values[i]!=reference(i)){if(first_mismatch<0)first_mismatch=i;++mismatches;}
  ++checked_outputs;
 }

 audit_report();
}

extern "C" volatile uint64_t tohost, fromhost;
static uint32_t cycles() {
 uint32_t x;
 asm volatile("fence rw,rw; csrr %0,mcycle" : "=r"(x) :: "memory");
 return x;
}
static void print_cycles(uint32_t cold, uint32_t warm) {
 char text[96]; int n=0;
 const char *label="APS_CYCLES cold=";
 while(*label) text[n++]=*label++;
 uint32_t counts[2]={cold,warm};
 for(int j=0;j<2;++j) {
  char digits[10]; int k=0; uint32_t x=counts[j];
  do {digits[k++]='0'+x%10; x/=10;} while(x);
  while(k) text[n++]=digits[--k];
  if(!j) {label=" warm=";while(*label)text[n++]=*label++;}
 }
 text[n++]='\n';
 alignas(64) volatile uint64_t request[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};
 asm volatile("fence rw,rw" ::: "memory");
 tohost=(uintptr_t)request;
 while(!fromhost) {}
 fromhost=0;
 asm volatile("fence rw,rw" ::: "memory");
}

extern "C" int main(){
 uint32_t t0=cycles();
 int result=scalar_selected_main();
 uint32_t cold=cycles()-t0;
 scalar_validate();
 t0=cycles();
 result|=scalar_selected_main();
 uint32_t warm=cycles()-t0;
 scalar_validate();
 print_cycles(cold,warm);
 return result || validation_calls!=2 || checked_outputs!=128 || mismatches;
}

static void audit_report() {
 char text[128]; int n=0;
 const char *label="APS_RESULT_CHECK checked/mismatches/first=";
 while(*label)text[n++]=*label++;
 uint32_t values[3]={(uint32_t)checked_outputs,(uint32_t)mismatches,(uint32_t)first_mismatch};
 const char *hex="0123456789abcdef";
 for(int j=0;j<3;++j) {
  for(int k=7;k>=0;--k)text[n++]=hex[(values[j]>>(k*4))&15];
  text[n++]=j==2?'\n':'/';
 }
 alignas(64) volatile uint64_t request[8]={64,1,(uintptr_t)text,(uint32_t)n,0,0,0,0};
 asm volatile("fence rw,rw" ::: "memory");
 tohost=(uintptr_t)request;
 while(!fromhost){}
 fromhost=0;
 asm volatile("fence rw,rw" ::: "memory");
}
