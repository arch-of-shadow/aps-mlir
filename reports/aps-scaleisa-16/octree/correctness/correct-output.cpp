static void audit_report();
#define main octree_reference_unused
#include "/newdisk/aps-mlir/scaleisa/benchmarks/pcl/octree_search/octree_search.cpp"
#undef main
#include <stdint.h>
static float expected_points[POINTS][3];
static int expected_counts[LEVELS][CELLS],expected_scores[QUERIES];
volatile int test_status,checked_outputs,validation_calls,mismatches;
volatile int first_mismatch=-1;
volatile uint32_t first_actual,first_expected;

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

static uint32_t end_cycle;
extern "C" int pcl_selected_main();
static uint32_t bits(float f){uint32_t b;__builtin_memcpy(&b,&f,4);return b;}
static void check(uint32_t a,uint32_t b){
 if(a!=b){if(first_mismatch<0){first_mismatch=checked_outputs;first_actual=a;first_expected=b;}++mismatches;}
 ++checked_outputs;
}
extern "C" void pcl_validate_outputs(float *points,int *counts,int *scores){
 end_cycle=cycles();
 ++validation_calls;
 for(int i=0;i<POINTS*3;++i)check(bits(points[i]),bits(expected_points[i/3][i%3]));
 for(int i=0;i<LEVELS*CELLS;++i)check(counts[i],expected_counts[i/CELLS][i%CELLS]);
 for(int i=0;i<QUERIES;++i)check(scores[i],expected_scores[i]);

 audit_report();
}
extern "C" int main(){
 for(int i=0;i<POINTS;++i){
  expected_points[i][0]=(float)((i*13+7)%127)/127.0f;
  expected_points[i][1]=(float)((i*17+11)%131)/131.0f;
  expected_points[i][2]=(float)((i*19+5)%137)/137.0f;
 }
 build_level_counts(expected_points,expected_counts);
 query_neighbors(expected_points,expected_counts,expected_scores);
 int expected=0;for(int i=0;i<QUERIES;++i)expected+=expected_scores[i]*((i&7)+1);
 if(expected!=17565906)return 4;
 uint32_t start_cycle=cycles();
 int actual=pcl_selected_main();
 uint32_t cold=end_cycle-start_cycle;
 if(actual!=expected || mismatches)return 2;
 start_cycle=cycles();
 actual=pcl_selected_main();
 uint32_t warm=end_cycle-start_cycle;
 print_cycles(cold,warm);
 if(validation_calls!=2 || checked_outputs!=45312)return 3;
 if(mismatches)return 1;
 if(actual!=expected)return 2;
 return 0;
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
