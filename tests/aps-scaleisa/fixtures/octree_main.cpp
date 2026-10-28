#define main octree_reference_unused
#include "../../../scaleisa/benchmarks/pcl/octree_search/octree_search.cpp"
#undef main
#include <stdint.h>
static float expected_points[POINTS][3];
static int expected_counts[LEVELS][CELLS],expected_scores[QUERIES];
volatile int test_status,checked_outputs,validation_calls,mismatches;
volatile int first_mismatch=-1;
volatile uint32_t first_actual,first_expected;
extern "C" int pcl_selected_main();
static uint32_t bits(float f){uint32_t b;__builtin_memcpy(&b,&f,4);return b;}
static void check(uint32_t a,uint32_t b){
 if(a!=b){if(first_mismatch<0){first_mismatch=checked_outputs;first_actual=a;first_expected=b;}++mismatches;}
 ++checked_outputs;
}
extern "C" void pcl_validate_outputs(float *points,int *counts,int *scores){
 ++validation_calls;
 for(int i=0;i<POINTS*3;++i)check(bits(points[i]),bits(expected_points[i/3][i%3]));
 for(int i=0;i<LEVELS*CELLS;++i)check(counts[i],expected_counts[i/CELLS][i%CELLS]);
 for(int i=0;i<QUERIES;++i)check(scores[i],expected_scores[i]);
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
 int actual=pcl_selected_main();
 if(validation_calls!=1 || checked_outputs!=22656)return 3;
 if(mismatches)return 1;
 if(actual!=expected)return 2;
 return 0;
}
