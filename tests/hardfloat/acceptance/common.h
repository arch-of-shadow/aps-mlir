#ifndef AQUAS_HARDFLOAT_ACCEPTANCE_H
#define AQUAS_HARDFLOAT_ACCEPTANCE_H
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
/* The RTL simulation implements this ABI by driving integer RoCC ports.
   Arithmetic is exclusively in synthesized ISAX RTL, never a CPU FPU. */
uint32_t isax_execute(uint32_t a, uint32_t b);
int run_test(const char *vectors);
#ifdef __cplusplus
}
#endif
static int check_vectors(const char *name, const char *vectors) {
  FILE *f=fopen(vectors,"r");
  if(!f) {perror(vectors);return 1;}
  uint32_t a,b,expected; unsigned count=0;
  while(fscanf(f,"%" SCNx32 " %" SCNx32 " %" SCNx32,&a,&b,&expected)==3) {
    uint32_t got=isax_execute(a,b);
    if(got!=expected) {
      fprintf(stderr,"%s vector=%u a=%08" PRIx32 " b=%08" PRIx32 " expected=%08" PRIx32 " got=%08" PRIx32 "\n",name,count,a,b,expected,got);
      fclose(f);return 1;
    }
    ++count;
  }
  int malformed=!feof(f);fclose(f);
  if(malformed||!count) {fprintf(stderr,"invalid/empty golden file\n");return 1;}
  printf("PASS %s requests=%u\n",name,count);return 0;
}
#endif
