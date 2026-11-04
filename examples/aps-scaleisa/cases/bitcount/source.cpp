extern "C" { int scalar_results[64]; }
static int bitcount(unsigned int value) {
  int count=0;
  for (int bit=0;bit<32;++bit) count+=(value>>bit)&1u;
  return count;
}
int main() {
  const unsigned int edges[8]={0u,1u,0xffffffffu,0x80000000u,0x7fffffffu,0x55555555u,0xaaaaaaaau,0x01010101u};
  for (unsigned int i=0;i<64;++i) {
    unsigned int value=i<8?edges[i]:((i*0x9e3779b9u)^((i+17u)<<19));
    scalar_results[i]=bitcount(value);
  }
  return 0;
}
