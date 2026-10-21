// Generated scratchpad protocol regression, not a GEMM accelerator model.
#include "VScratchpadMemoryPool.h"
#include "verilated.h"
#include <deque>
#include <random>
#include <cstdio>
#include <cstdlib>
static void check(bool ok, const char *what, unsigned cycle) {
  if (!ok) { fprintf(stderr,"FAIL %s cycle=%u\n",what,cycle); exit(1); }
}
int main(int argc,char **argv) {
  Verilated::commandArgs(argc,argv);
  VScratchpadMemoryPool d;
  std::mt19937 rng(argc>1 ? strtoul(argv[1],nullptr,0) : 1);
  auto edge=[&] {d.clk=0;d.eval();d.clk=1;d.eval();d.clk=0;d.eval();};
  d.rst=1; edge(); d.rst=0;
  for(unsigned i=0;i<32;++i) {
    d.data_write_enable=1;d.data_write_addr=i;d.data_write_data=0x12340000+i;
    d.eval();check(d.data_write_ready,"initial write ready",i);edge();
  }
  d.data_write_enable=0;
  // A false masked store leaves the bank unchanged. Readiness conservatively
  // reserves the bank even for false predicates.
  d.burst_write_enable=0; d.burst_write_addr=0;
  d.burst_write_data=0x1234000112340000ULL;
  d.data_write_if_enable=1; d.data_write_if_condition=0;
  d.data_write_if_addr=3; d.data_write_if_data=0xBAD;
  d.eval(); check(d.data_write_if_ready,"idle masked write ready",0);
  edge();
  d.burst_write_enable=1; d.data_write_if_condition=1; d.eval();
  check(!d.data_write_if_ready,"selected write conflicts with DMA",0);
  d.data_write_if_enable=0; d.burst_write_enable=0;

  std::deque<unsigned> q[2]; unsigned sent[2]={0,0}, received[2]={0,0};
  unsigned resets=0;
  for(unsigned c=0;c<10000;++c) {
    bool reset=c==257 || c==1021;
    d.rst=reset;
    // Leave site 0's FIFO full while site 1 continues. Include simultaneous
    // issue/collect, read address changes and reset with responses pending.
    d.data_client0_issue_enable=!reset && sent[0]<500 && rng()%4;
    d.data_client1_issue_enable=!reset && sent[1]<500 && rng()%4;
    d.data_client0_issue_addr=rng()%32;d.data_client1_issue_addr=rng()%32;
    d.data_client0_collect_enable=!reset && c%211>120 && rng()%3;
    d.data_client1_collect_enable=!reset && c%73>9 && rng()%3;
    d.eval();
    if(reset) {q[0].clear();q[1].clear();++resets;}
    else {
      if(d.data_client0_collect_enable && d.data_client0_collect_ready) {
        check(!q[0].empty(),"unexpected response 0",c);
        if(d.data_client0_collect_!=q[0].front()) fprintf(stderr,"site0 got=%08x expected=%08x\n",d.data_client0_collect_,q[0].front());
        check(d.data_client0_collect_==q[0].front(),"response data/site/order 0",c);
        q[0].pop_front();++received[0];
      }
      if(d.data_client1_collect_enable && d.data_client1_collect_ready) {
        check(!q[1].empty(),"unexpected response 1",c);
        check(d.data_client1_collect_==q[1].front(),"response data/site/order 1",c);
        q[1].pop_front();++received[1];
      }
      if(d.data_client0_issue_enable && d.data_client0_issue_ready) {
        q[0].push_back(0x12340000+d.data_client0_issue_addr);++sent[0];
      }
      if(d.data_client1_issue_enable && d.data_client1_issue_ready) {
        q[1].push_back(0x12340000+d.data_client1_issue_addr);++sent[1];
      }
      check(q[0].size()<=2 && q[1].size()<=2,"reservation overflow",c);
    }
    edge();
    if(sent[0]==500 && sent[1]==500 && q[0].empty() && q[1].empty()) {
      check(received[0]>490 && received[1]>490 && resets==2,"coverage",c);
      puts("PASS read channels: stalls, credits, ordering, reset");return 0;
    }
  }
  check(false,"deadlock",10000);
}
