#include "VHFUnit.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <random>
#include <vector>
struct Row{uint32_t a,b,c,rm,data,flags;uint64_t issued;bool observed=false;};
static void require(bool ok,const char*why,uint64_t cycle){if(!ok){std::fprintf(stderr,"FAIL cycle=%llu %s\n",(unsigned long long)cycle,why);std::exit(1);}}
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);require(argc==5,"expected vectors latency variable capacity",0);
 uint64_t latency_bound=std::strtoull(argv[2],nullptr,10);
 bool variable=std::strtoul(argv[3],nullptr,10)!=0;
 unsigned capacity=std::strtoul(argv[4],nullptr,10);
 std::ifstream f(argv[1]);require(f.good(),"cannot open vectors",0);std::vector<Row> rows;Row v;
 while(f>>std::hex>>v.a>>v.b>>v.c>>v.rm>>v.data>>v.flags)rows.push_back(v);
 require(f.eof()&&!rows.empty(),"invalid/empty vectors",0);
 VHFUnit d;std::mt19937 rng(0x163204);uint64_t cycle=0;size_t next=0,accepted=0,consumed=0,discarded=0;
 auto edge=[&](){d.clock=0;d.eval();d.clock=1;d.eval();d.clock=0;d.eval();++cycle;};
 d.reset=1;d.issue_enable=d.collect_enable=0;edge();edge();
 uint64_t min_ready=~uint64_t(0),max_ready=0;size_t peak=0;
 std::deque<Row> pending;bool stalled=false;uint32_t previous_data=0,previous_flags=0;
 while(next<rows.size()||!pending.empty()){
   require(cycle<rows.size()*300+5000,"timeout",cycle);
   bool reset=cycle>1000&&cycle<7000&&cycle%1201==0;
   d.reset=reset;d.issue_enable=!reset&&next<rows.size()&&(cycle<100 || rng()%7!=0);
   d.collect_enable=!reset&&(cycle<100 || (cycle%257<155&&rng()%4!=0));
   if(next<rows.size()){auto&r=rows[next];d.operand0=r.a;d.operand1=r.b;d.operand2=r.c;d.rounding_mode=r.rm;}
   d.eval();
   if(reset){
     // Invalidated requests are retried: exhaustive vectors must all have
     // a numerically checked response, not merely an accepted issue.
     discarded+=pending.size();next-=pending.size();pending.clear();stalled=false;
   }
   else {
     if(variable && d.collect_ready && !pending.empty() && !pending.front().observed){
       auto &r=pending.front();r.observed=true;auto age=cycle-r.issued;
       if(age<min_ready)min_ready=age;if(age>max_ready)max_ready=age;
     }
     if(cycle<100 && !variable && next<rows.size())require(d.issue_ready,"native fixed pipeline did not sustain II=1",cycle);
     if(d.collect_ready && !pending.empty() && !variable)require(cycle-pending.front().issued>=latency_bound,"response before native latency",cycle);
     if(latency_bound&&!pending.empty()&&cycle-pending.front().issued>=latency_bound)
       require(d.collect_ready,"transaction exceeded scheduling latency bound",cycle);
     if(stalled)require(d.collect_ready&&d.collect_data==previous_data&&d.collect_flags==previous_flags,"response changed under stall",cycle);
     if(d.collect_ready&&d.collect_enable){
       require(!pending.empty(),"spurious/reset-stale response",cycle);Row r=pending.front();pending.pop_front();
       if(d.collect_data!=r.data||d.collect_flags!=r.flags){
         std::fprintf(stderr,"a=%08x b=%08x c=%08x rm=%u expected=%08x/%02x got=%08x/%02x\n",r.a,r.b,r.c,r.rm,r.data,r.flags,d.collect_data,d.collect_flags);
         require(false,"arithmetic/flags/order mismatch",cycle);
       }
       ++consumed;
     }
     if(d.issue_enable&&d.issue_ready){rows[next].issued=cycle;rows[next].observed=false;pending.push_back(rows[next++]);++accepted;}
     if(pending.size()>peak)peak=pending.size();
     require(pending.size()<=capacity,"capacity reservation violated",cycle);
     stalled=d.collect_ready&&!d.collect_enable;previous_data=d.collect_data;previous_flags=d.collect_flags;
   }
   edge();
 }
 d.issue_enable=0;d.collect_enable=1;for(int i=0;i<200;++i){d.eval();require(!d.collect_ready,"duplicate response",cycle);edge();}
 require(consumed==rows.size()&&consumed+discarded==accepted,"loss/accounting mismatch",cycle);
 if(variable){require(min_ready==2,"special-case result padded instead of early completion",cycle);
   std::printf("ITERATIVE ready_min=%llu ready_max=%llu bound=%llu\n",(unsigned long long)min_ready,(unsigned long long)max_ready,(unsigned long long)latency_bound);}
 std::printf("PROTOCOL peak_outstanding=%zu capacity=%u\n",peak,capacity);
 std::printf("PASS IP accepted=%zu consumed=%zu reset-discarded=%zu cycles=%llu\n",accepted,consumed,discarded,(unsigned long long)cycle);
}
