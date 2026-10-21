// C acceptance program -> integer RoCC -> complete RTL / mapped gate netlist.
#include "Vmain.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <deque>
#include <fstream>
#include <random>
#include <vector>
static Vmain dut;
static std::mt19937 rng(0xBF1632);
static uint64_t cycles=0,calls=0,reset_discards=0;
static void require(bool ok,const char *why) {
 if(!ok){std::fprintf(stderr,"FAIL cycle=%llu: %s\n",(unsigned long long)cycles,why);std::exit(1);}
}
static void edge(){dut.clk=0;dut.eval();dut.clk=1;dut.eval();dut.clk=0;dut.eval();++cycles;}
static void reset(){dut.rocc_cmd_enable=0;dut.rocc_resp_rocc_resp_to_bus_ready=0;dut.rst=1;edge();edge();dut.rst=0;dut.eval();}
extern "C" uint32_t isax_execute(uint32_t a,uint32_t b) {
 ++calls;bool accepted=false,injected=false;uint32_t rd=uint32_t(calls%31)+1;
 dut.rocc_cmd_rocc_cmd_rs1data=a;dut.rocc_cmd_rocc_cmd_rs2data=b;
 for(unsigned t=0;t<20000;++t){
   if(!injected&&calls%257==23&&accepted&&t>=4){reset();accepted=false;injected=true;rd=(rd+13)%31+1;++reset_discards;}
   dut.rocc_cmd_rocc_cmd_rd=rd;
   dut.rocc_cmd_enable=!accepted&&rng()%5!=0;
   dut.rocc_resp_rocc_resp_to_bus_ready=t%199<137&&rng()%3!=0;
   dut.eval();
   if(dut.rocc_resp_rocc_resp_to_bus_enable){
     require(accepted,"response before accepted command / stale post-reset result");
     require(dut.rocc_resp_rocc_resp_to_bus_ready,"response consumed under stall");
     require(dut.rocc_resp_rocc_resp_to_bus_result_rd==rd,"destination/context mismatch");
     auto result=dut.rocc_resp_rocc_resp_to_bus_result_rddata;edge();dut.rocc_cmd_enable=0;return result;
   }
   if(dut.rocc_cmd_enable&&dut.rocc_cmd_ready)accepted=true;
   edge();
 }
 require(false,"transaction timeout");return 0;
}
extern "C" int run_test(const char*);
// Keep presenting independent commands while previous responses are stalled.
// The DUT decides admission; report actual overlap, never infer it from input
// traffic. Unique outstanding destinations expose command-context corruption.
static void burst_test(const char *path) {
 struct Vector { uint32_t a,b,expected; } row;
 std::ifstream file(path);std::vector<Vector> rows;
 while(file>>std::hex>>row.a>>row.b>>row.expected) rows.push_back(row);
 require(file.eof()&&!rows.empty(),"invalid burst vectors");
 reset();size_t next=0,completed=0,peak=0;std::deque<size_t> pending;
 uint64_t start=cycles;
 while(completed<rows.size()) {
   require(cycles-start<rows.size()*20000,"burst timeout");
   bool offer=next<rows.size()&&pending.size()<8;
   dut.rocc_cmd_enable=offer;
   if(offer) {
     dut.rocc_cmd_rocc_cmd_rs1data=rows[next].a;
     dut.rocc_cmd_rocc_cmd_rs2data=rows[next].b;
     dut.rocc_cmd_rocc_cmd_rd=next%31+1;
   }
   dut.rocc_resp_rocc_resp_to_bus_ready=(cycles-start)%257>=160&&rng()%4!=0;
   dut.eval();
   if(dut.rocc_resp_rocc_resp_to_bus_enable) {
     require(!pending.empty(),"burst stale/duplicate response");
     size_t index=pending.front();pending.pop_front();
     require(dut.rocc_resp_rocc_resp_to_bus_ready,"burst response under stall");
     require(dut.rocc_resp_rocc_resp_to_bus_result_rd==index%31+1,"burst destination/context mismatch");
     require(dut.rocc_resp_rocc_resp_to_bus_result_rddata==rows[index].expected,"burst arithmetic/context mismatch");
     ++completed;
   }
   if(offer&&dut.rocc_cmd_ready) pending.push_back(next++);
   peak=std::max(peak,pending.size());edge();
 }
 dut.rocc_cmd_enable=0;
 std::printf("PASS burst requests=%zu peak_outstanding=%zu\n",completed,peak);
}
int main(int argc,char**argv){
 Verilated::commandArgs(argc,argv);require(argc==2,"expected golden vector path");
 dut.rocc_cmd_rocc_cmd_opcode=11;dut.rocc_cmd_rocc_cmd_funct=0;
 dut.rocc_cmd_rocc_cmd_rs1=1;dut.rocc_cmd_rocc_cmd_rs2=2;
 dut.rocc_cmd_rocc_cmd_xs1=1;dut.rocc_cmd_rocc_cmd_xs2=1;dut.rocc_cmd_rocc_cmd_xd=1;
 dut.burst_read_0_enable=dut.burst_read_1_enable=dut.burst_write_enable=0;
 dut.hella_resp_enable=0;dut.hella_cmd_hella_cmd_to_bus_ready=1;reset();
 int status=run_test(argv[1]);
 if(!status) burst_test(argv[1]);
 dut.rocc_cmd_enable=0;dut.rocc_resp_rocc_resp_to_bus_ready=1;
 for(unsigned i=0;i<100;++i){dut.eval();require(!dut.rocc_resp_rocc_resp_to_bus_enable,"duplicate/leaked response");edge();}
 std::printf("protocol cycles=%llu reset-discard/retry=%llu\n",(unsigned long long)cycles,(unsigned long long)reset_discards);
 return status;
}
