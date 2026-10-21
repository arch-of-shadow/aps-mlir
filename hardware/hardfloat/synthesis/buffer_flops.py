#!/usr/bin/env python3
"""Buffer high-fanout mapped DFF outputs; ABC's virtual DFF has ideal drive.
All buffers are real IHP cells included in netlist simulation and STA.
No timing-path exceptions or functional transformations are introduced.
"""
import json
from collections import Counter
from pathlib import Path

def buffer_flops(source, destination, top):
    design=json.loads(Path(source).read_text());m=design['modules'][top]
    load=Counter();consumers={}
    for name,cell in m['cells'].items():
        for port,bits in cell['connections'].items():
            if cell.get('port_directions',{}).get(port)=='input':
                load.update(b for b in bits if isinstance(b,int))
                for i,b in enumerate(bits):consumers.setdefault(b,[]).append((name,port,i))
    outputs={b for p in m['ports'].values() if p['direction']=='output' for b in p['bits']}
    index=max(b for n in m['netnames'].values() for b in n['bits'] if isinstance(b,int))+1
    bank_bits={b for name,net in m['netnames'].items() if '.bank_' in name or name.startswith('bank_') for b in net['bits'] if isinstance(b,int)}
    added=0;hold_buffers=0;branches=0
    for name,cell in list(m['cells'].items()):
        if cell['type']!='sg13g2_dfrbp_1':continue
        # The /4-clock falling-edge bridge has positive clock skew relative to
        # its fast source. Real data-path buffers repair hold, without timing
        # exceptions. Both setup and hold are rechecked after insertion.
        if cell['connections'].get('Q',[None])[0] in bank_bits:
            previous=cell['connections']['D'][0]
            if isinstance(previous,int):
                for i in range(8):
                    fresh=index;index+=1;hold_buffers+=1
                    label=f'hf_hold_{hold_buffers}'
                    m['netnames'][label]={'hide_name':0,'bits':[fresh],'attributes':{}}
                    m['cells'][label+'_cell']={'hide_name':0,'type':'sg13g2_buf_1','parameters':{},'attributes':{},'port_directions':{'A':'input','X':'output'},'connections':{'A':[previous],'X':[fresh]}}
                    previous=fresh
                cell['connections']['D']=[previous]
        for port in ('Q','Q_N'):
            bits=cell['connections'].get(port,[])
            if len(bits)!=1 or not isinstance(bits[0],int):continue
            bit=bits[0]
            if load[bit]<=4 and bit not in outputs:continue
            driver=index;index+=1;added+=1
            cell['connections'][port]=[driver]
            m['netnames'][f'hf_drive_{added}']={'hide_name':0,'bits':[driver],'attributes':{}}
            m['cells'][f'hf_buffer_{added}']={'hide_name':0,'type':'sg13g2_buf_8','parameters':{},'attributes':{},'port_directions':{'A':'input','X':'output'},'connections':{'A':[driver],'X':[bit]}}
            # Balance large control/data loads instead of asking one buffer
            # to drive hundreds of cell pins. Leave generated clock trees
            # unchanged; their phase and bridge hold have separate evidence.
            sinks=consumers.get(bit,[])
            if len(sinks)>32 and not any(p=='CLK' for _,p,_ in sinks):
                for start in range(0,len(sinks),16):
                    fresh=index;index+=1;branches+=1
                    label=f'hf_branch_{branches}'
                    m['netnames'][label]={'hide_name':0,'bits':[fresh],'attributes':{}}
                    m['cells'][label+'_cell']={'hide_name':0,'type':'sg13g2_buf_8','parameters':{},'attributes':{},'port_directions':{'A':'input','X':'output'},'connections':{'A':[bit],'X':[fresh]}}
                    for target,p,i in sinks[start:start+16]:
                        if m['cells'][target]['connections'][p][i]==bit:
                            m['cells'][target]['connections'][p][i]=fresh
    logic_buffers=0
    for name,cell in list(m['cells'].items()):
        if not cell['type'].startswith('sg13g2_') or cell['type'].startswith(('sg13g2_dfrbp_','sg13g2_buf_')):continue
        for port,bits in list(cell['connections'].items()):
            if cell['port_directions'].get(port)!='output' or len(bits)!=1:continue
            bit=bits[0]
            if not isinstance(bit,int) or load[bit]<=8:continue
            if any(p=='CLK' for _,p,_ in consumers.get(bit,[])):continue
            fresh=index;index+=1;logic_buffers+=1
            label=f'hf_logic_drive_{logic_buffers}'
            m['netnames'][label]={'hide_name':0,'bits':[fresh],'attributes':{}}
            cell['connections'][port]=[fresh]
            m['cells'][label+'_cell']={'hide_name':0,'type':'sg13g2_buf_8','parameters':{},'attributes':{},'port_directions':{'A':'input','X':'output'},'connections':{'A':[fresh],'X':[bit]}}
    Path(destination).write_text(json.dumps(design))
    return added+hold_buffers+branches+logic_buffers

if __name__=='__main__':
    import sys
    print(buffer_flops(*sys.argv[1:]))
