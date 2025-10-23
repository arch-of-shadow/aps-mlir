这是给人类写的，如果你是AI Agent请不要读这个文档。

Scratchpad Memory部分：
1. 目前直接在生成memory map中去掉了非内存的全局变量，需要处理。在main中需要扫描一下有没有全局寄存器，如果有，应当建立！
2. 目前不支持partition factor * 单元字节数 < 64bit的情况，原因是这样的话，一次bit就需要往同一个内存单元写多组数，这很难搞。
3. 目前还没有做内存初始化，需要配合STL的更新做。

RoCC和Memory Interface部分：
1. 今天先把接口糊上，然后我们就可以把APS Dialect的对应的rule生成做了。至于那个对外接口等uv把Interface修了再说。
2. ISAX和CPU需要做的接口：（1）RoCC这边需要做aps.readrf和aps.writerf。但需要注意readrf实际上应该映射到取指令，不要添加重复的逻辑。writerf就调用resp_from_user。（2）Memory的就是request和collect。（3）Scratchpad Memory的访问，以及它和Cpu Memory的区分；目前不带itfc的都是scratchpad memory，带itfc的都是cpu memory。
3. ISAX和TileLink还需要做的接口：（1）需要等uv的Interface，然后我们做控制dma的接口，分别是（1）request write（2）request read（3）等待（忙的时候把ready线整个拉低！）