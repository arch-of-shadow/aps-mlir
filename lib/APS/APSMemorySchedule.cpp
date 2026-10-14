#include "APS/APSOps.h"
#include "APS/PassDetail.h"
#include "APS/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include <algorithm>
#include <deque>
#include <limits>
#include <optional>
#include <vector>
#include <unordered_map>

namespace {
using namespace mlir;

struct Model {
  int64_t width, inflight, lead, completion;
  int64_t minBytes, minWriteBytes, maxBytes;
  StringRef hint;
};
struct Group {
  SmallVector<Operation *> ops;
  SmallVector<int64_t> sizes;
  unsigned interface;
  int64_t hierarchy;
  SmallVector<unsigned> predecessors;
};
struct Profile {
  SmallVector<unsigned> members;
};
struct SegmentKind {
  int64_t bytes;
  SmallVector<unsigned> positions;
};
struct Timing {
  int64_t issue = -1;
  std::deque<int64_t> completions;
};
struct State {
  std::vector<int64_t> remaining;
  // A selected transfer remains locked until all of its segments are issued.
  int activeGroup = -1;
  std::vector<int64_t> segments;
  std::vector<int64_t> dependencyCompletions;
  std::vector<Timing> timing;
  int64_t lastIssue = -1;
  int64_t lastHierarchy = -1;
};
struct VectorHash {
  size_t operator()(const std::vector<int64_t> &key) const {
    size_t hash = 0;
    for (int64_t value : key)
      hash = hash * 31 + std::hash<int64_t>{}(value);
    return hash;
  }
};

std::optional<int64_t> constant(Value value) {
  if (auto op = value.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue()))
      return attr.getInt();
  return std::nullopt;
}

Value address(Operation *op) {
  if (auto load = dyn_cast<aps::LoadBy>(op)) return load.getCpuAddr();
  if (auto store = dyn_cast<aps::StoreBy>(op)) return store.getCpuAddr();
  return cast<aps::CopyBy>(op).getCpuAddr();
}

bool reads(Operation *op) {
  if (auto copy = dyn_cast<aps::CopyBy>(op))
    return copy.getDirection() == aps::CopyDirection::In;
  return isa<aps::LoadBy>(op);
}

bool sameMemory(Value a, Value b) {
  if (a == b) return true;
  auto ga = a.getDefiningOp<memref::GetGlobalOp>();
  auto gb = b.getDefiningOp<memref::GetGlobalOp>();
  if (ga && gb) return ga.getName() == gb.getName();
  // Unknown views/arguments may alias; distinct globals are the safe case.
  return !ga || !gb;
}

bool needsSourceOrder(const Group &a, const Group &b, bool isRead) {
  if (!isRead) {
    auto aa = constant(address(a.ops.front()));
    auto ba = constant(address(b.ops.front()));
    int64_t as = 0, bs = 0;
    for (auto size : a.sizes) as += size;
    for (auto size : b.sizes) bs += size;
    if (!aa || !ba || *aa < 0 || *ba < 0 ||
        (*aa <= *ba ? uint64_t(*ba - *aa) < uint64_t(as)
                    : uint64_t(*aa - *ba) < uint64_t(bs)))
      return true;
  }
  // The existing scalar response adapter consumes responses in source order.
  if (isa<aps::LoadBy>(a.ops.front()) && isa<aps::LoadBy>(b.ops.front()))
    return true;
  if (isRead) {
    auto ac = dyn_cast<aps::CopyBy>(a.ops.front());
    auto bc = dyn_cast<aps::CopyBy>(b.ops.front());
    if (ac && bc)
      for (Value am : ac.getMemrefs())
        for (Value bm : bc.getMemrefs())
          if (sameMemory(am, bm)) return true;
  }
  return false;
}

/// Reads alias through their scratchpad destinations, not through main memory.
/// Index known globals rather than comparing every pair of independent copies.
/// Unknown views/arguments are conservative fences for all scratchpad writes.
void buildReadDependencies(SmallVectorImpl<Group> &groups) {
  llvm::StringMap<unsigned> lastWriter;
  std::optional<unsigned> unknownWriter, scalarRead;
  for (auto [g, group] : llvm::enumerate(groups)) {
    if (isa<aps::LoadBy>(group.ops.front())) {
      if (scalarRead) group.predecessors.push_back(*scalarRead);
      scalarRead = g;
      continue;
    }
    auto copy = cast<aps::CopyBy>(group.ops.front());
    bool unknown = llvm::any_of(copy.getMemrefs(), [](Value memref) {
      return !memref.getDefiningOp<memref::GetGlobalOp>();
    });
    if (unknownWriter) group.predecessors.push_back(*unknownWriter);
    if (unknown) {
      for (const auto &entry : lastWriter)
        group.predecessors.push_back(entry.second);
      lastWriter.clear();
      unknownWriter = g;
    } else {
      for (Value memref : copy.getMemrefs()) {
        auto name = memref.getDefiningOp<memref::GetGlobalOp>().getName();
        if (auto previous = lastWriter.find(name);
            previous != lastWriter.end() && previous->second != g)
          group.predecessors.push_back(previous->second);
        lastWriter[name] = g;
      }
    }
    llvm::sort(group.predecessors);
    group.predecessors.erase(std::unique(group.predecessors.begin(),
                                        group.predecessors.end()),
                              group.predecessors.end());
  }
}

/// Memoized suffix search. Timing is normalized to the latest issue, retaining
/// only the last I completions per interface. Times earlier than that window
/// collapse to -1 because they cannot constrain any future issue. Equivalent
/// independent groups use counts rather than permutation/subset identities.
class OrderSearch {
public:
  OrderSearch(ArrayRef<Group> groups, ArrayRef<Model> models, bool isRead)
      : groups(groups), models(models), isRead(isRead) {
    SmallVector<bool> constrained(groups.size(), false);
    for (auto [g, group] : llvm::enumerate(groups))
      for (unsigned p : group.predecessors)
        constrained[g] = constrained[p] = true;
    groupProfile.resize(groups.size());
    groupPosition.resize(groups.size());
    completionSlot.assign(groups.size(), -1);
    for (const auto &group : groups)
      for (unsigned pred : group.predecessors)
        if (completionSlot[pred] == -1) completionSlot[pred] = completionSlots++;
    for (auto [g, group] : llvm::enumerate(groups)) {
      SmallVector<SegmentKind> kinds;
      for (auto [position, bytes] : llvm::enumerate(group.sizes)) {
        auto kind = llvm::find_if(kinds, [&](const auto &kind) {
          return kind.bytes == bytes;
        });
        if (kind == kinds.end()) {
          kinds.push_back({bytes, {}});
          kind = std::prev(kinds.end());
        }
        kind->positions.push_back(position);
      }
      segmentKinds.push_back(std::move(kinds));
      unsigned profile = profiles.size();
      if (!constrained[g])
        for (unsigned p = 0; p < profiles.size(); ++p) {
          unsigned other = profiles[p].members.front();
          if (!constrained[other] && group.interface == groups[other].interface &&
              group.hierarchy == groups[other].hierarchy &&
              group.sizes == groups[other].sizes) {
            profile = p;
            break;
          }
        }
      if (profile == profiles.size()) profiles.push_back({});
      groupProfile[g] = profile;
      groupPosition[g] = profiles[profile].members.size();
      profiles[profile].members.push_back(g);
    }
  }

  State initial() const {
    State state;
    state.dependencyCompletions.assign(completionSlots, -1);
    SmallVector<int64_t> count(models.size(), 0);
    for (const auto &group : groups)
      count[group.interface] += group.ops.size();
    for (const auto &profile : profiles)
      state.remaining.push_back(profile.members.size());
    for (auto [k, model] : llvm::enumerate(models)) {
      Timing timing;
      timing.completions.assign(std::min(model.inflight, count[k]), -1);
      state.timing.push_back(std::move(timing));
    }
    return state;
  }

  struct Event { Operation *op; unsigned interface; int64_t issue, complete; };
  void begin(State &state, unsigned g) const {
    state.activeGroup = g;
    state.segments.clear();
    for (const auto &kind : segmentKinds[g])
      state.segments.push_back(kind.positions.size());
  }

  int64_t advance(State &state, unsigned kindIndex,
                  SmallVectorImpl<Event> *events = nullptr,
                  int64_t origin = 0) const {
    unsigned g = state.activeGroup;
    const auto &group = groups[g];
    const auto &kind = segmentKinds[g][kindIndex];
    unsigned segment = kind.positions[kind.positions.size() - state.segments[kindIndex]];
    const auto &model = models[group.interface];
    auto &timing = state.timing[group.interface];
    int64_t readyAt = 0;
    for (unsigned pred : group.predecessors)
      readyAt = std::max(readyAt, state.dependencyCompletions[completionSlot[pred]] + 1);
    {
      int64_t bytes = kind.bytes;
      int64_t issue = 1 + std::max(timing.issue, timing.completions.front());
      // Different interfaces can issue in the same cycle. A hierarchy boundary
      // requires the next group to start strictly after the previous issue.
      int64_t hierarchyGap = state.lastHierarchy != -1 &&
                             state.lastHierarchy != group.hierarchy ? 1 : 0;
      issue = std::max({issue, state.lastIssue + hierarchyGap, readyAt});
      int64_t beats = (bytes + model.width - 1) / model.width;
      int64_t complete = isRead
          ? beats + std::max(timing.completions.back(), issue + model.lead - 1)
          : beats + model.completion + std::max(timing.completions.back(), issue - 1);
      timing.issue = state.lastIssue = issue;
      timing.completions.pop_front();
      timing.completions.push_back(complete);
      if (events)
        events->push_back({group.ops[segment], group.interface,
                           origin + issue, origin + complete});
    }
    --state.segments[kindIndex];
    if (llvm::all_of(state.segments, [](int64_t count) { return count == 0; })) {
      --state.remaining[groupProfile[g]];
      if (completionSlot[g] >= 0)
        state.dependencyCompletions[completionSlot[g]] = timing.completions.back();
      state.activeGroup = -1;
      state.segments.clear();
    }
    state.lastHierarchy = group.hierarchy;
    int64_t shift = state.lastIssue;
    for (auto &t : state.timing) {
      t.issue = std::max<int64_t>(-1, t.issue - shift);
      for (auto &b : t.completions) b = std::max<int64_t>(-1, b - shift);
    }
    for (auto &completion : state.dependencyCompletions)
      completion = std::max<int64_t>(-1, completion - shift);
    state.lastIssue = 0;
    return shift;
  }

  int64_t solve(const State &state) {
    auto key = makeKey(state);
    if (auto it = memo.find(key); it != memo.end()) {
      ++hits;
      return it->second.cost;
    }
    // Branch on distinct sizes, not individual equal-sized segment identities.
    // The lock excludes other transfers until this transfer is fully issued.
    if (state.activeGroup >= 0) {
      Result best{std::numeric_limits<int64_t>::max(), -1, -1};
      for (unsigned kind = 0; kind < state.segments.size(); ++kind) {
        if (!state.segments[kind]) continue;
        State next = state;
        int64_t shift = advance(next, kind);
        int64_t cost = shift + solve(next);
        if (cost < best.cost) best = {cost, state.activeGroup, int(kind)};
      }
      memo.emplace(std::move(key), best);
      return best.cost;
    }
    SmallVector<unsigned> ready;
    int64_t rank = isRead ? std::numeric_limits<int64_t>::max()
                          : std::numeric_limits<int64_t>::min();
    for (unsigned p = 0; p < profiles.size(); ++p) {
      if (!state.remaining[p]) continue;
      unsigned g = profiles[p].members[profiles[p].members.size() - state.remaining[p]];
      bool available = llvm::all_of(groups[g].predecessors, [&](unsigned pred) {
        unsigned pp = groupProfile[pred];
        return profiles[pp].members.size() - state.remaining[pp] > groupPosition[pred];
      });
      if (!available) continue;
      int64_t level = groups[g].hierarchy;
      bool better = isRead ? level < rank : level > rank;
      if (better) { ready.clear(); rank = level; }
      if (level == rank) ready.push_back(g);
    }
    Result best{std::numeric_limits<int64_t>::max(), -1, -1};
    if (ready.empty()) {
      best.cost = -1;
      for (const auto &timing : state.timing)
        if (!timing.completions.empty())
          best.cost = std::max(best.cost, timing.completions.back());
    } else {
      for (unsigned g : ready) {
        State next = state;
        begin(next, g);
        int64_t cost = solve(next);
        if (cost < best.cost) best = {cost, int(g), -1};
      }
    }
    memo.emplace(std::move(key), best);
    return best.cost;
  }

  SmallVector<Event> schedule() {
    State state = initial();
    solve(state);
    SmallVector<Event> events;
    int64_t origin = 0;
    while (true) {
      auto decision = memo.at(makeKey(state));
      if (decision.group < 0) break;
      if (decision.segment < 0)
        begin(state, decision.group);
      else
        origin += advance(state, decision.segment, &events, origin);
    }
    return events;
  }
  size_t states() const { return memo.size(); }
  size_t memoHits() const { return hits; }

private:
  struct Result { int64_t cost; int group; int segment; };
  std::vector<int64_t> makeKey(const State &state) const {
    auto key = state.remaining;
    key.insert(key.end(), state.dependencyCompletions.begin(), state.dependencyCompletions.end());
    key.push_back(state.activeGroup);
    key.insert(key.end(), state.segments.begin(), state.segments.end());
    key.push_back(state.lastIssue);
    key.push_back(state.lastHierarchy);
    for (const auto &timing : state.timing) {
      key.push_back(timing.issue);
      key.insert(key.end(), timing.completions.begin(), timing.completions.end());
    }
    return key;
  }
  ArrayRef<Group> groups;
  ArrayRef<Model> models;
  bool isRead;
  SmallVector<Profile> profiles;
  SmallVector<SmallVector<SegmentKind>> segmentKinds;
  SmallVector<unsigned> groupProfile, groupPosition;
  SmallVector<int> completionSlot;
  unsigned completionSlots = 0;
  std::unordered_map<std::vector<int64_t>, Result, VectorHash> memo;
  size_t hits = 0;
};

struct APSMemorySchedulePass : public APSMemoryScheduleBase<APSMemorySchedulePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Model> models;
    llvm::StringMap<unsigned> interfaces;
    for (auto op : module.getOps<aps::MemItfc>()) {
      auto positive = [&](StringRef attr) -> int64_t {
        auto value = op->getAttrOfType<IntegerAttr>(attr);
        return value ? value.getInt() : 0;
      };
      Model model{positive("width"), positive("inflight"), positive("read_latency"),
                  positive("write_completion"), 0, 0, 0, {}};
      if (!model.completion) model.completion = positive("write_latency");
      if (model.width <= 0 || model.inflight <= 0 || model.lead <= 0 || model.completion <= 0) {
        op.emitError("memory ordering requires positive width, inflight, read_latency and write_completion");
        return signalPassFailure();
      }
      int64_t minimum = op->hasAttr("min_beats") ? positive("min_beats") : 1;
      int64_t maximum = positive("max_beats");
      int64_t writeMinimum = op->hasAttr("min_write_beats") ? positive("min_write_beats") :
          (op.getSymName() == "busitfc" ? std::max<int64_t>(2, minimum) : minimum);
      if (minimum <= 0 || writeMinimum <= 0 || maximum < minimum || maximum < writeMinimum ||
          (minimum & (minimum - 1)) || (writeMinimum & (writeMinimum - 1)) ||
          (model.width & (model.width - 1)) ||
          maximum > std::numeric_limits<int64_t>::max() / model.width) {
        op.emitError("invalid whole-beat transaction bounds for memory ordering");
        return signalPassFailure();
      }
      model.minBytes = minimum * model.width;
      model.minWriteBytes = std::max(minimum, writeMinimum) * model.width;
      if (op.getSymName() == "busitfc" && (model.width != 8 || model.minWriteBytes < 16)) {
        op.emitError("native DMA requires 8-byte beats and at least two beats per write");
        return signalPassFailure();
      }
      model.maxBytes = maximum * model.width;
      if (auto hint = op->getAttrOfType<StringAttr>("cache_hint")) model.hint = hint.getValue();
      interfaces[op.getSymName()] = models.size();
      models.push_back(model);
    }
    uint64_t nextId = 0;
    auto result = module.walk([&](Operation *parent) -> WalkResult {
      for (Region &region : parent->getRegions())
        for (Block &block : region) {
          SmallVector<Group> groups;
          llvm::DenseSet<Operation *> members;
          bool isRead = true;
          auto flush = [&]() {
            if (groups.empty()) return;
            if (isRead) {
              buildReadDependencies(groups);
            } else {
              for (unsigned g = 0; g < groups.size(); ++g)
                for (unsigned p = 0; p < g; ++p)
                  if (needsSourceOrder(groups[p], groups[g], false))
                    groups[g].predecessors.push_back(p);
            }
            OrderSearch search(groups, models, isRead);
            auto events = search.schedule();
            int64_t latency = 0;
            for (auto event : events) latency = std::max(latency, event.complete);
            OpBuilder builder(module.getContext());
            SmallVector<SmallVector<OrderSearch::Event>> history(models.size());
            llvm::DenseMap<Operation *, unsigned> opGroup;
            SmallVector<std::optional<OrderSearch::Event>> groupEnd(groups.size());
            SmallVector<unsigned> issued(groups.size(), 0);
            for (auto [g, group] : llvm::enumerate(groups))
              for (Operation *op : group.ops) opGroup[op] = g;
            std::optional<OrderSearch::Event> previous;
            for (auto event : events) {
              Operation *op = event.op;
              op->setAttr("transaction_id", builder.getI64IntegerAttr(nextId++));
              op->setAttr("memory_order_solver", builder.getStringAttr("memoized-relative-window"));
              op->setAttr("memory_issue_cycle", builder.getI64IntegerAttr(event.issue));
              op->setAttr("memory_completion_cycle", builder.getI64IntegerAttr(event.complete));
              op->setAttr("memory_latency", builder.getI64IntegerAttr(event.complete - event.issue));
              op->setAttr("memory_schedule_latency", builder.getI64IntegerAttr(latency));
              op->setAttr("memory_search_states", builder.getI64IntegerAttr(search.states()));
              op->setAttr("memory_search_hits", builder.getI64IntegerAttr(search.memoHits()));
              SmallVector<int64_t> after, gaps;
              auto depend = [&](const OrderSearch::Event &source, int64_t gap) {
                after.push_back(source.op->getAttrOfType<IntegerAttr>("transaction_id").getInt());
                gaps.push_back(gap);
              };
              unsigned g = opGroup.lookup(op);
              if (issued[g] == 0)
                for (unsigned pred : groups[g].predecessors) {
                  const auto &source = *groupEnd[pred];
                  depend(source, source.complete - source.issue + 1);
                }
              if (op->hasAttr("transaction_group")) {
                op->setAttr("issue_segment", builder.getI64IntegerAttr(issued[g]));
                if (groupEnd[g])
                  op->setAttr("after_segment", groupEnd[g]->op->getAttr("segment"));
                else
                  op->removeAttr("after_segment");
              }
              if (previous) depend(*previous, event.issue - previous->issue);
              auto &stream = history[event.interface];
              int64_t limit = models[event.interface].inflight;
              if (int64_t(stream.size()) >= limit) {
                auto &retired = stream[stream.size() - limit];
                depend(retired, retired.complete - retired.issue + 1);
              }
              if (!stream.empty()) {
                auto &last = stream.back();
                int64_t beats = (op->getAttrOfType<IntegerAttr>("transfer_bytes").getInt() +
                                  models[event.interface].width - 1) / models[event.interface].width;
                int64_t serialize = last.complete - last.issue + beats +
                                    (isRead ? 0 : models[event.interface].completion) -
                                    (event.complete - event.issue);
                depend(last, serialize);
              }
              op->setAttr("after", builder.getDenseI64ArrayAttr(after));
              op->setAttr("after_gap", builder.getDenseI64ArrayAttr(gaps));
              stream.push_back(event);
              previous = event;
              ++issued[g];
              groupEnd[g] = event;
            }
            groups.clear();
            members.clear();
          };
          for (Operation &op : block) {
            if (!isa<aps::LoadBy, aps::StoreBy, aps::CopyBy>(op)) {
              bool usesResult = llvm::any_of(op.getOperands(), [&](Value value) {
                return members.contains(value.getDefiningOp());
              });
              if (usesResult || !isMemoryEffectFree(&op) || op.getNumRegions()) flush();
              continue;
            }
            auto itfc = op.getAttrOfType<FlatSymbolRefAttr>("itfc");
            auto model = itfc ? interfaces.find(itfc.getValue()) : interfaces.end();
            if (model == interfaces.end()) {
              op.emitError("unresolved modeled memory interface");
              return WalkResult::interrupt();
            }
            auto bytes = op.getAttrOfType<IntegerAttr>("transfer_bytes");
            int64_t minimum = reads(&op) ? models[model->second].minBytes : models[model->second].minWriteBytes;
            if (!bytes || bytes.getInt() < minimum ||
                bytes.getInt() > models[model->second].maxBytes ||
                bytes.getInt() % models[model->second].width ||
                (bytes.getInt() & (bytes.getInt() - 1))) {
              op.emitError("architectural memory op must already be a canonical whole-beat transaction within min_beats/max_beats; run functional-to-architectural conversion first");
              return WalkResult::interrupt();
            }
            if (auto addr = constant(address(&op));
                addr && uint64_t(*addr) % uint64_t(bytes.getInt()) != 0) {
              op.emitError("architectural transaction violates natural address alignment");
              return WalkResult::interrupt();
            }
            if (!groups.empty() && reads(&op) != isRead) flush();
            isRead = reads(&op);
            StringRef hint = models[model->second].hint;
            if (auto h = op.getAttrOfType<StringAttr>("cache_hint")) hint = h.getValue();
            int64_t hierarchy = hint == "cold" ? 1 : 0;
            if (auto level = op.getAttrOfType<IntegerAttr>("cache_level")) {
              if (level.getInt() < 0) {
                op.emitError("cache_level must be non-negative");
                return WalkResult::interrupt();
              }
              hierarchy = level.getInt();
            }
            auto group = op.getAttrOfType<IntegerAttr>("transaction_group");
            bool continuation = group && !groups.empty() &&
                groups.back().ops.back()->getAttr("transaction_group") == group &&
                groups.back().interface == model->second &&
                groups.back().hierarchy == hierarchy;
            if (!continuation) groups.push_back({{}, {}, model->second, hierarchy, {}});
            groups.back().ops.push_back(&op);
            groups.back().sizes.push_back(bytes.getInt());
            members.insert(&op);
          }
          flush();
        }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::createAPSMemorySchedulePass() {
  return std::make_unique<APSMemorySchedulePass>();
}
