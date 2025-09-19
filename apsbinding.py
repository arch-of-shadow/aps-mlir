# apsbinding.py (not working yet)


"""
NOT WORKING YET
"""
import aps
from aps.ir import Context, InsertionPoint, IntegerType, Location, Module
from aps.dialects import hw, comb

with Context() as ctx, Location.unknown():
  aps.register_dialects(ctx)
  i42 = IntegerType.get_signless(42)
  m = Module.create()
  with InsertionPoint(m.body):

    def magic(module):
      xor = comb.XorOp.create(module.a, module.b)
      return {"c": xor}

    hw.HWModuleOp(name="magic",
                  input_ports=[("a", i42), ("b", i42)],
                  output_ports=[("c", i42)],
                  body_builder=magic)
  print(m)