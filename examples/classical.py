"""Fit a classical radial DEA model and inspect its evidence."""

import frontierguard as fg

data = fg.load_five_dmu()
result = fg.RadialDEA(
    orientation="input",
    rts="vrs",
    slacks="normalized",
).fit(data)

print(result.to_frame())
print("Peers for A:", result.peers("A"))
print("Targets for A:", result.targets("A"))
print("Maximum audit residual:", float(result.audit_residual.max()))
