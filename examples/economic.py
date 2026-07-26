"""Decompose cost efficiency into technical and allocative components."""

import frontierguard as fg

data = fg.DEAData(
    inputs=[[3.0, 2.0], [1.0, 3.0], [4.0, 6.0]],
    outputs=[[3.0], [5.0], [6.0]],
    dmu_names=("A", "B", "C"),
    input_names=("labor", "capital"),
    output_names=("service",),
)

result = fg.CostEfficiencyDEA(
    input_prices=[4.0, 2.0],
    rts="crs",
).fit(data)

print(result.to_frame())
print("Price-optimal target for A:", result.targets("A"))
