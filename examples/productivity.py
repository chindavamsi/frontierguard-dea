"""Measure adjacent-period productivity change for an aligned panel."""

import frontierguard as fg

names = ("A", "B", "C")
period_1 = fg.DEAData(
    inputs=[[2.0], [3.0], [5.0]],
    outputs=[[1.0], [4.0], [6.0]],
    dmu_names=names,
    input_names=("resources",),
    output_names=("outcomes",),
)
period_2 = fg.DEAData(
    inputs=[[1.0], [2.0], [4.0]],
    outputs=[[1.0], [4.0], [6.0]],
    dmu_names=names,
    input_names=("resources",),
    output_names=("outcomes",),
)

result = fg.malmquist(
    period_1,
    period_2,
    orientation="input",
    rts="crs",
    period_labels=("2025", "2026"),
)

print(result.to_frame())
print("Maximum decomposition residual:", result.decomposition_residual.max())
