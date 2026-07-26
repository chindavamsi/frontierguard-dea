"""Analyze a two-stage production process with an explicit intermediate link."""

import frontierguard as fg

network = fg.SerialNetworkData(
    inputs=[[1.0], [2.0], [2.0]],
    links=([[1.0], [1.0], [2.0]],),
    outputs=[[1.0], [1.0], [1.5]],
    dmu_names=("A", "B", "C"),
    input_names=("resources",),
    link_names=(("completed_work",),),
    output_names=("outcomes",),
    stage_names=("operations", "delivery"),
)

result = fg.SerialNetworkDEA().fit(network)
print(result.to_frame())
print("Overall/stage closure residuals:", result.closure_error)
