"""HoloAssist-NF — Experiment C of the NF forget/mistake project.

A *sequential* conditional Normalizing Flow over the pace of a procedural task:

    log p( (log-duration, log-gap) of a step | GRU(history), action, task )

SURPRISE = -log p is "how unexpected was the timing of this step", used to
anticipate mistakes and human interventions. This activates the NF strengths
that Experiment A left dormant: a *sequential* exact-likelihood surprise curve,
*generation* of the expected-next-step timing fan, and *anticipatory* lead time
before an intervention. Data = HoloAssist open annotations (no video needed).
"""
