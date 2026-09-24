# Clinical AI Decision Support Guidance

## Role of the AI system

This AI system is intended to provide decision-support information.
The machine learning prediction is not itself a medical diagnosis.

The prediction model produces a probability and a classification based
on the input features provided to the model.

## Prediction and explanation

The prediction component and the explanation component have separate
responsibilities.

The prediction model is responsible for producing the model prediction.

The retrieval component provides supporting evidence from the available
knowledge sources.

The language model may use retrieved evidence to explain the model output,
but it must not modify the underlying prediction.

## Evidence grounding

An explanation should be grounded in retrieved evidence whenever evidence
is required by the workflow.

If relevant evidence cannot be retrieved, the system should not present
an unsupported evidence-grounded explanation.

## Human oversight

The output of the system should be interpreted by an appropriately
qualified human.

The system should not present a machine learning prediction as a confirmed
medical diagnosis.

## Traceability

The system should preserve information about the prediction model,
retrieved evidence, and generated explanation so that the workflow can
be inspected and evaluated.