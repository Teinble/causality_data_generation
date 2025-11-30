# Causal Pool Question Generation

## Question Types

### Descriptive

Descriptive questions are questions that describe events happened in the scenario.

Example 1: What happened in this video?

- A: The white ball hits the orange ball.
- B: ...

Example 2: Which balls remained stationary throughout the entire video?

Temporal questions like "Which of the following happened first?" are also descriptive.

Basically asking questions based on the current causal graph.

### Predictive

Given a partial scenario (like the first 20% of the whole video), predictive questions ask about what will happen.

### Explanatory

Asks about whehther an object (like a ball) is responsible for another event.

### Counterfactual

Given a scenario, a counterfactual question asks for what would happen after we perform a do(...), assuming anything else remains unchanged.

Counterfactual types:

- initial ball positions for both cue ball and other balls
- initial ball velocities for both cue ball and other balls
- number of balls
- properties of the system (physical constants)
- an external change DURING the scenario (like teleporting the ball in the middle of a scenario)
  - Example: If Ball 2 had been removed right before Ball 1 hit it, Ball 1 would have been pocketed instead
  - Example: If Ball 2 had started moving 0.5 seconds later, it would not collide with Ball 1
