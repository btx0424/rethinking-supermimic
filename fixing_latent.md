# Fixing latent imagination

The actor maximizes an H-step return through the latent world model, then bootstraps `V(z_H)`. On the LAFAN1 latent runs that return is a fantasy the critic then memorizes. Imagined survival stays near 1 while `term_frac` stays near 1 in the simulator. `value_mean` and `|pi_loss|` climb together to about 4 and stick, `v_loss` collapses, and the bootstrap term is the largest policy gradient. Real episodes fall immediately, so the buffer never contains a long trajectory that would contradict the imagined one.

The critic is accurate about its target. The target is wrong. Both ensemble heads are trained on the same imagined return, so averaging them still tracks the fantasy. `critic_source=sim` anchors `V` at encoded real states and still leaves the actor free to read `V` at an off-manifold `z_H`.

Two ways to cut that return follow. Both keep the current decode path: per-step reward and height/tilt termination come from the decoded pose, and the learned reward and termination heads stay auxiliary losses.

## 1. Two values, one cut

Train `V_sim` on simulator rewards and hard falls, and `V_imag` on the imagined rollout. Walk the latent trajectory from `h(s)`. When `|V_sim(z) - V_imag(z)|` exceeds a threshold, end the return there and bootstrap from `V_sim` at the last accepted code.

The cut is a constant with respect to the action. A differentiable gate lets the actor search for codes where the two values already agree and are high.

Read the gap as a comparison of two returns. On a code `z = h(s)` taken from the buffer they already differ, because one target dies in the simulator and the other survives in the model. That is the model's optimism, measured on states the policy visits every batch. Off the image of the encoder the imagined head is fitted to the fantasy, and `V_sim` is an extrapolation. The gap is small wherever that extrapolation is also high, which is the standing region the decoder already scores as alive. A `step` can also emit a code no physical state encodes.

A visitation test is a pair of critics trained on the same real return, from different initializations, or a density model on real embeddings `h(s)`. Those agree on the buffer and spread once `z` leaves it. The imagined critic is the signal being rejected, so it cannot be one of the two.

## 2. Dynamics ensemble, sim critic, no value gradient

Train one critic, on simulator returns only (`critic_source=sim`). Fit a small ensemble of dynamics heads to `h(s')` on real transitions. During the actor rollout, stop at the first step where the heads' spread crosses a threshold. That step ends the imagined episode. Add no bootstrap after the cut. Detach the spread so the threshold carries no action gradient.

The actor's gradient is the per-step tracking reward up to the cut. Replay actions keep the heads in agreement. An action the actor invented spreads them. Lowering the spread means taking an action the ensemble can predict, which is a property of the transition and has no return the actor can inflate.

Turn `policy_bootstrap` off. With it on, `alive[-1] · V(z_H)` flows into `z_H` and the actor walks the code toward a high value. The terminal value of `V_sim` then adds nothing to the action gradient, and the sim critic never trains on that `z_H`, so it cannot confirm the fantasy.

## First run

Before building the ensemble, run the latent trainer with `critic_source=sim` and `policy_bootstrap=false`, decode scoring left as it is. If imagined survival stays near 1 while the simulator still falls, the remaining hole is the actor's own actions, and the ensemble cut is what closes it.
