import torch


class TrackingTermination:
    def errors(self, state, frame):
        height = (state.aux[..., 2] - self.motion.root_p[frame, 2]).abs()
        q = state.aux[..., 3:7]
        ref_q = self.motion.root_quat[frame]
        # World gravity's body-frame Z for a unit quaternion; no CPU tensor copy.
        actual = 2 * (q[..., 1].square() + q[..., 2].square()) - 1
        ref = 2 * (ref_q[..., 1].square() + ref_q[..., 2].square()) - 1
        return height, (actual - ref).abs()

    def terminated(self, state, frame):
        height, tilt = self.errors(state, frame)
        return (height > self.cfg.height_threshold) | (tilt > self.cfg.tilt_threshold)

    def soft_termination(self, state, frame):
        height, tilt = self.errors(state, frame)
        return 1 - (
            1 - torch.sigmoid((height - self.cfg.height_threshold) / self.cfg.height_softness)
        ) * (1 - torch.sigmoid((tilt - self.cfg.tilt_threshold) / self.cfg.tilt_softness))
