from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
from rlpyt.utils.collections import namedarraytuple

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])


def initialize_replay_buffer(self, examples, batch_spec, async_=False):
    """Initializes a sequence replay buffer with single frame observations"""
    example_to_buffer = SamplesToBuffer(
        observation=examples["observation"],
        action=examples["action"],
        reward=examples["reward"],
        done=examples["done"],
    )
    replay_kwargs = dict(
        example=example_to_buffer,
        size=self.replay_size,
        B=batch_spec.B,
        rnn_state_interval=0,  # do not save rnn state
        discount=self.discount,
        n_step_return=self.n_step_return,
    )
    replay_buffer = UniformSequenceReplayBuffer(**replay_kwargs)
    return replay_buffer


def samples_to_buffer(samples):
    """Defines how to add data from sampler into the replay buffer. Called
    in optimize_agent() if samples are provided to that method.  In
    asynchronous mode, will be called in the memory_copier process."""
    return SamplesToBuffer(
        observation=samples.env.observation,
        action=samples.agent.action,
        reward=samples.env.reward,
        done=samples.env.done,
    )
