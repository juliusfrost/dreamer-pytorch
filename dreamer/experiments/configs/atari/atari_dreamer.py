config = dict(
    agent=dict(),
    algo=dict(),
    env=dict(
        game="pong",
        num_img_obs=1,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e6,
        # log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=20,
        batch_B=32,
        max_decorrelation_steps=1000,
    ),
)

configs = dict(
    default=config
)
