from rlpyt.utils.logging import logger
from torch.utils.tensorboard.writer import SummaryWriter


def video_summary(tag, video, step=None, fps=20):
    writer: SummaryWriter = logger.get_tf_summary_writer()
    writer.add_video(tag=tag, vid_tensor=video, global_step=step, fps=fps)
