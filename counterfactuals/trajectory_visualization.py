import cv2
import numpy as np
import copy

from counterfactuals.dataset import Trajectory


def add_centered_text(
    img: np.ndarray,
    text: str,
    color: tuple = (255, 255, 255),
    scale: int = 1,
    top_padding: int = 10,
):
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(text, font, scale, thickness)[0]
    textcoords = ((img.shape[1] - textsize[0]) // 2, textsize[1] + top_padding)
    cv2.putText(img, text, textcoords, font, scale, color, thickness, cv2.LINE_AA)


def gray_and_blur(img: np.ndarray):
    """
    Filter applied to an image that is not active
    """
    g_img = np.repeat(np.expand_dims(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 2), 3, 2)
    kernel = np.ones((15, 15), np.float32) / 225
    dst = cv2.filter2D(g_img, -1, kernel)
    return dst


def make_img(
    state,
    text: str,
    text_scale: float = 0.6,
    downscale_factor: int = 1,
    active: bool = True,
):
    """
    :param state: State to render
    :param text: Text to write in center
    :param text_scale: Size of the text, usually 0.6 or 1.2
    :param downscale_factor: Most env renders are too big, so downsize it
    :active 

    """
    frame = state.render(mode="rgb_array")
    new_size = (frame.shape[1] // downscale_factor, frame.shape[0] // downscale_factor)
    frame = cv2.resize(frame, new_size)
    add_centered_text(frame, text, scale=text_scale)
    if not active:
        return gray_and_blur(frame)
    return frame


def compose_frames(
    big_frame: np.ndarray, top_right: np.ndarray, bottom_right: np.ndarray
):
    """
                            |--------------|--------|
                            |              |        |
    Makes frames like this: |              |--------|
                            |              |        |
                            |______________|________|
    """
    stacked_frames = np.append(top_right, bottom_right, 0)
    return np.append(big_frame, stacked_frames, 1)


def side_by_side_visualization(original: Trajectory, counterfactual: Trajectory):
    """
    Creates a visualization to view how the behavior of the counterfactual
    trajectory differs from the original, or pretrained trajectory.
    """
    divergent_step = counterfactual.step[0]
    o_divergent_state = copy.deepcopy(original.state[divergent_step])
    counterfactual_step = 0
    video = []
    for state, step in zip(original.state, original.step):
        # Focus on big frame until the divergent step
        if step <= divergent_step:
            big_frame = make_img(state, "Original Trajectory", text_scale=1.2)
        elif step > divergent_step:
            big_frame = make_img(
                o_divergent_state, "Original Trajectory", text_scale=1.2, active=False
            )

        c_state = counterfactual.state[counterfactual_step]

        diverged = step >= divergent_step
        if step >= divergent_step:
            o_divergent_img = make_img(
                state, "Original Trajectory", downscale_factor=2, active=diverged,
            )
        else:
            o_divergent_img = make_img(
                o_divergent_state,
                "Original Trajectory",
                downscale_factor=2,
                active=diverged,
            )
        c_divergent_img = make_img(
            c_state, "Counterfactual Trajectory", downscale_factor=2, active=diverged
        )
        # Some trajectories had different lengths, so this preserves
        # the counterfactual state if this trajectory ends first
        if counterfactual_step < len(counterfactual.state) - 1:
            if step >= divergent_step:
                counterfactual_step += 1
            c_state.close()
        elif counterfactual_step == len(counterfactual.state) - 1:
            add_centered_text(c_divergent_img, "DONE", top_padding=100)

        # And this checks if the counterfactual trajectory is longer
        if step == original.step[-1]:
            add_centered_text(o_divergent_img, "DONE", top_padding=100)
            while counterfactual_step < len(counterfactual.step) - 1:
                c_state = counterfactual.state[counterfactual_step]
                c_divergent_img = make_img(
                    c_state,
                    "Counterfactual Trajectory",
                    downscale_factor=2,
                    active=diverged,
                )
                video.append(
                    compose_frames(big_frame, o_divergent_img, c_divergent_img)
                )
                c_state.close()
                counterfactual_step += 1
            c_state = counterfactual.state[counterfactual_step]
            c_divergent_img = make_img(
                c_state,
                "Counterfactual Trajectory",
                downscale_factor=2,
                active=diverged,
            )
            add_centered_text(c_divergent_img, "DONE", top_padding=100)
        video.append(compose_frames(big_frame, o_divergent_img, c_divergent_img))

        c_state.close()
        state.close()
    o_divergent_state.close()
    return video
