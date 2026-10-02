import numpy as np

from padel_analysis.geometry.calibration import Calibration
from padel_analysis.geometry.camera import pose_from_calibration
from padel_analysis.render import figure_style as style
from padel_analysis.render.court_drawing import court_backdrop

POSE = pose_from_calibration(Calibration.load("ground_truth/calibrations/FinalF.json").points,
                             (1920, 1080))


def _bgr(colour):
    return np.array([int(colour[i:i + 2], 16) for i in (5, 3, 1)])


def test_the_backdrop_is_the_size_of_the_video():
    assert court_backdrop(POSE, (1920, 1080)).shape == (1080, 1920, 3)


def test_the_floor_is_blue_where_the_court_projects_and_dark_outside():
    picture = court_backdrop(POSE, (1920, 1080)).astype(int)
    x, y = POSE.project(np.array([[2.5, -4.0, 0.0]]))[0]
    floor = picture[int(y), int(x)]
    assert np.abs(floor - _bgr(style.COURT_BLUE)).max() < 30
    assert np.abs(picture[5, 5] - _bgr(style.BACKGROUND)).max() < 5
