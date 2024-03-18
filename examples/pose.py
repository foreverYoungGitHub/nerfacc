from typing import Dict, List, Optional, Tuple, Union
from functools import cached_property
import numpy as np
from scipy.spatial.transform import Rotation
import collections


class Pose:
    def __init__(self, r=None, t=None, covar=None):
        # rotation
        if r is None:
            r = [1, 0, 0, 0]
        if isinstance(r, (list, np.ndarray, np.generic)):
            if isinstance(r, list):
                r = np.array(r, dtype=float)
            if r.shape == (4,):
                qvec_scipy = r[[1, 2, 3, 0]]
                r = Rotation.from_quat(qvec_scipy)
            elif r.shape == (3, 3):
                r = Rotation.from_matrix(r)
            else:
                raise ValueError(f'Invalid rotation: {r}')
        elif not isinstance(r, Rotation):
            raise ValueError(f'Unknown rotation format: {r}')

        # translation
        if t is None:
            t = [0, 0, 0]
        if isinstance(t, list):
            t = np.array(t, dtype=float)
        elif not isinstance(t, (np.ndarray, np.generic)):
            raise ValueError(f'Unknown translation format: {t}')
        if t.shape != (3,) or not np.all(np.isfinite(t)):
            raise ValueError(f'Invalid translation: {t}')

        if covar is not None and covar.shape != (6, 6):
            raise ValueError(f'Invalid covariance: {covar}')

        self._r = r
        self._t = t
        self._covar = covar

    def to_list(self) -> List[str]:
        data = [self.qvec, self.t]
        if self.covar is not None:
            data.append(self.covar.flatten())
        return np.concatenate(data).astype(str).tolist()

    @classmethod
    def from_list(cls, qt: List[str]) -> 'Pose':
        qw, qx, qy, qz, tx, ty, tz, *covar = qt
        if len(covar) == 0:
            covar = None
        elif len(covar) == 36:
            covar = np.reshape(np.array(covar, float), (6, 6))
        else:
            raise ValueError(
                'Invalid format. Expected: [qw, qx, qy, qz, tx, ty, tz] '
                f'or [qw, qx, qy, qz, tx, ty, tz, covar_6x6]; '
                f'Obtained: {qt}')
        return Pose([qw, qx, qy, qz], [tx, ty, tz], covar=covar)

    @classmethod
    def from_4x4mat(cls, T) -> 'Pose':
        if isinstance(T, list):
            T = np.array(T, dtype=float)
        elif not isinstance(T, (np.ndarray, np.generic)):
            raise ValueError(f'Unknown type for 4x4 transformation matrix: {T}')
        if T.shape != (4, 4):
            raise ValueError(f'Invalid 4x4 transformation matrix: {T}')
        return Pose(T[:3, :3], T[:3, 3])

    @property
    def r(self) -> Rotation:
        return self._r

    @cached_property
    def R(self) -> np.ndarray:
        return self.r.as_matrix()

    @property
    def qvec(self) -> np.ndarray:
        qvec_scipy = self._r.as_quat()
        qvec = qvec_scipy[[3, 0, 1, 2]]
        return qvec

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def covar(self) -> np.ndarray:
        return self._covar

    @property
    def qt(self) -> Tuple[np.ndarray]:
        return (self.qvec, self.t)

    @cached_property
    def adjoint(self) -> np.ndarray:
        tx, ty, tz = self.t
        skew_t = np.array([[0, -tz, ty],
                           [tz, 0, -tx],
                           [-ty, tx, 0]])
        return np.block([[self.R, np.zeros((3, 3))], [skew_t@self.R, self.R]])

    def to_4x4mat(self) -> np.ndarray:
        T = np.hstack((self.R, self.t[:, None]))
        T = np.vstack((T, (0, 0, 0, 1)))
        return T

    @cached_property
    def inv(self) -> 'Pose':
        r_inv = self.r.inv()
        rotmat_inv = r_inv.as_matrix()
        t_inv = -1.0 * (rotmat_inv @ self.t)
        covar = self.covar
        if covar is not None:
            # here we assume that the noise is applied on the right
            covar = self.adjoint @ covar @ self.adjoint.T
        return Pose(r_inv, t_inv, covar)

    def inverse(self) -> 'Pose':
        return self.inv

    def __mul__(self, other) -> 'Pose':
        if not isinstance(other, self.__class__):
            return NotImplemented
        r_new = self.r * other.r
        t_new = self.t + self.R @ other.t
        covar = other.covar
        if self.covar is not None:
            adj = other.inv.adjoint
            covar = (0 if covar is None else covar) + adj @ self.covar @ adj.T
        return Pose(r_new, t_new, covar)

    def transform_points(self, points3d: np.ndarray) -> np.ndarray:
        if points3d.shape[-1] != 3:
            raise ValueError(f'Points must be in shape (..., 3): {points3d.shape}')
        return points3d @ self.R.transpose() + self.t

    def magnitude(self) -> Tuple[float, float]:
        dr = np.rad2deg(self.r.magnitude())
        dt = np.linalg.norm(self.t)
        return dr, dt
    
    def clone(self) -> 'Pose':
        return Pose(self.r, self.t, self.covar)

    def __repr__(self) -> str:
        return 'q:{},  t:{}'.format(self.qvec, self.t)

def to_homogeneous(p: np.ndarray) -> np.ndarray:
    return np.pad(p, ((0, 0),)*(p.ndim-1) + ((0, 1),), constant_values=1)

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

# Details at https://github.com/colmap/colmap/blob/master/src/base/camera_models.h.
CAMERA_MODEL_PARAM_NAMES = {
    'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
    'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
    'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
    'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
    'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
    'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
    'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
    'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
    'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
    'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
    'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
}

# TODO: add method to (un)distort points, get camera matrix
class Camera:
    def __init__(self,
                 camera_model: Union[CameraModel, str],
                 camera_params: List,
                 name: Optional[str] = None,
                 sensor_type: str = 'camera'):
        if isinstance(camera_model, str):
            camera_model = CAMERA_MODEL_NAMES[camera_model]
        # check params are consistent with model
        assert isinstance(camera_params, list)
        assert len(camera_params) == camera_model.num_params + 2  # extra width and height
        assert sensor_type in ['camera', 'depth']
        camera_params = list(map(str, camera_params))  # convert to str before next checks
        assert camera_params[0].isnumeric()  # width
        assert camera_params[1].isnumeric()  # height
        assert len(camera_params[2:]) == len(CAMERA_MODEL_PARAM_NAMES[camera_model.model_name])

        # make sure it crashes if camera_params cannot be cast to float, store as string
        camera_params = [float(v) for v in camera_params]
        camera_params = [str(int(v)) if v.is_integer() else str(v) for v in camera_params]
        sensor_params = [camera_model.model_name] + camera_params
        self._sensor_type = sensor_type
        self.sensor_params = sensor_params

    @cached_property
    def model_name(self) -> str:
        return self.sensor_params[0]

    @cached_property
    def model(self) -> CameraModel:
        return CAMERA_MODEL_NAMES[self.sensor_params[0]]

    @cached_property
    def width(self) -> int:
        return int(self.sensor_params[1])

    @cached_property
    def height(self) -> int:
        return int(self.sensor_params[2])

    @cached_property
    def params(self) -> List[float]:
        return [float(p) for p in self.sensor_params[3:]]

    @cached_property
    def params_dict(self) -> Dict[str, float]:
        return dict(zip(CAMERA_MODEL_PARAM_NAMES[self.model_name],
                        self.params))

    @cached_property
    def projection_params(self) -> List[float]:
        if self.model_name in {'SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL',
                               'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'}:
            f, cx, cy, *_ = self.params
            fx = fy = f
        elif self.model_name in {'PINHOLE', 'OPENCV', 'OPENCV_FISHEYE',
                                 'FULL_OPENCV', 'FOV', 'THIN_PRISM_FISHEYE'}:
            fx, fy, cx, cy, *_ = self.params
        else:
            raise ValueError('Unsupported camera type.')
        return [fx, fy, cx, cy]

    @cached_property
    def f(self) -> np.ndarray:
        return np.array(self.projection_params[:2])

    @cached_property
    def c(self) -> np.ndarray:
        return np.array(self.projection_params[2:])

    @cached_property
    def size(self) -> np.ndarray:
        return np.array([self.width, self.height])

    @property
    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.projection_params
        return np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])

    @property
    def asdict(self) -> dict:
        return {
            'model': self.model_name,
            'width': self.width,
            'height': self.height,
            'params': self.params
        }

    def world2image(self, pts: np.ndarray) -> np.ndarray:
        if self.model_name not in {'SIMPLE_PINHOLE', 'PINHOLE'}:
            raise ValueError('Unsupported camera type.')
        return (pts * self.f) + self.c

    def image2world(self, pts: np.ndarray) -> np.ndarray:
        if self.model_name not in {'SIMPLE_PINHOLE', 'PINHOLE'}:
            raise ValueError('Unsupported camera type.')
        return (pts - self.c) / self.f

    def in_image(self, pts: np.ndarray):
        return np.all((pts >= 0) & (pts <= (self.size - 1)), -1)
