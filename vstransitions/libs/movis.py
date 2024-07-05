from __future__ import annotations

from fractions import Fraction
from threading import Lock
from typing import Any

import numpy as np
from vspyplugin import PyPluginNumpy
from vstools import FramesCache, KwargsT, cachedproperty, core, vs

from movis.layer.composition import Composition, LayerItem


class MovisVSVideoSource:
    def __init__(
        self, video_node: vs.VideoNode | tuple[vs.VideoNode, vs.VideoNode | bool],
        audio: vs.AudioNode | None = None, **kwargs: Any
    ) -> None:
        video_alpha_node = True

        if isinstance(video_node, tuple):
            video_node, video_alpha_node = video_node

        if not isinstance(video_alpha_node, vs.VideoNode):
            try:
                if not video_alpha_node:
                    raise TypeError

                video_alpha_node = video_node.std.PropToClip('_Alpha')
            except Exception:
                video_alpha_node = core.std.BlankClip(video_node, format=vs.GRAY8, color=255, keep=True)

        self.video_node = video_node.resize.Bicubic(format=vs.RGB24, **kwargs)
        self.video_alpha_node = video_alpha_node.resize.Bicubic(format=vs.GRAY8)

        self.video_node.std.SetVideoCache(True)
        self.video_alpha_node.std.SetVideoCache(True)

        self._audio = audio
        self._audio_layer = MovisVSAudioSource(audio) if audio else None

        self._video_cache = FramesCache(self.video_node, core.num_threads * 2)
        self._video_alpha_cache = FramesCache(self.video_alpha_node, core.num_threads * 2)

    def __getstate__(self) -> KwargsT:
        return self.__dict__.copy()

    @property
    def fps(self) -> float:
        """The frame rate of the video."""
        return float(self.video_node.fps)

    @property
    def size(self) -> tuple[int, int]:
        """The size of the video with a tuple of ``(width, height)``."""
        return (self.video_node.width, self.video_node.height)

    @property
    def n_frame(self) -> int:
        """The number of frames in the video."""
        return self.video_node.num_frames

    @property
    def duration(self) -> float:
        """The duration of the video."""
        # FIXME: ASSUMING CFR
        return float(self.video_node.num_frames / self.video_node.fps)

    def has_audio(self) -> bool:
        """Return True if the video has audio layer."""
        return self._audio_layer is not None

    @property
    def audio(self) -> bool:
        """Whether the video has audio data."""
        return self._audio is not None

    def get_key(self, time: float) -> int:
        """Get the state index for the given time."""
        if time < 0 or self.duration < time:
            return -1

        return int(time * self.video_node.fps)

    def __call__(self, time: float) -> np.ndarray | None:
        frame_index = self.get_key(time)

        if frame_index < 0 or frame_index >= self.n_frame:
            return None

        frame, alpha_frame = (self._video_cache[frame_index], self._video_alpha_cache[frame_index])

        return np.dstack([
            *(
                np.asarray(arr, np.uint8)
                for arr in (frame[0], frame[1], frame[2])
            ),
            np.asarray(alpha_frame[0], np.uint8)
        ])

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        if self._audio and self._audio_layer is not None:
            return self._audio_layer.get_audio(start_time, end_time)
        return None


class MovisVSAudioSource:
    def __init__(self, audio_node: vs.AudioNode) -> None:
        raise NotImplementedError

    def get_audio(self, start_time: float, end_time: float) -> np.ndarray | None:
        raise NotImplementedError


class MovisSceneVSWrap(cachedproperty.baseclass):
    def __init__(self, scene: Composition, fps: Fraction | None = None) -> None:
        self.scene = scene
        self.scene._cache = {}
        self.width = self.scene.size[0]
        self.height = self.scene.size[1]

        self.mutex = Lock()

        if fps is None:
            for layer in self.scene._layers:
                if isinstance(layer, LayerItem):
                    layer = layer.layer

                if isinstance(layer, MovisVSVideoSource):
                    self.fps = layer.video_node.fps
                    break
            else:
                layer = self.scene._layers[0].layer

                if isinstance(layer, LayerItem):
                    layer = layer.layer

                self.fps = Fraction(layer.fps)
        else:
            self.fps = fps

        self._base_clip = core.std.BlankClip(
            None, self.width, self.height, vs.RGB24, self.scene.duration * self.fps,
            self.fps.numerator, self.fps.denominator, [0, 0, 0], True
        )

    def _to_node(self, n: int, plane: int, dst: PyPluginNumpy.DT) -> None:
        frame = self.scene(float(n / self.fps))

        if frame is None:
            raise KeyError

        np.copyto(dst, frame[:, :, plane])

        with self.mutex:
            if len(self.scene._cache) > core.num_threads * 2:
                for k in list(self.scene._cache.keys())[:core.num_threads]:
                    del self.scene._cache[k]

    @cachedproperty
    def video(self) -> vs.VideoNode:
        return PyPluginNumpy(self._base_clip, output_per_plane=True)(self._to_node)

    @cachedproperty
    def alpha(self) -> vs.VideoNode:
        alpha_clip = self._base_clip.std.BlankClip(format=vs.GRAY8, color=255)

        @PyPluginNumpy(alpha_clip)
        def node(n: int, dst: PyPluginNumpy.DT) -> None:
            self._to_node(n, 3, dst)

        return node
