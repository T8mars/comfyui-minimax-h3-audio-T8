"""FinalLayer delegation never guesses a protocol from foreign marker names."""
import pytest

from h3_audio_t8_pkg.h3_core_compat import call_h3_final_layer


def test_foreign_marker_and_kwargs_do_not_get_an_invented_schedule_or_retry():
    calls = []

    def original(x, t, video, audio, sigma, sample_sigmas, shifts):
        raise AssertionError("The adapter must not bypass a user-selected wrapper")

    def user(x, t, video, audio, *args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("user body failed once")

    user._t8_hyperflow_owner = "copied-marker"
    user._t8_hyperflow_inner = original

    class UserLayer:
        forward = staticmethod(user)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    with pytest.raises(TypeError, match="user body failed once"):
        call_h3_final_layer(UserLayer(), 1, 2, (0, 1, 0), (1, 2, 0),
            sigma=.5, sample_sigmas=[1., .5, 0.], shifts=(12., 3.))
    assert calls == [((), {})]


def test_real_partial_schedule_interface_fails_before_forward():
    calls = []

    class PartialLayer:
        def __call__(self, x, t, video, audio, sigma):
            calls.append(True)

    with pytest.raises(RuntimeError, match="partial H3 FinalLayer schedule"):
        call_h3_final_layer(PartialLayer(), 1, 2, (0, 1, 0), (1, 2, 0),
            sigma=.5, sample_sigmas=[1., .5, 0.], shifts=(12., 3.))
    assert calls == []
