import asyncio
import contextlib
import sys
import time
from types import SimpleNamespace
from typing import Any, ClassVar, Optional, cast
from unittest.mock import Mock

import pytest

from pymobiledevice3.exceptions import CoreDeviceError, ScreenMirrorStopped
from pymobiledevice3.remote.core_device import screen_mirror, vnc_server
from pymobiledevice3.remote.core_device.display_service import (
    MEDIA_IN_USE_ERROR_CODE,
    MEDIA_IN_USE_MESSAGE,
    DisplayService,
)
from pymobiledevice3.remote.core_device.hid_service import (
    DIGITIZER_SURFACE_MAIN_TOUCHSCREEN,
    HID_BUTTON_STATE_DOWN,
    TOUCHSCREEN_STATE_CONTACT,
    TOUCHSCREEN_STATE_RELEASE,
)
from pymobiledevice3.remote.core_device.screen_mirror import (
    STOP_ALL_STREAMS_TIMEOUT,
    STOP_STEP_TIMEOUT,
    ScreenMirrorSession,
)
from pymobiledevice3.remote.core_device.vnc_server import VncStreamServer, _VncClient
from pymobiledevice3.remote.remote_service_discovery import RemoteServiceDiscoveryService


async def _hang() -> None:
    await asyncio.Event().wait()


class _FakeTransport:
    port = 0

    def __init__(self, recv_error: Optional[BaseException] = None) -> None:
        self.closed = 0
        self._recv_error = recv_error

    async def recv(self) -> bytes:
        if self._recv_error is not None:
            raise self._recv_error
        await _hang()
        raise AssertionError("unreachable")

    async def sendto(self, data: bytes, host: str, port: int) -> None:
        pass

    def close(self) -> None:
        self.closed += 1


class _FakeDevice:
    """Records what the session asks of the device-side services."""

    def __init__(self) -> None:
        self.start_video_stream_calls = 0
        self.stop_all_streams_calls = 0
        self.close_calls = 0
        self.transports: list[_FakeTransport] = []
        # Failure injection.
        self.recv_error: Optional[BaseException] = None
        self.open_error: Optional[BaseException] = None
        self.start_error: Optional[BaseException] = None
        self.hang_start = False
        self.hang_close = False
        self.hang_stop_all = False


@pytest.fixture
def device(monkeypatch) -> _FakeDevice:
    fake = _FakeDevice()

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        fake.close_calls += 1
        if fake.hang_close:
            await _hang()

    async def start_video_stream(self, **kwargs):
        fake.start_video_stream_calls += 1
        if fake.hang_start:
            await _hang()
        if fake.start_error is not None:
            raise fake.start_error
        return {"connection": {"streamConfig": {}}}

    async def stop_all_streams(cls, rsd):
        fake.stop_all_streams_calls += 1
        if fake.hang_stop_all:
            await _hang()
        return {"stopped": True}

    def open_media_receiver(svc, rcvbuf_sizes, bind_port=0):
        if fake.open_error is not None:
            raise fake.open_error
        transport = _FakeTransport(fake.recv_error)
        fake.transports.append(transport)
        return transport, "127.0.0.1"

    monkeypatch.setattr(DisplayService, "connect", connect)
    monkeypatch.setattr(DisplayService, "close", close)
    monkeypatch.setattr(DisplayService, "start_video_stream", start_video_stream)
    monkeypatch.setattr(DisplayService, "stop_all_streams", classmethod(stop_all_streams))
    monkeypatch.setattr(screen_mirror, "open_media_receiver", open_media_receiver)
    return fake


@pytest.fixture
def rsd() -> RemoteServiceDiscoveryService:
    return RemoteServiceDiscoveryService(("127.0.0.1", 0))


async def test_start_and_stop_from_code_leave_other_tasks_running(device, rsd):
    other = asyncio.create_task(asyncio.Event().wait())
    session = ScreenMirrorSession(rsd)
    await session.start()
    await session.stop()
    assert not other.done()  # no straggler cancellation in the library class
    assert [t.closed for t in device.transports] == [1]
    other.cancel()


async def test_stop_is_idempotent(device, rsd):
    session = ScreenMirrorSession(rsd)
    await session.start()
    await session.stop()
    await session.stop()
    assert device.stop_all_streams_calls == 1


async def test_stop_before_start_is_a_no_op(device, rsd):
    await ScreenMirrorSession(rsd).stop()
    assert device.stop_all_streams_calls == 0


async def test_on_frame_receives_bgra_and_size(device, rsd):
    got = []
    session = ScreenMirrorSession(rsd, on_frame=lambda b, w, h: got.append((b, w, h)))
    assert session.frame_size == (0, 0)
    assert session.latest_frame is None
    session._transcoder = cast(Any, SimpleNamespace(width=4, height=2))  # what _broadcast_frame reads
    session._broadcast_frame(b"\x00" * 32)
    assert got == [(b"\x00" * 32, 4, 2)]
    assert session.frame_size == (4, 2)
    assert session.latest_frame == b"\x00" * 32
    await asyncio.wait_for(session.wait_first_frame(), timeout=1.0)


async def test_async_with_starts_and_stops(device, rsd):
    async with ScreenMirrorSession(rsd):
        assert device.start_video_stream_calls == 1
    assert device.stop_all_streams_calls == 1


async def test_vnc_server_wakes_its_clients_on_a_frame(device, rsd):
    server = VncStreamServer(rsd)
    assert server._consumer_count() == 0  # no viewer: the decoder-refresh loop idles
    client = _VncClient(Mock(), Mock())
    server._clients.add(client)
    server._transcoder = cast(Any, SimpleNamespace(width=4, height=2))
    server._broadcast_frame(b"\x00" * 32)
    assert client.wants_update.is_set()
    assert server.frame_size == (4, 2)
    assert server._consumer_count() == 1


class _FakeUhs:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    async def send_touchscreen(self, state, x, y, *, service_id=None) -> None:
        assert service_id == DIGITIZER_SURFACE_MAIN_TOUCHSCREEN
        self.calls.append((state, x, y))


class _FakeIndigo:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int]] = []

    async def send_button(self, page, code, state) -> None:
        self.calls.append((page, code, state))


@pytest.fixture
def input_session(rsd) -> ScreenMirrorSession:
    session = ScreenMirrorSession(rsd)
    session._started = True  # input is accepted only while the session is running
    session._fb_width, session._fb_height = 100, 200
    session._uhs = cast(Any, _FakeUhs())
    session._indigo = cast(Any, _FakeIndigo())
    return session


async def test_touch_scales_frame_pixels_to_hid_range(input_session):
    fake = cast(Any, input_session._uhs)
    await input_session.send_touch(0, 0, contact=True)
    await input_session.send_touch(99, 199, contact=False)
    assert fake.calls == [(TOUCHSCREEN_STATE_CONTACT, 0, 0), (TOUCHSCREEN_STATE_RELEASE, 65535, 65535)]


async def test_touch_clamps_out_of_frame_coordinates(input_session):
    await input_session.send_touch(-5, 500, contact=True)
    assert cast(Any, input_session._uhs).calls == [(TOUCHSCREEN_STATE_CONTACT, 0, 65535)]


async def test_touch_before_first_frame_is_ignored(input_session):
    input_session._fb_width = input_session._fb_height = 0
    await input_session.send_touch(10, 10, contact=True)
    assert cast(Any, input_session._uhs).calls == []


async def test_send_button_uses_named_usage(input_session):
    await input_session.send_button("home", down=True)
    assert cast(Any, input_session._indigo).calls == [(0x0C, 0x40, HID_BUTTON_STATE_DOWN)]


async def test_send_button_unknown_name_raises(input_session):
    with pytest.raises(KeyError):
        await input_session.send_button("nope", down=True)


async def test_vnc_pointer_sends_release_only_after_press(input_session, rsd):
    server = VncStreamServer(rsd)
    server._started = True
    server._fb_width, server._fb_height = 100, 200
    server._uhs = input_session._uhs
    client = _VncClient.__new__(_VncClient)
    client.last_button_mask = 0
    client.pressed = False
    await server._handle_pointer(client, 0, 10, 10)
    await server._handle_pointer(client, 1, 0, 0)
    await server._handle_pointer(client, 0, 99, 199)
    assert cast(Any, server._uhs).calls == [
        (TOUCHSCREEN_STATE_CONTACT, 0, 0),
        (TOUCHSCREEN_STATE_RELEASE, 65535, 65535),
    ]


class _FakeHidConnection:
    """Stands in for UniversalHIDServiceService / IndigoHIDService; records connects and closes."""

    instances: ClassVar[list["_FakeHidConnection"]] = []
    connect_gate: ClassVar[Optional[asyncio.Event]] = None

    def __init__(self, rsd) -> None:
        self.connected = False
        self.closed = False
        self.calls: list[tuple[Any, ...]] = []
        _FakeHidConnection.instances.append(self)

    async def connect(self) -> None:
        if _FakeHidConnection.connect_gate is not None:
            await _FakeHidConnection.connect_gate.wait()
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def send_touchscreen(self, *args, **kwargs) -> None:
        self.calls.append(args)

    async def send_button(self, *args) -> None:
        self.calls.append(args)


@pytest.fixture
def hid_connections(monkeypatch) -> list[_FakeHidConnection]:
    _FakeHidConnection.instances = []
    _FakeHidConnection.connect_gate = None
    monkeypatch.setattr(screen_mirror, "UniversalHIDServiceService", _FakeHidConnection)
    monkeypatch.setattr(screen_mirror, "IndigoHIDService", _FakeHidConnection)
    return _FakeHidConnection.instances


async def test_input_after_stop_opens_no_hid_connection(device, rsd, hid_connections):
    session = ScreenMirrorSession(rsd)
    await session.start()
    session._fb_width, session._fb_height = 100, 200
    await session.stop()
    await session.send_touch(10, 10, contact=True)
    await session.send_button("home", down=True)
    assert hid_connections == []


async def test_send_button_before_start_opens_no_hid_connection(device, rsd, hid_connections):
    session = ScreenMirrorSession(rsd)
    await session.send_button("home", down=True)
    session._fb_width, session._fb_height = 100, 200
    await session.send_touch(10, 10, contact=True)
    assert hid_connections == []


@pytest.mark.parametrize("send", ["touch", "button"])
async def test_hid_connect_completing_after_stop_is_closed(device, rsd, hid_connections, send):
    session = ScreenMirrorSession(rsd)
    await session.start()
    session._fb_width, session._fb_height = 100, 200
    gate = _FakeHidConnection.connect_gate = asyncio.Event()
    if send == "touch":
        sending = asyncio.create_task(session.send_touch(10, 10, contact=True))
    else:
        sending = asyncio.create_task(session.send_button("home", down=True))
    while not hid_connections:
        await asyncio.sleep(0)
    await session.stop()
    gate.set()  # the connect finishes only after the teardown
    await asyncio.wait_for(sending, 1)
    assert [(c.connected, c.closed, c.calls) for c in hid_connections] == [(True, True, [])]
    assert session._uhs is None and session._indigo is None


# ----- bounded teardown + failure reporting -----------------------------------


@pytest.fixture
def fast_timeouts(monkeypatch) -> None:
    monkeypatch.setattr(screen_mirror, "STOP_STEP_TIMEOUT", 0.1)
    monkeypatch.setattr(screen_mirror, "STOP_ALL_STREAMS_TIMEOUT", 0.1)


class _HangingHid:
    async def close(self) -> None:
        await _hang()


def test_stop_timeouts_bound_stop_under_ten_seconds():
    assert STOP_STEP_TIMEOUT + STOP_ALL_STREAMS_TIMEOUT < 10


async def test_wait_stopped_reports_feed_failure(device, rsd):
    device.recv_error = OSError("socket closed")
    session = ScreenMirrorSession(rsd)
    await session.start()
    exc = await asyncio.wait_for(session.wait_stopped(), 1)
    assert isinstance(exc, OSError)
    await session.stop()
    assert await asyncio.wait_for(session.wait_stopped(), 1) is exc


async def test_wait_stopped_returns_none_after_stop(device, rsd):
    session = ScreenMirrorSession(rsd)
    await session.start()
    await session.stop()
    assert await asyncio.wait_for(session.wait_stopped(), 1) is None


async def test_wait_stopped_blocks_while_streaming(device, rsd):
    session = ScreenMirrorSession(rsd)
    await session.start()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(session.wait_stopped(), 0.05)
    await session.stop()


async def test_rtcp_send_failure_ends_the_task_with_the_error(rsd, monkeypatch):
    class _BrokenSend(_FakeTransport):
        async def sendto(self, data: bytes, host: str, port: int) -> None:
            raise OSError("network down")

    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda _delay: real_sleep(0))  # skip the 1 s RR cadence
    session = ScreenMirrorSession(rsd)
    session._rtcp_dest = ("::1", 1)
    session._local_ssrc = session._remote_ssrc = 1
    with pytest.raises(OSError, match="network down"):
        await asyncio.wait_for(session._rtcp_send_loop(cast(Any, _BrokenSend())), 1)


async def test_stop_is_bounded_when_device_calls_hang(device, rsd, fast_timeouts):
    device.hang_close = True
    device.hang_stop_all = True
    session = ScreenMirrorSession(rsd)
    await session.start()
    session._uhs = cast(Any, _HangingHid())
    session._indigo = cast(Any, _HangingHid())
    t0 = time.monotonic()
    await session.stop()
    assert time.monotonic() - t0 < 10
    assert device.close_calls == 1
    assert device.stop_all_streams_calls == 1
    assert [t.closed for t in device.transports] == [1]
    assert await asyncio.wait_for(session.wait_stopped(), 1) is None


async def test_stop_during_start_does_not_hang(device, rsd, fast_timeouts):
    device.hang_start = True
    device.hang_close = True
    device.hang_stop_all = True
    session = ScreenMirrorSession(rsd)
    start = asyncio.create_task(session.start())
    await asyncio.sleep(0)
    await asyncio.wait_for(session.stop(), 10)
    with contextlib.suppress(ScreenMirrorStopped):
        await asyncio.wait_for(start, 1)
    assert start.done()
    assert not start.cancelled()
    assert isinstance(start.exception(), ScreenMirrorStopped)
    assert [t.closed for t in device.transports] == [1]
    assert device.close_calls == 1


async def test_stop_during_start_raises_screen_mirror_stopped_and_leaves_the_caller_running(device, rsd):
    device.hang_start = True
    session = ScreenMirrorSession(rsd)

    async def embedder() -> str:
        with pytest.raises(ScreenMirrorStopped):
            await session.start()
        if sys.version_info >= (3, 11):
            task = asyncio.current_task()
            assert task is not None and task.cancelling() == 0
        await asyncio.sleep(0.01)  # the caller's own task is not cancelled: it keeps running
        return "kept running"

    runner = asyncio.create_task(embedder())
    while device.start_video_stream_calls == 0:
        await asyncio.sleep(0)
    await asyncio.wait_for(session.stop(), 2)
    assert await asyncio.wait_for(runner, 1) == "kept running"
    assert [t.closed for t in device.transports] == [1]


async def test_external_cancel_of_start_propagates_cancelled_error(device, rsd):
    device.hang_start = True
    session = ScreenMirrorSession(rsd)
    start = asyncio.create_task(session.start())
    while device.start_video_stream_calls == 0:
        await asyncio.sleep(0)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start, 2)
    assert [t.closed for t in device.transports] == [1]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="needs Task.cancelling() to tell the two apart")
async def test_external_cancel_racing_stop_still_propagates_cancelled_error(device, rsd):
    device.hang_start = True
    session = ScreenMirrorSession(rsd)
    start = asyncio.create_task(session.start())
    while device.start_video_stream_calls == 0:
        await asyncio.sleep(0)
    start.cancel()
    await asyncio.wait_for(session.stop(), 2)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start, 1)


async def test_stop_during_start_whose_device_call_swallows_cancellation_starts_no_tasks(device, rsd, monkeypatch):
    calls = []

    async def stubborn_start_video_stream(self, **kwargs):
        calls.append(kwargs)
        with contextlib.suppress(asyncio.CancelledError):
            await _hang()
        return {"connection": {"streamConfig": {}}}

    monkeypatch.setattr(DisplayService, "start_video_stream", stubborn_start_video_stream)
    session = ScreenMirrorSession(rsd)
    start = asyncio.create_task(session.start())
    while not calls:
        await asyncio.sleep(0)
    await asyncio.wait_for(session.stop(), 2)
    with pytest.raises(ScreenMirrorStopped):
        await asyncio.wait_for(start, 1)
    assert session._feed_task is None and session._refresh_task is None and session._rtcp_task is None
    assert [t.closed for t in device.transports] == [1]


async def test_start_is_single_use(device, rsd):
    session = ScreenMirrorSession(rsd)
    await session.start()
    with pytest.raises(RuntimeError):
        await session.start()
    await session.stop()
    with pytest.raises(RuntimeError):
        await session.start()
    assert device.start_video_stream_calls == 1


async def test_start_after_stop_raises(device, rsd):
    session = ScreenMirrorSession(rsd)
    await session.stop()
    with pytest.raises(RuntimeError):
        await session.start()


async def test_failed_start_closes_what_it_created(device, rsd):
    device.start_error = OSError("tunnel gone")
    session = ScreenMirrorSession(rsd)
    with pytest.raises(OSError, match="tunnel gone"):
        await session.start()
    assert [t.closed for t in device.transports] == [1]
    assert device.close_calls == 1
    assert device.stop_all_streams_calls == 0  # no stream came up: nothing to release
    assert await asyncio.wait_for(session.wait_stopped(), 1) is None
    await session.stop()  # no-op after a failed start
    assert device.close_calls == 1


async def test_failed_receiver_open_closes_the_service(device, rsd):
    device.open_error = OSError("no socket")
    session = ScreenMirrorSession(rsd)
    with pytest.raises(OSError, match="no socket"):
        await session.start()
    assert device.close_calls == 1


async def test_media_in_use_is_still_reported_cleanly(device, rsd):
    device.start_error = CoreDeviceError("busy", code=MEDIA_IN_USE_ERROR_CODE)
    session = ScreenMirrorSession(rsd)
    with pytest.raises(CoreDeviceError) as info:
        await session.start()
    assert str(info.value) == MEDIA_IN_USE_MESSAGE
    assert [t.closed for t in device.transports] == [1]
    assert device.close_calls == 1


async def test_stop_is_bounded_when_a_device_call_ignores_cancellation(device, rsd, fast_timeouts, monkeypatch):
    release = asyncio.Event()

    async def stubborn_close(self) -> None:
        device.close_calls += 1
        while not release.is_set():
            with contextlib.suppress(asyncio.CancelledError):
                await release.wait()

    monkeypatch.setattr(DisplayService, "close", stubborn_close)
    session = ScreenMirrorSession(rsd)
    await session.start()
    try:
        await asyncio.wait_for(session.stop(), 2)
        assert [t.closed for t in device.transports] == [1]
        assert device.stop_all_streams_calls == 1
    finally:
        release.set()
        await asyncio.sleep(0)


async def test_concurrent_stops_both_wait_for_the_teardown(device, rsd, fast_timeouts):
    device.hang_stop_all = True
    session = ScreenMirrorSession(rsd)
    await session.start()
    first = asyncio.create_task(session.stop())
    second = asyncio.create_task(session.stop())
    await asyncio.wait_for(second, 2)
    assert first.done()  # the second stop() returned only once the one teardown had finished
    assert device.stop_all_streams_calls == 1
    assert [t.closed for t in device.transports] == [1]
    assert await asyncio.wait_for(session.wait_stopped(), 0.01) is None


async def test_stop_during_a_failed_start_waits_for_its_teardown(device, rsd, fast_timeouts):
    device.start_error = OSError("tunnel gone")
    device.hang_close = True  # keeps the failed start() inside its bounded teardown
    session = ScreenMirrorSession(rsd)
    start = asyncio.create_task(session.start())
    while device.close_calls == 0:
        await asyncio.sleep(0)
    await asyncio.wait_for(session.stop(), 2)
    assert start.done()
    assert isinstance(start.exception(), OSError)
    assert [t.closed for t in device.transports] == [1]
    assert await asyncio.wait_for(session.wait_stopped(), 0.01) is None


async def test_cancelled_stop_still_closes_the_transport(device, rsd):
    device.hang_close = True  # keeps stop() inside its bounded close step (3 s)
    session = ScreenMirrorSession(rsd)
    await session.start()
    stopping = asyncio.create_task(session.stop())
    while device.close_calls == 0:
        await asyncio.sleep(0)
    stopping.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stopping, 1)
    assert [t.closed for t in device.transports] == [1]


async def test_serve_vnc_releases_the_stream_when_it_cannot_listen(device, rsd, monkeypatch):
    async def cannot_listen(*args, **kwargs):
        raise OSError("address already in use")

    monkeypatch.setattr(vnc_server.asyncio, "start_server", cannot_listen)
    server = VncStreamServer(rsd, port=0)
    with pytest.raises(OSError, match="address already in use"):
        await asyncio.wait_for(server.serve(), 2)
    assert [t.closed for t in device.transports] == [1]
    assert device.stop_all_streams_calls == 1
    assert await asyncio.wait_for(server.wait_stopped(), 0.01) is None
