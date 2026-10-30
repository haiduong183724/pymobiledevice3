import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from pymobiledevice3.remote.core_device import screen_mirror
from pymobiledevice3.remote.core_device.display_service import DisplayService
from pymobiledevice3.remote.core_device.hid_service import (
    DIGITIZER_SURFACE_MAIN_TOUCHSCREEN,
    HID_BUTTON_STATE_DOWN,
    TOUCHSCREEN_STATE_CONTACT,
    TOUCHSCREEN_STATE_RELEASE,
)
from pymobiledevice3.remote.core_device.screen_mirror import ScreenMirrorSession
from pymobiledevice3.remote.core_device.vnc_server import VncStreamServer, _VncClient
from pymobiledevice3.remote.remote_service_discovery import RemoteServiceDiscoveryService


class _FakeTransport:
    port = 0

    def __init__(self) -> None:
        self.closed = 0

    async def recv(self) -> bytes:
        await asyncio.Event().wait()
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
        self.transports: list[_FakeTransport] = []


@pytest.fixture
def device(monkeypatch) -> _FakeDevice:
    fake = _FakeDevice()

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def start_video_stream(self, **kwargs):
        fake.start_video_stream_calls += 1
        return {"connection": {"streamConfig": {}}}

    async def stop_all_streams(cls, rsd):
        fake.stop_all_streams_calls += 1
        return {"stopped": True}

    def open_media_receiver(svc, rcvbuf_sizes, bind_port=0):
        transport = _FakeTransport()
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
